"""Targeted fixed-cohort precision check for S6 age 40-49, both doses."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

K = None
CFG = None
WEEKS = None

def initialize(candidate, weeks):
    global K, CFG, WEEKS
    os.environ.update(DATASET="Czech", KCOR_NPH_MODE="identified_only")
    sys.path.insert(0, str(Path(candidate)/"code"))
    import KCOR as module
    K = module
    CFG = K.get_gompertz_theta_fit_config()
    WEEKS = weeks

def estimate(counts):
    risk, dead = counts
    t = np.arange(len(risk), dtype=float)
    skip = K.DYNAMIC_HVE_SKIP_WEEKS
    h = K.hazard_from_mr(dead/risk)
    eff = np.where(t >= skip, h, 0.)
    quiet, _, _ = K._build_theta_quiet_mask(WEEKS, t)
    (k, theta), diag = K.fit_theta0_gompertz(eff, t-skip, quiet,
        CFG["k_anchor_weeks"], CFG["gamma_per_week"], deaths_arr=dead, risk_arr=risk)
    applied, _, _ = K._apply_theta_degenerate_guard(theta, K.get_theta_degenerate_fit_config())
    H = np.cumsum(eff)
    inv = K.invert_gamma_frailty(H, applied)
    anchor = K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    return dict(k=float(k), theta=float(applied), theta_raw=float(diag.get("theta0_raw",theta)),
                status=diag["status"], upper_bound=bool(diag.get("theta_hit_bound",False)),
                count_flag=bool(diag.get("sampling_flag",False)), growth=float(inv[-1]/inv[anchor]),
                fit_mask=np.asarray(diag["fit_mask_theta"],bool).tolist())

def worker(counts):
    a,b=estimate(counts[0]),estimate(counts[1])
    return a,b,b["growth"]/a["growth"]

def theoretical(risk, dead, fit_mask, theta, k=None):
    skip=K.DYNAMIC_HVE_SKIP_WEEKS
    risk,dead,quiet=risk[skip:],dead[skip:],np.asarray(fit_mask)[skip:]
    gamma=CFG["gamma_per_week"]
    F=np.expm1(gamma*np.arange(len(risk)+1))/gamma
    h=K.hazard_from_mr(dead/risk)
    H=np.r_[0.,np.cumsum(h)]
    inc=np.diff(F)
    if k is None:
        def nll(log_k):
            q=np.log1p(theta*np.exp(log_k)*inc[quiet]*np.exp(-theta*H[:-1][quiet]))/theta
            return -np.sum(dead[quiet]*np.log(-np.expm1(-q))-(risk[quiet]-dead[quiet])*q)
        k=np.exp(minimize_scalar(nll,bounds=(-20,-1),method="bounded").x)
    ix=np.flatnonzero(quiet)
    segments=np.split(ix,np.where(np.diff(ix)>1)[0]+1)
    intervals=[];offsets=[];prev_end=0;prev_segment=-1
    for j,seg in enumerate(segments):
        start,end=int(seg[0]),int(seg[-1]+1)
        if start>prev_end:
            prior=k*F[prev_end]+(offsets[prev_segment] if prev_segment>=0 else 0.)
            prior_h=np.log1p(theta*prior)/theta
            new_exposure=np.expm1(theta*(prior_h+H[start]-H[prev_end]))/theta
            offsets.append(new_exposure-k*F[start])
            intervals.append((prev_end,start,prev_segment,j))
        else:offsets.append(0.)
        intervals.extend((i,i+1,j,j) for i in seg)
        prev_end,prev_segment=end,j
    starts,ends,from_seg,to_seg=map(np.array,zip(*intervals))
    offsets=np.array(offsets)
    free=np.flatnonzero([seg[0]>0 for seg in segments])
    parameters=np.r_[np.log(k),theta,offsets[free]]
    def predict(p):
        rate,th=np.exp(p[0]),p[1]
        a=offsets.copy();a[free]=p[2:];a=np.r_[a,0.]
        x=rate*F[starts]+a[from_seg];y=rate*F[ends]+a[to_seg]
        return np.log1p(th*(y-x)/(1+th*x))/th
    q=predict(parameters);p=-np.expm1(-q)
    assert np.all(p>0)
    derivatives=[]
    for j in range(len(parameters)):
        step=1e-5 if j<2 else 1e-7
        z=parameters.copy();z[j]+=step
        w=parameters.copy();w[j]-=step
        derivatives.append((predict(z)-predict(w))/(2*step))
    J=np.column_stack(derivatives)*np.sqrt(risk[starts]*(1-p)/p)[:,None]
    signal=J[:,1];nuisance=J[:,[0]+list(range(2,J.shape[1]))]
    nuisance=nuisance/np.linalg.norm(nuisance,axis=0)
    signal=signal-nuisance@np.linalg.lstsq(nuisance,signal,rcond=1e-8)[0]
    se=float(1/np.linalg.norm(signal))
    return dict(theta_reference=theta,k_reference=k,se=se,half_width_95=1.96*se,
                required_theta_for_50pct_local_precision=1.96*se/.5,
                required_theta_for_25pct_local_precision=1.96*se/.25)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate",type=Path)
    ap.add_argument("--draws",type=int,default=999)
    ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args()
    source=args.candidate/"data/Czech/KCOR_CMR.xlsx"
    raw=pd.read_excel(source,sheet_name="2021_24",usecols=["YearOfBirth","Dose","ISOweekDied","Alive","Dead"])
    raw["week"]=raw.ISOweekDied.astype(str).str.replace("-","",regex=False).str[:6].astype(int)
    raw=raw[raw.YearOfBirth.isin([1975,1980])&raw.Dose.isin([0,2])&raw.week.between(202124,202416)]
    groups=raw.groupby(["YearOfBirth","Dose","week"],as_index=False)[["Alive","Dead"]].sum()
    weeks=np.sort(groups.week.unique());assert len(weeks)==149
    initialize(args.candidate,weeks)
    rng=np.random.default_rng(20261126)
    observed=np.zeros((2,2,len(weeks)),dtype=np.int64)
    draws=np.zeros((args.draws,2,2,len(weeks)),dtype=np.int64)
    for j,dose in enumerate((0,2)):
        for yob in (1975,1980):
            x=groups[(groups.YearOfBirth==yob)&(groups.Dose==dose)].sort_values("week")
            assert np.array_equal(x.week, weeks)
            a,d=x.Alive.to_numpy(int),x.Dead.to_numpy(int)
            exits=a[:-1]-d[:-1]-a[1:]
            # Verified death-only histories; categories are week of death or survivor.
            assert np.all(exits==0) and a[-1]>=d[-1]
            cells=np.r_[d,a[-1]-d[-1]]
            assert cells.sum()==a[0]
            boot=rng.multinomial(a[0],cells/a[0],size=args.draws)
            observed[j]+=[a,d]
            draws[:,j,0]+=np.cumsum(boot[:,::-1],axis=1)[:,::-1][:,:-1]
            draws[:,j,1]+=boot[:,:-1]
    a,b,ratio=worker(observed)
    point=[a,b]
    for j,dose in enumerate((0,2)):
        risk,dead=observed[j];mask=np.array(point[j]["fit_mask"])
        point[j].update(dose=dose,enrollment_people=int(risk[0]),first_fit_risk=int(risk[mask][0]),
                        all_deaths=int(dead.sum()),fit_deaths=int(dead[mask].sum()),fit_weeks=int(mask.sum()),
                        local_information_low=theoretical(risk,dead,mask,.5),
                        local_information_fitted=theoretical(risk,dead,mask,point[j]["theta_raw"],point[j]["k"]))
    print("POINT",json.dumps(point), "KCOR",ratio,flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.candidate,weeks)) as pool:
        rows=[]
        for i,v in enumerate(pool.map(worker,draws,chunksize=8),1):
            rows.append(v)
            if i%100==0:print("completed",i,flush=True)
    def percentiles(v):return np.percentile(v,[2.5,50,97.5]).tolist()
    intervals=[]
    for j,dose in enumerate((0,2)):
        values=[v[j] for v in rows]
        intervals.append(dict(dose=dose,theta=percentiles([x["theta"] for x in values]),
            theta_raw=percentiles([x["theta_raw"] for x in values]),
            upper_bound=sum(x["upper_bound"] for x in values),
            near_zero_applied=sum(x["theta"]<1e-6 for x in values),
            near_zero_raw=sum(x["theta_raw"]<1e-6 for x in values),
            count_flag=sum(x["count_flag"] for x in values),
            status_counts=pd.Series([x["status"] for x in values]).value_counts().to_dict()))
    result=dict(draws=args.draws,seed=20261126,point=point,intervals=intervals,
                anchored_kcor=ratio,anchored_kcor_interval=percentiles([r[2] for r in rows]))
    output=Path(__file__).resolve().parent
    (output/"frailty-resolution.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    pd.DataFrame([dict(draw=i,theta0=a["theta"],theta2=b["theta"],theta0_raw=a["theta_raw"],theta2_raw=b["theta_raw"],kcor=v) for i,(a,b,v) in enumerate(rows)]).to_csv(output/"frailty-resolution-draws.csv",index=False)
    print(json.dumps({k:v for k,v in result.items() if k!="point"},indent=2))

if __name__=="__main__":main()

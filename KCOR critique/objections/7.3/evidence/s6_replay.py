"""Matched multi-window and first-window precision check for all S6 age groups."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("precision",HERE.parent/"s6/frailty-resolution.py")
precision=importlib.util.module_from_spec(spec);spec.loader.exec_module(precision)
BANDS=[1975,1965,1955,1945,1935,1925,-2]

def initialize(candidate, weeks):
    precision.initialize(candidate, weeks)

def estimate(counts, first_only=False):
    K=precision.K;cfg=precision.CFG;weeks=precision.WEEKS
    risk,dead=counts
    t=np.arange(len(risk),dtype=float);skip=K.DYNAMIC_HVE_SKIP_WEEKS
    h=K.hazard_from_mr(dead/risk);eff=np.where(t>=skip,h,0.)
    quiet,_,_=K._build_theta_quiet_mask(weeks,t)
    if first_only:
        indices=np.flatnonzero(quiet & (t>=skip))
        first=np.split(indices,np.where(np.diff(indices)>1)[0]+1)[0]
        quiet=np.isin(t,first)
    (k,th),diag=K.fit_theta0_gompertz(eff,t-skip,quiet,cfg["k_anchor_weeks"],cfg["gamma_per_week"],
                                      deaths_arr=dead,risk_arr=risk)
    theta=th if diag.get("success") and np.isfinite(th) and th>=0 else 0.
    theta,_,_=K._apply_theta_degenerate_guard(theta,K.get_theta_degenerate_fit_config())
    inv=K.invert_gamma_frailty(np.cumsum(eff),theta)
    anchor=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    return dict(k=float(k),theta=float(theta),theta_raw=float(diag.get("theta0_raw",th)),
                status=diag["status"],upper_bound=bool(diag.get("theta_hit_bound",False)),
                residual_flag=bool(diag.get("sampling_flag",False)),growth=float(inv[-1]/inv[anchor]),
                fit_mask=np.asarray(diag.get("fit_mask_theta",np.zeros(len(t),bool)),bool).tolist())

def worker(task):
    band,draw,counts=task
    rows=[]
    for mode in ("all",):
        a,b=[estimate(c,mode=="first") for c in counts]
        rows.append(dict(birth_band=band,draw=draw,mode=mode,
          theta0=a["theta"],theta2=b["theta"],theta0_raw=a["theta_raw"],theta2_raw=b["theta_raw"],
          bound0=a["upper_bound"],bound2=b["upper_bound"],status0=a["status"],status2=b["status"],
          residual0=a["residual_flag"],residual2=b["residual_flag"],kcor=b["growth"]/a["growth"]))
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate",type=Path);parser.add_argument("--draws",type=int,default=999)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    source=args.candidate/"data/Czech/KCOR_CMR.xlsx"
    raw=pd.read_excel(source,sheet_name="2021_24",usecols=["YearOfBirth","Dose","ISOweekDied","Alive","Dead"])
    raw["week"]=raw.ISOweekDied.astype(str).str.replace("-","",regex=False).str[:6].astype(int)
    raw=raw[raw.Dose.isin([0,2])&raw.week.between(202124,202416)]
    groups=raw.groupby(["YearOfBirth","Dose","week"],as_index=False)[["Alive","Dead"]].sum()
    weeks=np.sort(groups.week.unique());assert len(weeks)==149
    precision.initialize(args.candidate,weeks)
    rng=np.random.default_rng(20261126)
    component_keys=[(yob,dose) for dose in (0,2) for yob in (1975,1980)]
    component_keys += [(int(yob),int(dose)) for (yob,dose),_ in groups.groupby(["YearOfBirth","Dose"])
                       if (yob>0 or yob==-1) and (yob,dose) not in component_keys]
    observed=np.zeros((len(BANDS),2,2,len(weeks)),dtype=np.int64)
    sampled=np.zeros((len(BANDS),args.draws,2,2,len(weeks)),dtype=np.int64)
    for yob,dose in component_keys:
        x=groups[(groups.YearOfBirth==yob)&(groups.Dose==dose)].sort_values("week")
        assert np.array_equal(x.week, weeks)
        risk,dead=x.Alive.to_numpy(int),x.Dead.to_numpy(int)
        exits=risk[:-1]-dead[:-1]-risk[1:]
        assert np.all(exits==0) and risk[-1]>=dead[-1]
        cells=np.r_[dead,risk[-1]-dead[-1]];assert cells.sum()==risk[0]
        if risk[0]: counts=rng.multinomial(risk[0],cells/risk[0],size=args.draws)
        else: counts=np.zeros((args.draws,len(cells)),dtype=np.int64)
        risk_draw=np.cumsum(counts[:,::-1],axis=1)[:,::-1][:,:-1]
        dead_draw=counts[:,:-1]
        targets=[BANDS.index(-2)]
        if yob>0:
            band=(yob-5)//10*10+5
            if band in BANDS:targets.append(BANDS.index(band))
        j=(0,2).index(dose)
        for i in targets:
            observed[i,j]+=[risk,dead]
            sampled[i,:,j,0]+=risk_draw;sampled[i,:,j,1]+=dead_draw
    for j,dose in enumerate((0,2)):
        total=groups[(groups.YearOfBirth==-2)&(groups.Dose==dose)].sort_values("week")
        assert np.array_equal(total[["Alive","Dead"]].to_numpy().T,observed[-1,j])
    points=[]
    saved=json.loads((args.candidate/"analysis/s6/results.json").read_text(encoding="utf-8"))
    for i,band in enumerate(BANDS):
        for mode in ("all",):
            values=[estimate(c,mode=="first") for c in observed[i]]
            for j,dose in enumerate((0,2)):
                risk,dead=observed[i,j];v=values[j];mask=np.array(v["fit_mask"])
                item=dict(birth_band=band,dose=dose,mode=mode,**v,enrollment_people=int(risk[0]),
                          fit_deaths=int(dead[mask].sum()),fit_weeks=int(mask.sum()),total_deaths=int(dead.sum()))
                if mask.sum()>2:
                    item["information_low"]=precision.theoretical(risk,dead,mask,.5)
                points.append(item)
            print("POINT",band,mode,[(v["theta"],v["status"]) for v in values],flush=True)
    tasks=((band,n,sampled[i,n]) for i,band in enumerate(BANDS) for n in range(args.draws))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             initargs=(args.candidate,weeks)) as pool:
        for pos,chunk in enumerate(pool.map(worker,tasks,chunksize=12),1):
            rows.extend(chunk)
            if pos%args.draws==0:
                print("COMPLETED",pos,"of",len(BANDS)*args.draws,flush=True)
                # Persist completed groups to survive interruptions.
                pd.DataFrame(rows).to_csv(HERE/"s6-all-window-draws.csv",index=False)
    frame=pd.DataFrame(rows);assert len(frame)==len(BANDS)*args.draws
    def percentiles(x):
        a=np.asarray(x,float);finite=a[np.isfinite(a)]
        return np.percentile(finite,[2.5,50,97.5]).tolist() if len(finite) else None
    results=[]
    for (band,mode),group in frame.groupby(["birth_band","mode"],sort=False):
        for dose in (0,2):
            col=str(dose);p=next(x for x in points if x["birth_band"]==band and x["dose"]==dose and x["mode"]==mode)
            results.append(dict(birth_band=int(band),mode=mode,dose=dose,point=p["theta"],point_raw=p["theta_raw"],
                                population=p["enrollment_people"],fit_deaths=p["fit_deaths"],fit_weeks=p["fit_weeks"],
                                low_theta_se=p.get("information_low",{}).get("se"),
                                theta_interval=percentiles(group["theta"+col]),raw_interval=percentiles(group["theta"+col+"_raw"]),
                                raw_valid=int(group["theta"+col+"_raw"].notna().sum()),
                                upper_bound=int(group["bound"+col].sum()),
                                near_zero_applied=int(group["theta"+col].lt(1e-6).sum()),
                                near_zero_raw=int(group["theta"+col+"_raw"].lt(1e-6).sum()),
                                status_counts=group["status"+col].value_counts().to_dict()))
    ratios=[]
    for (band,mode),group in frame.groupby(["birth_band","mode"],sort=False):
        a,b=[next(x for x in points if x["birth_band"]==band and x["dose"]==dose and x["mode"]==mode) for dose in (0,2)]
        ratios.append(dict(birth_band=int(band),mode=mode,point=b["growth"]/a["growth"],
                           interval=percentiles(group.kcor),valid=int(group.kcor.notna().sum())))
    result=dict(draws=args.draws,seed=20261126,params=results,ratios=ratios,points=points)
    (HERE/"s6-all-window.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("DONE",len(frame),flush=True)

if __name__=="__main__":main()

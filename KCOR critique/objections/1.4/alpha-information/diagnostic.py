"""Local information in quiet survival shape for a common unknown alpha.

Each cohort has its own unknown k, theta and gap exposures, including wave
exposure: no cross-cohort efficacy or exposure-ratio assumptions. This is a
model-conditional information benchmark, not an alpha estimate.
"""
from pathlib import Path
import sys, math
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"2.7/low-frailty/theoretical-limits"),str(ROOT/"3.2/nonlinear-survivor-fit")]
from limits import Model,old
from fitter import make_fitter
from scipy.stats import norm

def alpha_info(m,p,scale_step=1.):
    # Conditional multinomial/binomial information. Profile out every
    # cohort-specific parameter, including independent aggregate wave exposure.
    q=m.q(p);probs=-np.expm1(-q);n=m.N*np.exp(m.logS(p)[:-1])
    weight=np.sqrt(n*(1-probs)/probs)
    J=[]
    for j in range(len(p)):
        step=(1e-5 if j<2 else 1e-7)*scale_step
        a=p.copy();b=p.copy();a[j]+=step;b[j]-=step
        J.append((m.q(a)-m.q(b))/(2*step))
    alpha=m.alpha
    try:
        m.alpha=alpha+1e-4*scale_step;a=m.q(p)
        m.alpha=alpha-1e-4*scale_step;b=m.q(p)
    finally:m.alpha=alpha
    signal=(a-b)/(2e-4*scale_step)*weight
    A=np.array(J).T*weight[:,None]
    scale=np.linalg.norm(A,axis=0)
    A=A[:,scale>0]/scale[scale>0]
    # Alpha's information remaining after re-estimating each nuisance.
    residual=signal-A@np.linalg.lstsq(A,signal,rcond=1e-9)[0]
    info=float(np.dot(residual,residual))
    return info,float(np.dot(signal,signal))

def actual_rows():
    table=old.table
    table.initialize([],"entry-moment")
    for kind,enroll,group,dose,dates,risk,dead in old.read_czech():
        time=np.arange(len(dates),dtype=float);skip=table.K.DYNAMIC_HVE_SKIP_WEEKS
        quiet,_,_=table.K._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(),time)
        t=time-skip;wave=np.asarray((dates>=table.WAVE["start_dt"])&(dates<=table.WAVE["end_dt"]))
        effective=wave&(t>=0)
        if not effective.any():continue
        hazard=table.K.hazard_from_mr(np.clip(dead/np.maximum(risk,1),0,.999));hazard[t<0]=0
        trial=make_fitter(table.K,(t[effective][0],t[effective][-1]),1.18)
        (k,theta),diagnostic=trial(hazard,t,quiet,table.CONFIG["k_anchor_weeks"],table.CONFIG["gamma_per_week"],deaths_arr=dead)
        post= t>=0
        h=hazard[post]
        observed=np.r_[0.,np.cumsum(h)]
        q=quiet[post]&~effective[post];w=effective[post]
        # Only genuine fitted baseline rows (same trial mask). Do not use
        # a marked wave as baseline. Retain all gap survival transitions.
        q|=(t[post]<table.CONFIG["k_anchor_weeks"])&~w
        N=int(risk[post][0]);anchor=table.K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE+1-skip
        m=Model(N,q,w,anchor,len(q),1.18)
        for name,th in (("fitted-reference",theta),("theta_0.5",.5)):
            if th<1e-4:
                info=known=alt=0.;wave_exp=float("nan")
            else:
                p=m.reference(k,th,observed)
                info,known=alpha_info(m,p)
                alt,_=alpha_info(m,p,5.)
                B,W=m.paths(p);wave_exp=float(W[-1])
            yield dict(source="actual Czech",enrollment=enroll,group=group,dose=dose,N=N,k=k,theta=th,fitted_theta=theta,scenario=name,info=info,alternate_info=alt,alpha_se=1/math.sqrt(info) if info>0 else np.inf,
                       known_nuisance_se=1/math.sqrt(known) if known>0 else np.inf,wave_exposure=wave_exp,fit_status=diagnostic["status"])
def synthetic_rows():
    for name,windows,N in (("main",((2,26),(35,52)),250000),("stress",((2,5),(35,52)),12500)):
        t=np.arange(2,53);quiet=np.logical_or.reduce([(t>=a)&(t<=b) for a,b in windows]);wave=(t>=27)&(t<=34)
        for age,k in ((1940,.002),(1950,.001)):
            for theta in (.5,4.):
                m=Model(N,quiet,wave,4,len(t),1.18)
                extra=np.array([3*k*(m.cuts[i+1]-m.cuts[i]) if wave[m.cuts[i]] else 0. for i in m.nuisance])
                p=m.reference(k,theta,known=extra);info,known=alpha_info(m,p);alt,_=alpha_info(m,p,5.)
                yield dict(source="synthetic",enrollment=name,group=age,N=N,k=k,theta=theta,scenario="known-truth",info=info,alternate_info=alt,alpha_se=1/math.sqrt(info),known_nuisance_se=1/math.sqrt(known),wave_exposure=3*k*8)
def main():
    a=pd.DataFrame([*synthetic_rows(),*actual_rows()])
    a.to_csv(HERE/"cohort-information.csv",index=False)
    summary=[]
    for (source,enroll,scenario),g in a.groupby(["source","enrollment","scenario"]):
        info=g["info"].sum();se=1/math.sqrt(info) if info>0 else np.inf
        summary.append(dict(source=source,enrollment=enroll,scenario=scenario,cohorts=len(g),people=int(g.N.sum()),info=info,alternate_se=1/math.sqrt(g.alternate_info.sum()) if g.alternate_info.sum()>0 else np.inf,alpha_se=se,half95=1.96*se,z_for_alpha_1=0.18/se))
    df=pd.DataFrame(summary);df.to_csv(HERE/"pooled-information.csv",index=False)
    print(df.round(5).to_string(index=False))
    print(a[a.source=="actual Czech"][["enrollment","group","dose","scenario","theta","alpha_se","wave_exposure","fit_status"]].round(4).to_string(index=False))
if __name__=="__main__":main()


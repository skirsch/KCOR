"""Replay the historical pooled alpha objective and replace the frailty inputs.

The exact arm refits k/theta at every alpha and uses mean-normalized exact
survivor moments at the same week-end convention as the historical proxy.
This is an illustrative re-evaluation of the historical excess-ratio objective,
not identification of an intervention-effect scale or causal alpha.
"""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
from pathlib import Path
import sys,types,importlib.util,copy,json
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.special import logsumexp
from prepare import HERE,REPO,COMMIT,CFG,git
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"3.2/nonlinear-survivor-fit"))
from fitter import make_fitter,make_correction,candidate
from trial import gamma_quadrature
sys.path.insert(0,str(ROOT/"1.3"))
import table8_bootstrap as table

def load_code(name,source,path):
    mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    exec(compile(source,str(path),"exec"),vars(mod))
    return mod

def setup():
    global hist,config,grid,history,wave,model,working,oldK
    hist=load_code("historical_alpha",git("test/alpha/code/estimate_alpha.py"),HERE/"historical_alpha.py")
    config=copy.deepcopy(CFG)
    config["analysis"]["enrollment_dates"]=[s for s in config["analysis"]["enrollment_dates"] if s.replace("_","-")<="2022-12"]
    grid=hist.expand_alpha_grid(config["alpha_grid"])
    history=pd.read_csv(HERE/"aggregate-histories.csv")
    # Point historical loader at its pinned YAML. No changes to existing config.
    folder=HERE/"historical-config";(folder/"data/Czech").mkdir(parents=True,exist_ok=True)
    (folder/"data/Czech/Czech.yaml").write_bytes(git("data/Czech/Czech.yaml"))
    oldK=load_code("historical_kcor",git("code/KCOR.py"),folder/"code/KCOR.py")
    os.environ["DATASET"]="Czech"
    wave=hist.build_wave_window(hist.load_yaml(folder/"data/Czech/Czech.yaml"))
    table.compare.initialize("entry-moment");model=table.compare.adapter.K
    working=candidate(model,"multistart")
    model._load_dataset_dose_pairs_config()
    return folder

def original_arms():
    folder=setup()
    def read(path,sheet_name=None,**kwargs):return history[history.enrollment==sheet_name].drop(columns="enrollment").copy()
    with patch.object(hist.pd,"read_excel",read):
        a,da=hist.build_real_cohort_table(folder,config,oldK)
        with patch.object(model,"fit_theta0_gompertz",working):
            b,db=hist.build_real_cohort_table(folder,config,model)
    a.to_csv(HERE/"historical-wave-table.csv",index=False);da.to_csv(HERE/"historical-cohort-fits.csv",index=False)
    b.to_csv(HERE/"working-wave-table.csv",index=False);db.to_csv(HERE/"working-cohort-fits.csv",index=False)
    print("historical eligible",da.eligible.sum(),"working eligible",db.eligible.sum(),flush=True)
    return a,b

def moment(theta,alpha,B,W):
    if theta<1e-8 or abs(alpha-1)<1e-12:return 1.
    z,w=gamma_quadrature(theta);z=z/(1+theta*B)
    with np.errstate(divide="ignore"):
        logw=np.log(w)-W*z**alpha
        normal=logsumexp(logw)
        mean=np.exp(logsumexp(logw+np.log(z))-normal)
        mp=np.exp(logsumexp(logw+alpha*np.log(z))-normal)
    return mp/mean**alpha

def exact_task(alpha):
    result=[];fits=[]
    for (enroll,age,dose),raw in history.groupby(["enrollment","YearOfBirth","Dose"]):
        raw=raw.sort_values("ISOweekDied");id=f"{enroll}|{age}|{dose}"
        alive=raw.Alive.to_numpy();dead=raw.Dead.to_numpy()
        dates=pd.DatetimeIndex([hist.iso_label_to_monday(x) for x in raw.ISOweekDied])
        tt=np.asarray((dates-hist.enrollment_label_to_monday(enroll)).days/7,dtype=float)
        valid=alive>0;alive,dead,dates,tt=alive[valid],dead[valid],dates[valid],tt[valid]
        pre=(dates>=wave.start_monday)&(dates<=wave.end_monday)
        if not pre.any():continue
        cfg=model.get_gompertz_theta_fit_config();t=tt-model.DYNAMIC_HVE_SKIP_WEEKS
        h=model.hazard_from_mr(np.clip(dead/alive,0,1-1e-12));h[t<0]=0.
        q,_,_=model._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(),tt)
        active=np.asarray(pre&(t>=0))
        if not active.any():continue
        fit=make_fitter(model,(t[active][0],t[active][-1]),float(alpha))
        (k,theta),diag=fit(h,t,q,cfg["k_anchor_weeks"],cfg["gamma_per_week"],deaths_arr=dead)
        if not diag["success"] or not np.isfinite(theta):theta=0.
        theta,_,_=model._apply_theta_degenerate_guard(theta,model.get_theta_degenerate_fit_config())
        factors={}
        if abs(alpha-1.)>1e-12 and theta>=model.KCOR6_THETA_EPS and np.isfinite(k) and k>0:
            keep=dates<=wave.end_monday;obs=np.cumsum(h)[keep]
            H=model.invert_gamma_frailty(obs,theta)
            _,wd=make_correction(model)(H,tt[keep],dates.to_numpy()[keep],theta,float(alpha),wave.start_monday,wave.end_monday,k,cfg["gamma_per_week"],observed_cumhaz=obs)
            b=wd.get("baseline_exposure_path",[]);w=wd.get("wave_exposure_path",[])
            if len(b):
                assert len(b)==int(active.sum())
                for week,B,W in zip(tt[active],b,w):factors[int(week)]=moment(theta,float(alpha),B,W)
        # The historical objective retains pre-skip wave rows and uses their
        # modeled-origin frailty state; preserve this convention for comparison.
        for week in tt[pre]:
            factor=factors.get(int(week),moment(theta,float(alpha),0.,0.))
            result.append(dict(cohort_id=id,t_week=int(week),alpha=float(alpha),factor=factor))
        fits.append(dict(cohort_id=id,alpha=float(alpha),k=k,theta=theta,success=bool(diag["success"]),flag=bool(diag.get("insufficient_signal",False)),status=diag["status"]))
    return float(alpha),result,fits

def init_exact():setup()

def exact_arm():
    setup()
    results=[];fits=[]
    with ProcessPoolExecutor(7,initializer=init_exact) as pool:
        for n,(alpha,r,d) in enumerate(pool.map(exact_task,grid),1):
            results.extend(r);fits.extend(d)
            if n%5==0 or n==len(grid):print("exact alpha grid",n,"/",len(grid),flush=True)
    pd.DataFrame(results).to_csv(HERE/"exact-moments.csv",index=False)
    pd.DataFrame(fits).to_csv(HERE/"exact-fits.csv",index=False)

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="exact":exact_arm()
    else:original_arms()

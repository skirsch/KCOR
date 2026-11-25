"""Verify exact model and benchmark noiseless and sampled synthetic fits."""
import time,sys,json,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from fitter import HERE,ROOT,Survival,reference_hazard,make_fitter,make_correction,candidate
sys.path.insert(0,str(ROOT/"1.3/multiwindow-coverage"))
import study

def main():
    for th in (.000001,.01,.5,4.,15.,100.):
        for a in (.7,1.18,2.):
            for B,W in ((0,0),(.01,.001),(.1,.05),(.5,.2),(2.,1.)):
                val=Survival(th,a).hazard(B,W)
                ref=reference_hazard(B,W,th,a)
                np.testing.assert_allclose(val,ref,rtol=2e-5,atol=2e-8,err_msg=f"{th},{a},{B},{W}")
    study.initialize()
    model=study.model
    fit=make_fitter(model,(study.WAVE[0]-study.skip,study.WAVE[1]-study.skip))
    # Proportional routing preserves the same all-window fitter exactly.
    p0,_=study.population("no_wave");d0,a0=study.arrays(p0*study.POPULATION/4)
    h0=model.hazard_from_mr(d0/(a0+model.EPS));h0[...,:study.skip]=0
    fixed=candidate(model,"multistart")
    prop=make_fitter(model,(study.WAVE[0]-study.skip,study.WAVE[1]-study.skip),1.)
    for j in np.ndindex(study.THETA.shape):
        args=(h0[j],study.TIME-study.skip,study.quiet,study.config["k_anchor_weeks"],study.config["gamma_per_week"])
        np.testing.assert_array_equal(fixed(*args,deaths_arr=d0[j])[0],prop(*args,deaths_arr=d0[j])[0])
    print("Proportional routing identical to existing all-window candidate.")
    prob,_=study.population("nonlinear_wave");c=prob*study.POPULATION/4
    dead,alive=study.arrays(c)
    h=model.hazard_from_mr(dead/(alive+model.EPS));h[...,:study.skip]=0
    start=time.perf_counter()
    for age,dose in np.ndindex(study.THETA.shape):
        (k,theta),d=fit(h[age,dose],study.TIME-study.skip,study.quiet,study.config["k_anchor_weeks"],study.config["gamma_per_week"],deaths_arr=dead[age,dose])
        np.testing.assert_allclose([k,theta],[study.RATES[age],study.THETA[age,dose]],rtol=1e-6,atol=1e-10)
        print("expected",age,dose,k,theta,"truth",study.RATES[age],study.THETA[age,dose],"residual",d["rmse_hazard"],"nfev",d.get("trial_nfev"),flush=True)
    print("expected fit seconds",time.perf_counter()-start)
    for rep in (0,34):
        batch,_=study.samples("nonlinear_wave",rep,0);dead,alive=study.arrays(batch[0]);h=model.hazard_from_mr(dead/(alive+model.EPS));h[...,:study.skip]=0
        start=time.perf_counter()
        for age,dose in np.ndindex(study.THETA.shape):
            (k,theta),d=fit(h[age,dose],study.TIME-study.skip,study.quiet,study.config["k_anchor_weeks"],study.config["gamma_per_week"],deaths_arr=dead[age,dose])
            print("sample",rep,age,dose,k,theta,d["status"],d.get("trial_nfev"),flush=True)
        print("sample fit seconds",time.perf_counter()-start,flush=True)
if __name__=="__main__":main()



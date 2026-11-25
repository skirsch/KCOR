"""Independent quadrature checks of predicted weekly hazards and gap exposure."""
import fitter as f
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("nonlinear_survivor_replay",Path(__file__).resolve().parent/"replay.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
from scipy.optimize import brentq
import numpy as np
import json

class Reference:
    def __init__(self,theta,alpha):self.theta=theta;self.alpha=alpha
    def hazard(self,B,W):
        b,w=np.broadcast_arrays(B,W);result=np.empty(b.shape)
        for idx in np.ndindex(b.shape):
            result[idx]=b[idx]+w[idx] if self.theta<=1e-10 else f.reference_hazard(float(b[idx]),float(w[idx]),self.theta,self.alpha)
        return result.item() if result.ndim==0 else result
    def increase(self,B,W,target,axis,guess):
        if target<=self.hazard(B,W)+1e-15:return 0.
        def fn(x):return self.hazard(B+x,W)-target if axis==0 else self.hazard(B,W+x)-target
        u=max(guess,1e-10)
        while fn(u)<0:u*=2
        return brentq(fn,0.,u,xtol=1e-12,rtol=1e-13)

def main():
    table=r.table
    data=r.subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=r.ROOT)
    pop=[table.read_population(r.io.BytesIO(data),s) for s in table.SHEETS]
    r.table_init(pop)
    saved=np.load(f.HERE/"table8-draws.npz")["parameters"]
    tests=[];native=f.Survival
    for i,(dates,time,quiet,keep,counts) in enumerate(table.DATA):
        t=time-table.K.DYNAMIC_HVE_SKIP_WEEKS
        wave=(dates>=np.datetime64(table.WAVE["start_dt"]))&(dates<=np.datetime64(table.WAVE["end_dt"]))&(t>=0)
        counts=counts.copy();counts[2]+=counts[0]+counts[1]
        alive=np.cumsum(counts[:,:,::-1],axis=2)[:,:,::-1][:,:,:-1]
        dead=counts[:,:,:-1]
        for a,d in np.ndindex((3,3)):
            h=table.K.hazard_from_mr(np.clip(dead[a,d]/alive[a,d],0,.999));h[t<0]=0
            p=f.Profile(h,t,quiet,table.CONFIG["k_anchor_weeks"],table.CONFIG["gamma_per_week"],(t[wave][0],t[wave][-1]))
            k,theta,_=saved[i,0,a*3+d]
            x=p.predict(k,theta,1.18)
            try:
                f.Survival=Reference
                y=p.predict(k,theta,1.18)
            finally:f.Survival=native
            np.testing.assert_allclose(x[0],y[0],atol=1e-8,rtol=1e-5)
            assert x[3]==y[3]
            tests.append(dict(enrollment=table.SHEETS[i],group=a,pseudo_dose=d,theta=theta,max_abs_prediction_difference=float(np.max(abs(x[0]-y[0]))),deficit_weeks=x[3]))
    (f.HERE/"quadrature-verification.json").write_text(json.dumps(tests,indent=2)+"\n",encoding="utf8")
    print("18 actual fitted curves agree with independent probability-space integration; max absolute error",max(x["max_abs_prediction_difference"] for x in tests),flush=True)
if __name__=="__main__":main()


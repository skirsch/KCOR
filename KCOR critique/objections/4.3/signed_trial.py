"""Isolated signed cumulative wave exposure experiment, matched existing main design."""
import os
for x in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[x]="1"
import sys,json
from pathlib import Path
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gamma
from scipy.stats import t as student
from concurrent.futures import ProcessPoolExecutor
HERE=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m95"
sys.path.insert(0,str(D/"code"))
os.environ["DATASET"]="Czech"
import KCOR as K
import kcor_survivor as S
from types import SimpleNamespace
src=(D/"code/kcor_survivor.py").read_text()
a=src.index("            if target > planned_loss:")
b=src.index("            output[i]",a)
new=src[:a]+'''            if target >= cumulative_hazard(planned, 0., theta, alpha):
                baseline = planned
                wave = solve_increase(lambda x: cumulative_hazard(baseline, x, theta, alpha),
                                      target, max(wave, target-planned_loss, 1e-10))
            else:
                deficits += 1
                wave = 0.
                baseline = solve_increase(lambda x: cumulative_hazard(x, 0., theta, alpha),
                                          target, planned)
'''+src[b:]
a=new.index("                    loss=sv.hazard(planned,W)")
b=new.index("        return np.array(pred)",a)
new=new[:a]+'''                    if target>=sv.hazard(planned,0.):
                        B=planned
                        W=sv.increase(B,0.,target,1,max(W,1e-10))
                    else:
                        deficits+=1
                        W=0.
                        B=sv.increase(0.,0.,target,0,planned)
'''+new[b:]
M=SimpleNamespace()
exec(compile(new,"<signed trial>","exec"),M.__dict__)
model=SimpleNamespace(**vars(K));model.apply_nph_correction_post_inversion=K._apply_nph_identity_or_fallback
corrections=[S.make_correction(model),M.make_correction(model)]
fits=[S.make_fitter(model,(25,32),1.18),M.make_fitter(model,(25,32),1.18)]
TIME=np.arange(53);dates=np.datetime64("2021-06-14")+TIME*7
quiet=((TIME>=2)&(TIME<=26))|((TIME>=35)&(TIME<=52))
rates=[.002,.001];theta=[4.,.5];g=.085/52
def probs(wave):
 out=[]
 for k in rates:
  B=k*np.expm1(g*np.maximum(TIME-1,0))/g
  W=3*k*np.cumsum((TIME>=27)&(TIME<=34))*wave
  for th in theta:
   f=lambda u:np.exp(-u**th-th*u**th*B-(th*u**th)**1.18*W)
   surv=quad_vec(f,0,np.inf,epsabs=1e-12,epsrel=1e-12)[0]*th/gamma(1/th)
   surv[:2]=np.exp(-k*(TIME[:2]+1));surv[2:]*=surv[1]
   out.append(np.r_[-np.diff(np.r_[1.,surv]),surv[-1]])
 return np.array(out)
P=[probs(0),probs(1)]
def estimate(c,arm,fitted):
 ratios=[];params=[];waves=[];decreases=0;negativeB=0;negativeW=0
 for i,x in enumerate(c):
  alive=np.cumsum(x[::-1])[::-1][:-1];dead=x[:-1]
  h=-np.log1p(-dead/alive);h[:2]=0
  k,th=rates[i//2],theta[i%2]
  if fitted:
   (k,th),d=fits[arm](h,TIME-2,quiet,4,g,deaths_arr=dead)
   if not d["success"] or not np.isfinite(th):raise RuntimeError(d)
  H=np.cumsum(h);inv=K.invert_gamma_frailty(H,th)
  y,d=corrections[arm](inv,TIME,dates,th,1.18,dates[27],dates[34],k,g,observed_cumhaz=H)
  ratios.append(y[-1]/y[5]);params.append(th);waves.append(d.get("post_wave_exposure",0.))
  decreases+=int(np.any(np.diff(y)<-1e-10))
  negativeB+=int(np.any(np.diff(d.get("baseline_exposure_path",[]))<-1e-10))
  negativeW+=int(np.any(np.diff(d.get("wave_exposure_path",[]))<-1e-10))
 return [np.exp(np.average(np.log(np.array(ratios)[1::2]/np.array(ratios)[::2]),weights=[2,1])),*params,*waves,decreases,negativeB,negativeW]
def job(args):
 w,n,rep,fitted,draws=args
 rng=np.random.default_rng([20261125,w,n,rep])
 c=np.array([rng.multinomial(n,p) for p in P[w]])
 batch=[c]+[np.array([rng.multinomial(n,x/n) for x in c]) for _ in range(draws)]
 return args,np.array([[estimate(x,a,fitted) for a in range(2)] for x in batch])
def main():
 tasks=[(w,n,r,False,0) for w in (0,1) for n in (12500,250000) for r in range(100)]
 tasks += [(w,250000,r,True,99 if r==0 else 0) for w in (0,1) for r in range(20)]
 res=[]
 with ProcessPoolExecutor(5) as pool:
  for i,(args,v) in enumerate(pool.map(job,tasks,chunksize=1),1):
   res.append((args,v))
   if i%50==0:print(i,"/",len(tasks),flush=True)
 report=[]
 for w,n,f in [(w,n,False) for w in (0,1) for n in (12500,250000)]+[(w,250000,True) for w in (0,1)]:
  subset=[v for a,v in res if a[0]==w and a[1]==n and a[3]==f]
  x=np.array([v[0] for v in subset]);mean=x.mean(0);half=student.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
  report.append(dict(wave=w,N_per_cohort=n,fitted=f,reps=len(x),mean=mean.tolist(),lower=(mean-half).tolist(),upper=(mean+half).tolist(),bootstrap_first=np.quantile(subset[0][1:,:,0],[.025,.975],axis=0).tolist() if f else None,first_point=subset[0][0,:,0].tolist()))
 (HERE/"signed-results.json").write_text(json.dumps(report,indent=2))
 np.savez_compressed(HERE/"signed-draws.npz",**{f"{a[0]}_{a[1]}_{a[2]}_{a[3]}":v for a,v in res})
 print(json.dumps(report),flush=True)
if __name__=="__main__":main()
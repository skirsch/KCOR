"""Coverage in the original contamination design with no extra mortality."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import sys,importlib.util,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import t,binomtest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("contamination_replay",HERE.parents[1]/"2.7/all-window-contamination/replay.py")
cont=importlib.util.module_from_spec(spec);spec.loader.exec_module(cont)
DRAWS=99
SIZES=(50000,5000000)
REPS=200

def init():
    global cfg
    cfg=cont.setup()
    cont.study.K.fit_theta0_gompertz=cfg[2]

def fit(a,d):
    _,_,_,k,gamma,skip,weeks,rebased,quiet,anchors=cfg
    return cont.study.cohort_pipeline(d,a,weeks,rebased,quiet,anchors,gamma)

def task(args):
    size,rep=args
    _,_,_,k,gamma,skip,*_=cfg
    stream=np.random.default_rng(np.random.SeedSequence([10042,2,0,0,rep]))
    samples=[cont.study.simulate_one_cohort(stream,size,cont.study.N_WEEKS,k,gamma,theta,skip,None) for theta in (cont.study.THETA0_A,cont.study.THETA0_B)]
    rng=np.random.default_rng([20261125,size,rep])
    draws=[]
    for a,d in samples:
        outcomes=np.r_[d,a[-1]-d[-1]]
        assert outcomes.sum()==size
        draws.append(np.vstack([outcomes,rng.multinomial(size,outcomes/size,size=DRAWS)]))
    points=[]
    for i in range(DRAWS+1):
        fitted=[]
        for outcomes in draws:
            p=outcomes[i]
            alive=np.cumsum(p[::-1])[::-1][:-1]
            fitted.append(fit(alive,p[:-1]))
        a,b=fitted
        ratio=cont.study.kcor_asymptote_value(cont.study.anchored_kcor(a[4],b[4],skip),b[4])
        points.append([ratio,a[1],b[1],int(not a[2]["success"])+int(not b[2]["success"]),int(a[2].get("insufficient_signal",False))+int(b[2].get("insufficient_signal",False))])
    return size,rep,np.array(points)

def main():
    args=[(size,i) for i in range(REPS) for size in SIZES]
    out={str(n):[] for n in SIZES}
    with ProcessPoolExecutor(4,initializer=init) as pool:
        for j,(size,rep,v) in enumerate(pool.map(task,args),1):
            assert rep==len(out[str(size)])
            out[str(size)].append(v)
            if j%20==0:
                print("single window",j,"/",len(args),flush=True)
                np.savez_compressed(HERE/"contamination-draws.npz",**{n:np.asarray(rows) for n,rows in out.items()})
    ref=pd.read_csv(HERE.parents[1]/"2.7/all-window-contamination/runs.csv")
    result={};records=[]
    for size in SIZES:
        v=np.asarray(out[str(size)])
        r=ref[(ref["size"]==size)&(ref.rep>=0)&(ref.epsilon_comorb==0)&(ref.fitter=="all_window")].sort_values("rep")
        np.testing.assert_allclose(v[:,0,0],r.KCOR_asymptote,rtol=1e-10,atol=1e-12)
        lo,hi=np.quantile(v[:,1:,0],[.025,.975],axis=1)
        covered=(lo<=1)&(hi>=1)
        ci=binomtest(int(covered.sum()),REPS).proportion_ci()
        result[str(size)]=dict(coverage=float(covered.mean()),covered=int(covered.sum()),coverage_interval=[ci.low,ci.high],mean_point=cont.ci(v[:,0,0]),mean_width=cont.ci(hi-lo),median_width=float(np.median(hi-lo)),rmse=float(np.sqrt(np.mean((v[:,0,0]-1)**2))),failed=int(v[:,:,3].sum()),flags=int(v[:,:,4].sum()))
        records.extend(dict(size=size,rep=j,point=v[j,0,0],lo=lo[j],hi=hi[j],covered=bool(covered[j])) for j in range(REPS))
    (HERE/"contamination-coverage.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    pd.DataFrame(records).to_csv(HERE/"contamination-intervals.csv",index=False)
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()

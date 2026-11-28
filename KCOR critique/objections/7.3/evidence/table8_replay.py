"""Full matched Table 8 replay after residual scaling; preserve original seed."""
from pathlib import Path
import os,sys,io,json,argparse,time
from concurrent.futures import ProcessPoolExecutor
for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[k]="1"
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"analysis/table8"))
import table8_bootstrap as T
def init(pop,mode):
    os.environ["KCOR_NPH_MODE"]=mode
    T.initialize(pop,"revised")
    assert T.WAVE["apply_correction"]==(mode=="forced_alpha")
def one(job):
    i,j=job
    recorded=[]
    native=T.compare.joint_fit
    def logged(*a,**kw):
        p,d=native(*a,**kw)
        recorded.append([float(p[0]),float(p[1]),float(d.get("theta0_raw",p[1]))])
        return p,d
    T.compare.joint_fit=logged
    try:
        point,flags,failed=T.estimate(i,j)
    finally:T.compare.joint_fit=native
    return i,j,point,flags,failed,np.asarray(recorded)
def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["identified_only","forced_alpha","both"],default="both");p.add_argument("--draws",type=int,default=999);p.add_argument("--workers",type=int,default=7);args=p.parse_args()
    populations=[T.read_population(io.BytesIO((ROOT/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
    for mode in (["identified_only","forced_alpha"] if args.mode=="both" else [args.mode]):
        begin=time.monotonic()
        n=args.draws+1
        vals=np.full((2,n,3),np.nan);flags=np.full((2,n,2),np.nan);params=np.full((2,n,9,3),np.nan)
        done=np.zeros((2,n),bool)
        checkpoint=HERE/(mode+"-partial.npz")
        if checkpoint.exists():
            s=np.load(checkpoint)
            assert s["points"].shape==vals.shape
            vals[:]=s["points"];flags[:]=s["flags"];params[:]=s["parameters"];done[:]=s["done"]
        jobs=[(i,j) for j in range(n) for i in range(2) if not done[i,j]]
        with ProcessPoolExecutor(args.workers,initializer=init,initargs=(populations,mode)) as pool:
            for i,j,v,f,e,p in pool.map(one,jobs,chunksize=2):
                vals[i,j]=v;flags[i,j]=f,e;params[i,j]=p;done[i,j]=True
                count=int(done.sum())
                if j==0 or count%100==0:
                    print(mode,count,"of",2*n,"elapsed seconds",round(time.monotonic()-begin), "point" if j==0 else "", v if j==0 else "",flush=True)
                if j==0 or count%50==0:np.savez_compressed(checkpoint,points=vals,flags=flags,parameters=params,done=done)
        assert done.all() and np.isfinite(vals).all()
        report={}
        for i,sheet in enumerate(T.SHEETS):
            low,high=np.quantile(vals[i,1:],[.025,.975],axis=0)
            report[sheet]=dict(point=vals[i,0].tolist(),lower=low.tolist(),upper=high.tolist(),flagged_draws=int((flags[i,1:,0]>0).sum()),failed=int(flags[i,:,1].sum()))
        (HERE/(mode+"-table8.json")).write_text(json.dumps(report,indent=2),encoding="utf8")
        np.savez_compressed(HERE/(mode+"-draws.npz"),points=vals,flags=flags,parameters=params)
        print(mode,"complete",json.dumps(report),flush=True)
if __name__=="__main__":main()

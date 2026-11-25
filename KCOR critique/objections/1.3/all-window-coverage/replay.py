"""Matched coverage rerun of previously specified high-information designs."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
from pathlib import Path
import sys, inspect,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE.parent/"multiwindow-coverage"),str(ROOT/"2.7/low-frailty/all-window-joint")]
import study
from experiment import candidate
from scipy.stats import binomtest

def initialize():
    global estimate_one
    study.initialize()
    study.compare.joint_fit=candidate(study.model,"multistart")
    # Retain exactly the already verified fast replay, executing just this arm.
    source=inspect.getsource(study.estimate)
    original="for fit in (compare.production_fit, compare.joint_fit):"
    assert source.count(original)==1
    ns=vars(study).copy()
    exec(compile(source.replace(original,"for fit in (compare.joint_fit,):"),"<single candidate replay>","exec"),ns)
    estimate_one=ns["estimate"]

def task(args):
    scenario,rep=args
    batch,truth=study.samples(scenario,rep,study.DRAWS)
    w=study.pooling_weights(batch)
    result=np.asarray([estimate_one(c,w,scenario)[0] for c in batch])
    return scenario,rep,truth,w,result

def report(archive):
    old=np.load(HERE.parent/"multiwindow-coverage/coverage-draws.npz")
    summary={}
    records=[]
    for s in study.SCENARIOS:
        v=archive[s]
        if len(v)!=study.REPETITIONS:continue
        x=np.asarray(v)
        w=np.asarray(archive[s+"_weights"])
        np.testing.assert_allclose(w,old[s+"_weights"],rtol=0,atol=1e-13)
        assert np.isfinite(x).all()
        combined=np.concatenate((old[s],x[:,:,None,:]),axis=2)
        labels=("fixed","previous_joint","all_window")
        lo,hi=np.quantile(combined[:,1:,:,0],[.025,.975],axis=1)
        covered=(lo<=1)&(hi>=1)
        entry={}
        for j,lab in enumerate(labels):
            vals=combined[:,0,j,0]
            n=int(covered[:,j].sum())
            ci=binomtest(n,len(vals)).proportion_ci()
            width=hi[:,j]-lo[:,j]
            entry[lab]=dict(coverage=n/len(vals),covered=n,n=len(vals),coverage_interval=[ci.low,ci.high],
                            mean_point=study.mean_interval(vals),rmse=float(np.sqrt(np.mean((vals-1)**2))),
                            mean_width=study.mean_interval(width),median_width=float(np.median(width)),
                            max_width=float(np.max(width)),flags=int(combined[:,:,j,3].sum()),failures=int(combined[:,:,j,4].sum()),
                            mean_theta=[study.mean_interval(combined[:,0,j,i]) for i in range(6,10)])
            for rep in range(len(vals)):
                records.append(dict(scenario=s,rep=rep,fitter=lab,point=vals[rep],lower95=lo[rep,j],upper95=hi[rep,j],covered=bool(covered[rep,j])))
        for j,lab in enumerate(labels[:2]):
            gain=int((~covered[:,j]&covered[:,2]).sum());loss=int((covered[:,j]&~covered[:,2]).sum())
            entry["vs_"+lab]=dict(gained=gain,lost=loss,p=binomtest(gain,gain+loss).pvalue if gain+loss else 1.,
                paired_point_change=study.mean_interval(combined[:,0,2,0]-combined[:,0,j,0]),
                paired_width_change=study.mean_interval((hi[:,2]-lo[:,2])-(hi[:,j]-lo[:,j])))
        summary[s]=entry
    (HERE/"results.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    pd.DataFrame(records).to_csv(HERE/"intervals.csv",index=False)
    print(json.dumps(summary,indent=2),flush=True)

def main():
    initialize()
    # Deterministic expected count case, same truth and candidate.
    for s in study.SCENARIOS:
        p,_=study.population(s);counts=p*study.POPULATION/4
        print(s,"expected counts",estimate_one(counts,study.pooling_weights(counts[None]),s),flush=True)
    args=[(s,i) for i in range(study.REPETITIONS) for s in study.SCENARIOS]
    archive={s:[] for s in study.SCENARIOS}
    archive.update({s+"_weights":[] for s in study.SCENARIOS})
    with ProcessPoolExecutor(7,initializer=initialize) as pool:
        for n,(s,rep,truth,w,points) in enumerate(pool.map(task,args,chunksize=1),1):
            assert rep==len(archive[s]) and truth==1
            archive[s].append(points);archive[s+"_weights"].append(w)
            if n%10==0 or n==len(args):
                print(f"{n}/{len(args)} paired independent datasets",flush=True)
                np.savez_compressed(HERE/"coverage-draws.npz",**{k:np.array(v) for k,v in archive.items()})
    report(archive)
if __name__=="__main__": main()

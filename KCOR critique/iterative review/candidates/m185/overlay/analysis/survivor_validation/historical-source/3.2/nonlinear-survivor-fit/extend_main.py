"""Extend the candidate main coverage replay to all 199 archived bootstrap draws."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("nonlinear_survivor_replay",Path(__file__).resolve().parent/"replay.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
import numpy as np
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor

def initialize():
    r.configure("main")

def task(rep):
    batch,_=r.study.samples("nonlinear_wave",rep,199);w=r.study.pooling_weights(batch)
    v=np.asarray([r.estimate_one(c,w,"nonlinear_wave")[0] for c in batch[100:]])
    return rep,v,w
def main():
    r.configure("main")
    original=np.load(r.HERE/"main-draws.npz")
    first=original["points"];weights=original["weights"]
    assert first.shape==(100,100,10)
    tail=[]
    with ProcessPoolExecutor(7,initializer=initialize) as pool:
        for n,(rep,v,w) in enumerate(pool.map(task,range(100)),1):
            assert rep==len(tail);np.testing.assert_allclose(w,weights[rep],rtol=0,atol=1e-13)
            tail.append(v)
            if n%5==0:
                print("extension",n,"/100",flush=True)
                np.savez_compressed(r.HERE/"main-extension-partial.npz",points=tail)
    points=np.concatenate((first,np.asarray(tail)),axis=1)
    old=np.load(r.ROOT/"1.2/all-window-weekly-survival/main-draws.npz")
    prev=old["points"];np.testing.assert_allclose(weights,old["weights"],rtol=0,atol=1e-13)
    combined=np.stack([prev,points],axis=2)
    report=r.study.summarize(combined,1.)
    report["gamma_working_fit"]=report.pop("fixed");report["nonlinear_survivor_fit"]=report.pop("joint")
    report["mean_theta"]={name:[r.study.mean_interval(combined[:,0,j,i]) for i in range(6,10)] for j,name in enumerate(("gamma_working_fit","nonlinear_survivor_fit"))}
    report["median_width"]=np.median(np.diff(np.quantile(combined[:,1:,:,0],[.025,.975],axis=1),axis=0)[0],axis=0).tolist()
    np.savez_compressed(r.HERE/"main-full-draws.npz",points=points,weights=weights)
    (r.HERE/"main-full-results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    lo,hi=np.quantile(combined[:,1:,:,0],[.025,.975],axis=1);y=combined[:,0,:,0]
    rows=[]
    for i in range(100):
        for j,name in enumerate(("gamma_working_fit","nonlinear_survivor_fit")):rows.append(dict(rep=i,fitter=name,point=y[i,j],lower95=lo[i,j],upper95=hi[i,j],covered=bool(lo[i,j]<=1<=hi[i,j])))
    pd.DataFrame(rows).to_csv(r.HERE/"main-full-intervals.csv",index=False)
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()



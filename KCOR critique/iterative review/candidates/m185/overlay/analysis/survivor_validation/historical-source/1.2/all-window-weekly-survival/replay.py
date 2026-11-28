"""Paired coverage and Table 8 trial combining all-window fitting with weekly survival."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
from pathlib import Path
import sys,inspect,importlib.util,io,json,subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import binomtest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"1.3/multiwindow-coverage"),str(ROOT/"2.7/low-frailty/all-window-joint"),str(HERE.parent/"weekly-survival")]
import study
from experiment import candidate
from trial import make_correction
import table8_bootstrap as table

def install(model):
    global records
    records=[]
    correction=make_correction(model)
    def logged(*a,**kw):
        result,diag=correction(*a,**kw)
        if diag["correction_mode"]=="weekly_survival_trial":
            records.append((diag["wave_points"],diag["deficit_weeks"],diag["max_survival_residual"]))
        return result,diag
    model.apply_nph_correction_post_inversion=logged

def diag():
    return np.array([sum(x[0] for x in records),sum(x[1] for x in records),max((x[2] for x in records),default=0.)])

def init_synth(mode):
    global estimate_one
    if mode=="stress":
        study.POPULATION=50000;study.WINDOWS=((2,5),(35,52));study.SEED=20261126
    study.initialize()
    study.compare.joint_fit=candidate(study.model,"multistart")
    source=inspect.getsource(study.estimate)
    original="for fit in (compare.production_fit, compare.joint_fit):"
    assert source.count(original)==1
    ns=vars(study).copy()
    exec(compile(source.replace(original,"for fit in (compare.joint_fit,):"),"<combined weekly candidate replay>","exec"),ns)
    estimate_one=ns["estimate"]
    install(study.model)
def synth_task(task):
    rep,draws=task
    batch,truth=study.samples("nonlinear_wave",rep,draws)
    weights=study.pooling_weights(batch)
    out=[];checks=[]
    for c in batch:
        records.clear()
        out.append(estimate_one(c,weights,"nonlinear_wave")[0])
        checks.append(diag())
    return rep,np.array(out),np.array(checks),weights

def synth(mode):
    init_synth(mode)
    reps,draws=(50,99) if mode=="stress" else (100,199)
    if mode=="stress":
        prev=np.load(ROOT/"2.7/low-frailty/all-window-joint/multistart-stress.npz")["points"][:,:,1]
    else:
        old=np.load(ROOT/"1.3/all-window-coverage/coverage-draws.npz")
        prev=old["nonlinear_wave"]
    p,_=study.population("nonlinear_wave");c=p*study.POPULATION/4
    print(mode,"expected",estimate_one(c,study.pooling_weights(c[None]),"nonlinear_wave"),flush=True)
    # Check the no-wave identity without recomputing the complete no-wave trial.
    p,_=study.population("no_wave");c=p*study.POPULATION/4
    print(mode,"no-wave expected",estimate_one(c,study.pooling_weights(c[None]),"no_wave"),flush=True)
    points=[];diags=[];weights=[]
    with ProcessPoolExecutor(7,initializer=init_synth,initargs=(mode,)) as pool:
        for n,(rep,v,d,w) in enumerate(pool.map(synth_task,[(i,draws) for i in range(reps)]),1):
            assert rep==len(points);points.append(v);diags.append(d);weights.append(w)
            if n%5==0 or n==reps:
                print(mode,n,"/",reps,"datasets",flush=True)
                np.savez_compressed(HERE/(mode+"-draws.npz"),points=points,diagnostics=diags,weights=weights)
    pts=np.array(points);ds=np.array(diags)
    # The fitter is upstream of the correction, so every applied theta and
    # fit diagnostic must reproduce the prior archive.
    np.testing.assert_array_equal(pts[:,:,3:],prev[:,:,3:])
    if mode=="main":np.testing.assert_allclose(weights,old["nonlinear_wave_weights"],rtol=0,atol=1e-13)
    assert np.isfinite(pts).all() and ds[:,:,2].max()<1e-10
    combined=np.stack([prev,pts],axis=2)
    summary=study.summarize(combined,1.)
    summary["approximate"]=summary.pop("fixed");summary["weekly_survival"]=summary.pop("joint")
    lo,hi=np.quantile(combined[:,1:,:,0],[.025,.975],axis=1)
    summary["median_width"]=np.median(hi-lo,axis=0).tolist()
    summary["max_width"]=np.max(hi-lo,axis=0).tolist()
    summary["diagnostics"]=dict(weeks=int(ds[:,:,0].sum()),deficits=int(ds[:,:,1].sum()),max_survival_residual=float(ds[:,:,2].max()))
    rows=[]
    for arm,label in enumerate(("approximate","weekly_survival")):
        for i in range(reps):rows.append(dict(rep=i,fitter=label,point=combined[i,0,arm,0],lower95=lo[i,arm],upper95=hi[i,arm],covered=bool(lo[i,arm]<=1<=hi[i,arm])))
    pd.DataFrame(rows).to_csv(HERE/(mode+"-intervals.csv"),index=False)
    (HERE/(mode+"-results.json")).write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,indent=2),flush=True)

def init_table(pop):
    table.initialize(pop,"entry-moment")
    table.compare.joint_fit=candidate(table.K,"multistart")
    install(table.K)
def table_task(task):
    i,j=task;records.clear()
    v,f,e=table.estimate(i,j)
    return i,j,v,f,e,diag()
def empirical():
    book=io.BytesIO(subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=ROOT))
    pop=[table.read_population(book,s) for s in table.SHEETS]
    prev=np.load(ROOT/"2.7/low-frailty/all-window-joint/multistart-table.npz")["points"]
    table.initialize(pop,"entry-moment")
    table.compare.joint_fit=candidate(table.K,"multistart")
    for i in range(2):
        for d in (0,501,999):
            np.testing.assert_allclose(table.estimate(i,d)[0],prev[i,d],rtol=1e-10,atol=1e-12)
    data=np.empty_like(prev);ds=np.empty((2,1000,5))
    with ProcessPoolExecutor(5,initializer=init_table,initargs=(pop,)) as pool:
        for n,(i,j,v,f,e,d) in enumerate(pool.map(table_task,[(i,j) for j in range(1000) for i in range(2)],chunksize=4),1):
            data[i,j]=v;ds[i,j]=np.r_[f,e,d]
            if n%200==0:print("Table8",n,"/2000",flush=True)
    assert np.isfinite(data).all() and ds[:,:,4].max()<1e-10
    np.savez_compressed(HERE/"table8-draws.npz",points=data,diagnostics=ds)
    rows=[]
    for i,s in enumerate(table.SHEETS):
        for j,pair in enumerate(("1/0","2/0","2/1")):
            for label,v in (("approximate",prev[i,:,j]),("weekly_survival",data[i,:,j]),("paired_change",data[i,:,j]-prev[i,:,j])):
                lo,hi=np.quantile(v[1:],[.025,.975])
                rows.append(dict(enrollment=s,pair=pair,method=label,point=v[0],lower95=lo,upper95=hi,width=hi-lo))
    pd.DataFrame(rows).to_csv(HERE/"table8-comparison.csv",index=False)
    print(pd.DataFrame(rows).round(6).to_string(index=False),flush=True)
    print("diagnostics",ds.sum(axis=1),flush=True)
if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="table":empirical()
    else:synth(mode)

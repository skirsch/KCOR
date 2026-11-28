"""Matched synthetic and Table 8 replays of nonlinear survivor-law fitting."""
import sys,inspect,io,json,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.stats import binomtest
from fitter import HERE,ROOT,make_fitter,make_correction,candidate
sys.path.insert(0,str(ROOT/"1.3/multiwindow-coverage"))
import study
import table8_bootstrap as table

def configure(mode):
    if mode=="stress":
        study.POPULATION=50000;study.WINDOWS=((2,5),(35,52));study.SEED=20261126
    study.initialize()
    model=study.model
    study.compare.joint_fit=make_fitter(model,(study.WAVE[0]-study.skip,study.WAVE[1]-study.skip),1.18)
    model.apply_nph_correction_post_inversion=make_correction(model)
    global estimate_one
    src=inspect.getsource(study.estimate)
    old="for fit in (compare.production_fit, compare.joint_fit):"
    assert src.count(old)==1
    ns=vars(study).copy()
    exec(compile(src.replace(old,"for fit in (compare.joint_fit,):"),"<exact survivor fit replay>","exec"),ns)
    estimate_one=ns["estimate"]

def task(job):
    rep,draws,full=job
    batch,_=study.samples("nonlinear_wave",rep,full)
    w=study.pooling_weights(batch)
    return rep,np.asarray([estimate_one(c,w,"nonlinear_wave")[0] for c in batch[:draws+1]]),w
def synthetic(mode):
    configure(mode)
    reps,draws,full=(50,99,99) if mode=="stress" else (100,99,199)
    p,_=study.population("nonlinear_wave");counts=p*study.POPULATION/4
    expected=estimate_one(counts,study.pooling_weights(counts[None]),"nonlinear_wave")[0]
    print(mode,"noiseless",expected,flush=True)
    assert abs(expected[0]-1)<1e-6
    old=np.load(ROOT/("1.2/all-window-weekly-survival/"+mode+"-draws.npz"))
    previous=old["points"][:,:draws+1]
    out=[];weights=[]
    with ProcessPoolExecutor(7,initializer=configure,initargs=(mode,)) as pool:
        for n,(rep,v,w) in enumerate(pool.map(task,[(i,draws,full) for i in range(reps)]),1):
            assert rep==len(out);out.append(v);weights.append(w)
            if n%5==0:
                print(mode,n,"/",reps,"datasets",flush=True)
                np.savez_compressed(HERE/(mode+"-draws.npz"),points=out,weights=weights)
    np.testing.assert_allclose(weights,old["weights"],rtol=0,atol=1e-13)
    combined=np.stack([previous,np.array(out)],axis=2)
    report=study.summarize(combined,1.)
    report["gamma_working_fit"]=report.pop("fixed");report["nonlinear_survivor_fit"]=report.pop("joint")
    report["expected_counts"]=expected
    report["median_width"]=np.median(np.diff(np.quantile(combined[:,1:,:,0],[.025,.975],axis=1),axis=0)[0],axis=0).tolist()
    report["mean_theta"]={name:[study.mean_interval(combined[:,0,j,i]) for i in range(6,10)] for j,name in enumerate(("gamma_working_fit","nonlinear_survivor_fit"))}
    (HERE/(mode+"-results.json")).write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    y=combined[:,0,:,0];lo,hi=np.quantile(combined[:,1:,:,0],[.025,.975],axis=1)
    rows=[]
    for i in range(reps):
        for j,label in enumerate(("gamma_working_fit","nonlinear_survivor_fit")):
            rows.append(dict(rep=i,fitter=label,point=y[i,j],lower95=lo[i,j],upper95=hi[i,j],covered=bool(lo[i,j]<=1<=hi[i,j])))
    pd.DataFrame(rows).to_csv(HERE/(mode+"-intervals.csv"),index=False)
    print(json.dumps(report,indent=2),flush=True)

def table_init(pop):
    table.initialize(pop,"entry-moment")
    table.K.apply_nph_correction_post_inversion=make_correction(table.K)
    global fits
    fits=[]
    for dates,time,quiet,keep,counts in table.DATA:
        t=time-table.K.DYNAMIC_HVE_SKIP_WEEKS
        wave=(dates>=np.datetime64(table.WAVE["start_dt"]))&(dates<=np.datetime64(table.WAVE["end_dt"]))&(t>=0)
        assert wave.any()
        fits.append(make_fitter(table.K,(t[wave][0],t[wave][-1]),table.WAVE["apply_alpha"]))
def table_task(job):
    i,j=job;params=[];flags=[]
    def logged(*a,**kw):
        p,d=fits[i](*a,**kw)
        params.append([p[0],p[1],float(d.get("theta0_raw",p[1]))])
        flags.append([float(d["success"]),float(d.get("insufficient_signal",False)),float(d.get("nonlinear_wave_deficits",0))])
        return p,d
    table.compare.joint_fit=logged
    v,f,e=table.estimate(i,j)
    return i,j,v,np.array(params),np.array(flags)
def empirical():
    book=io.BytesIO(subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=ROOT))
    pop=[table.read_population(book,s) for s in table.SHEETS]
    prev=np.load(ROOT/"1.2/all-window-weekly-survival/table8-draws.npz")["points"]
    table_init(pop)
    for i in range(2):
        q=table_task((i,0));print("Table8 point",q[0],q[2],q[3],q[4],flush=True)
    points=np.empty_like(prev);pars=np.empty((2,1000,9,3));flags=np.empty_like(pars)
    with ProcessPoolExecutor(5,initializer=table_init,initargs=(pop,)) as pool:
        for n,(i,j,v,p,f) in enumerate(pool.map(table_task,[(i,j) for j in range(1000) for i in range(2)],chunksize=2),1):
            points[i,j]=v;pars[i,j]=p;flags[i,j]=f
            if n%100==0:
                print("Table8",n,"/2000",flush=True)
                np.savez_compressed(HERE/"table8-partial.npz",draws_done=n,points=points,parameters=pars,flags=flags)
    assert np.isfinite(points).all()
    np.savez_compressed(HERE/"table8-draws.npz",points=points,parameters=pars,flags=flags)
    rows=[];params=[]
    for i,s in enumerate(table.SHEETS):
        for j,pair in enumerate(("1/0","2/0","2/1")):
            for label,v in (("gamma_working_fit",prev[i,:,j]),("nonlinear_survivor_fit",points[i,:,j]),("paired_change",points[i,:,j]-prev[i,:,j])):
                lo,hi=np.quantile(v[1:],[.025,.975])
                rows.append(dict(enrollment=s,pair=pair,fitter=label,point=v[0],lower95=lo,upper95=hi,width=hi-lo))
        for g in range(9):
            for j,label in enumerate(("k","theta","theta_raw")):
                v=pars[i,:,g,j];lo,hi=np.quantile(v[1:],[.025,.975])
                params.append(dict(enrollment=s,group=("two-dose","unvaccinated","all-persons")[g//3],pseudo_dose=g%3,parameter=label,point=v[0],lower95=lo,upper95=hi))
    pd.DataFrame(rows).to_csv(HERE/"table8-comparison.csv",index=False)
    pd.DataFrame(params).to_csv(HERE/"table8-parameters.csv",index=False)
    print(pd.DataFrame(rows).round(6).to_string(index=False),flush=True)
    print("failed and flags",(1-flags[:,:,:,0]).sum(axis=(1,2)),flags[:,:,:,1].sum(axis=(1,2)),flush=True)
if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="table":empirical()
    else:synthetic(mode)

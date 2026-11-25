"""Combine the previous 3.2 entry trial with the validated all-window candidate."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
from pathlib import Path
import sys,inspect,io,json,subprocess,importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import t
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
# Import the original synthetic module with its own experiment module resolved first.
import synthetic as syn
table=syn.table
# Load the all-window trial under a unique name; it shares table8_bootstrap.
spec=importlib.util.spec_from_file_location("all_window_source",ROOT/"2.7/low-frailty/all-window-joint/experiment.py")
allwindow=importlib.util.module_from_spec(spec);spec.loader.exec_module(allwindow)

def entry_source(model):
    source=inspect.getsource(model.fit_theta0_gompertz)
    seed="            H_gom_full = k_fixed * unit_start"
    boundaries="    reentry_indices = [start for start, _ in segments[1:]]"
    assert source.count(seed)==source.count(boundaries)==1
    source=source.replace(seed,"""
            if first_seg_start > post_idx[0]:
                before = post_idx[post_idx < first_seg_start]
                observed = float(np.sum(h_arr[before]))
                log_ratio = theta_seed * observed - np.log1p(
                    theta_seed * k_fixed * unit_start[first_seg_start])
                k_fixed = np.exp(min(np.log(0.1), np.log(k_fixed) + max(0., log_ratio)))
"""+seed)
    source=source.replace(boundaries,boundaries+"""
    if joint_seed and first_seg_start > post_idx[0]:
        reentry_indices.insert(0, first_seg_start)
""")
    return source

def combined(model):
    src=inspect.getsource(allwindow.candidate)
    original="source=inspect.getsource(model.fit_theta0_gompertz)"
    assert src.count(original)==1
    ns=dict(vars(allwindow),entry_source=entry_source)
    exec(compile(src.replace(original,"source=entry_source(model)"),"<combined 3.2 and all-window factory>","exec"),ns)
    return ns["candidate"](model,"multistart")

def initialize_synthetic():
    global fits,model
    table.compare.initialize("entry-moment")
    model=table.compare.adapter.K
    fits=(allwindow.candidate(model,"multistart"),combined(model))

def synth_task(i):
    runs=[]
    for rep in range(200):
        rng=np.random.default_rng([32022,i,rep])
        r=syn.run_case(syn.CASES[i],fits,model,rng)
        runs.append(r)
    return i,runs

def mean_ci(v):
    v=np.asarray(v);m=v.mean(axis=0);d=t.ppf(.975,len(v)-1)*v.std(axis=0,ddof=1)/np.sqrt(len(v))
    return np.stack((m,m-d,m+d),axis=-1).tolist()

def synthetic():
    initialize_synthetic()
    no=[syn.run_case(c,fits,model) for c in syn.CASES]
    for r in no:
        print(r["case"],r["target"],r["ratio"], "theta/k", [[(c["theta"],c["k"]) for c in arm] for arm in r["cohorts"]],flush=True)
    record=[];archive={}
    with ProcessPoolExecutor(3,initializer=initialize_synthetic) as pool:
        for i,runs in pool.map(synth_task,(2,5,6)):
            v=np.array([r["ratio"] for r in runs]);truth=runs[0]["target"]
            params=np.array([[[[c["k"],c["theta"],c["growth"]] for c in arm] for arm in r["cohorts"]] for r in runs])
            err=np.abs(v-truth);delta=err[:,1]-err[:,0]
            record.append(dict(case=syn.CASES[i][0],truth=truth,reps=len(v),mean_ci=mean_ci(v),central95=np.quantile(v,[.025,.975],axis=0).T.tolist(),mean_abs_error_ci=mean_ci(err),mean_paired_abs_error_change_ci=mean_ci(delta),
                improved_tied_worse=[int((delta<0).sum()),int((delta==0).sum()),int((delta>0).sum())],
                theta_k_growth_mean_ci=mean_ci(params),theta_k_growth_central95=np.moveaxis(np.quantile(params,[.025,.975],axis=0),0,-1).tolist(),
                failed=[sum(not c["success"] for rr in runs for c in rr["cohorts"][arm]) for arm in range(2)]))
            archive[f"case_{i}_ratios"]=v;archive[f"case_{i}_parameters"]=params
            print("sampled",syn.CASES[i][0],record[-1]["mean_ci"],record[-1]["mean_paired_abs_error_change_ci"],record[-1]["improved_tied_worse"],flush=True)
    np.savez_compressed(HERE/"synthetic-draws.npz",**archive)
    (HERE/"synthetic-results.json").write_text(json.dumps(dict(noiseless=no,sampled=record),indent=2)+"\n",encoding="utf8")

def table_init(pop):
    global tablefits
    table.initialize(pop,"entry-moment")
    tablefits=(allwindow.candidate(table.K,"multistart"),combined(table.K))

def draw_task(t):
    i,d=t
    results=[];pars=[]
    for fit in tablefits:
        params=[]
        def record(*a,**kw):
            p,diag=fit(*a,**kw)
            params.append([p[0],p[1],int(diag["success"])])
            return p,diag
        table.compare.joint_fit=record
        results.append(table.estimate(i,d))
        pars.append(params)
    return i,d,results,np.array(pars)

def empirical():
    book=subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=ROOT)
    pop=[table.read_population(io.BytesIO(book),s) for s in table.SHEETS]
    before=np.load(ROOT/"2.7/low-frailty/all-window-joint/multistart-table.npz")["points"]
    table_init(pop)
    # Both are identical if first quiet window starts at modeled origin.
    for d in (0,1,99,501,999):
        i,j,r,p=draw_task((0,d))
        np.testing.assert_allclose(r[0][0],before[0,d],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(r[1][0],r[0][0],rtol=1e-12,atol=1e-13)
    result=np.empty((1000,2,3));flags=np.empty((1000,2,2));pars=np.empty((1000,2,9,3))
    with ProcessPoolExecutor(7,initializer=table_init,initargs=(pop,)) as pool:
        for n,(i,d,r,p) in enumerate(pool.map(draw_task,[(1,j) for j in range(1000)],chunksize=5),1):
            result[d]=[x[0] for x in r];flags[d]=[x[1:] for x in r];pars[d]=p
            if n%100==0:print("Table 8",n,"/1000 paired draws",flush=True)
    np.testing.assert_allclose(result[:,0],before[1],rtol=1e-10,atol=1e-12)
    np.savez_compressed(HERE/"table8-draws.npz",points=result,parameters=pars,flags=flags)
    rows=[]
    for j,pair in enumerate(("1/0","2/0","2/1")):
        for arm,label in enumerate(("all_window","entry_plus_all_window")):
            v=result[:,arm,j];lo,hi=np.quantile(v[1:],[.025,.975])
            rows.append(dict(enrollment="2022_06",pair=pair,fitter=label,point=v[0],lower95=lo,upper95=hi,width=hi-lo))
        v=result[:,1,j]-result[:,0,j];lo,hi=np.quantile(v[1:],[.025,.975])
        rows.append(dict(enrollment="2022_06",pair=pair,fitter="paired_change",point=v[0],lower95=lo,upper95=hi,width=hi-lo))
    pd.DataFrame(rows).to_csv(HERE/"table8-summary.csv",index=False)
    param=[]
    for arm,label in enumerate(("all_window","entry_plus_all_window")):
        for i in range(9):
            for j,name in enumerate(("k","theta")):
                v=pars[:,arm,i,j];lo,hi=np.quantile(v[1:],[.025,.975])
                param.append(dict(fitter=label,group=("two-dose","unvaccinated","all-persons")[i//3],pseudo_dose=i%3,parameter=name,point=v[0],lower95=lo,upper95=hi))
    pd.DataFrame(param).to_csv(HERE/"table8-parameters.csv",index=False)
    print(pd.DataFrame(rows).round(6).to_string(index=False),"flags and failures",flags.sum(axis=0),flush=True)
if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="synthetic":synthetic()
    else:empirical()


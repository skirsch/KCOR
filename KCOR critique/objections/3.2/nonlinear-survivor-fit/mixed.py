"""Earlier mixed-window experiments, with both correction arms using exact wave inversion."""
import importlib.util,sys,json
from pathlib import Path
import numpy as np
from scipy.stats import t
from concurrent.futures import ProcessPoolExecutor
from fitter import HERE,ROOT,make_fitter,make_correction,candidate
# Reuse original 3.2 generator and endpoint calculation, without altering files.
saved=sys.modules.get("experiment")
spec=importlib.util.spec_from_file_location("entry_experiment",HERE.parent/"experiment.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sys.modules["experiment"]=old
spec=importlib.util.spec_from_file_location("entry_synthetic",HERE.parent/"synthetic.py")
syn=importlib.util.module_from_spec(spec);spec.loader.exec_module(syn)
if saved is not None: sys.modules["experiment"]=saved
# Reuse the prior combined all-window plus entry fitter.
spec=importlib.util.spec_from_file_location("entry_combined_replay",HERE.parent/"all-window-entry/replay.py")
prior=importlib.util.module_from_spec(spec)
sys.modules["synthetic"]=syn
spec.loader.exec_module(prior)

def initialize():
    global model
    syn.table.compare.initialize("entry-moment")
    model=syn.table.compare.adapter.K
    model.apply_nph_correction_post_inversion=make_correction(model)
def pair(case):
    alpha=case[1];last=case[5]
    return (prior.combined(model),make_fitter(model,(0,last),alpha))
def task(i):
    case=syn.CASES[i];fits=pair(case)
    return i,[syn.run_case(case,fits,model,np.random.default_rng([32022,i,rep])) for rep in range(200)]
def ci(x):
    a=np.asarray(x);m=a.mean(axis=0);d=t.ppf(.975,len(a)-1)*a.std(axis=0,ddof=1)/np.sqrt(len(a))
    return np.stack((m,m-d,m+d),axis=-1).tolist()
def main():
    initialize()
    records=[]
    for case in syn.CASES[3:]:
        r=syn.run_case(case,pair(case),model)
        print("noiseless",r["case"],r["target"],r["ratio"],[[[q["k"],q["theta"]] for q in arm] for arm in r["cohorts"]],flush=True)
        records.append(r)
    sampled=[];draws={}
    with ProcessPoolExecutor(3,initializer=initialize) as pool:
        for i,results in pool.map(task,(3,4,5,6)):
            y=np.array([r["ratio"] for r in results]);truth=results[0]["target"];errors=abs(y-truth);delta=errors[:,1]-errors[:,0]
            p=np.array([[[[q["k"],q["theta"],q["growth"]] for q in arm] for arm in r["cohorts"]] for r in results])
            row=dict(case=syn.CASES[i][0],truth=truth,mean_ci=ci(y),central95=np.quantile(y,[.025,.975],axis=0).T.tolist(),
                    paired_abs_error_change=ci(delta),improved_tied_worse=[int((delta<0).sum()),int((delta==0).sum()),int((delta>0).sum())],
                    parameters_mean_ci=ci(p),failures=[sum(not q["success"] for r in results for q in r["cohorts"][j]) for j in range(2)])
            print("sampled",row["case"],row["mean_ci"],row["improved_tied_worse"],flush=True)
            sampled.append(row);draws[f"scenario_{i}_ratios"]=y;draws[f"scenario_{i}_params"]=p
    np.savez_compressed(HERE/"mixed-draws.npz",**draws)
    (HERE/"mixed-results.json").write_text(json.dumps(dict(noiseless=records,sampled=sampled),indent=2)+"\n",encoding="utf8")
if __name__=="__main__":main()

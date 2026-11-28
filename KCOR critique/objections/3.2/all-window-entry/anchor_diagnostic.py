"""Diagnostic only: remove nonquiet early anchor rows from the fitting residual.
The early normalization of KCOR itself is unchanged.
"""
import replay as r
from scipy.stats import t
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
def source(model):
    s=r.entry_source(model)
    old="fit_mask_theta = (quiet | anchor_mask) & post_skip"
    assert s.count(old)==1
    return s.replace(old,"fit_mask_theta = quiet & post_skip")
def diagnostic(model):
    src=r.inspect.getsource(r.allwindow.candidate)
    o="source=inspect.getsource(model.fit_theta0_gompertz)"
    assert src.count(o)==1
    ns=dict(vars(r.allwindow),source_trial=source)
    exec(compile(src.replace(o,"source=source_trial(model)"),"<quiet-only diagnostic>","exec"),ns)
    return ns["candidate"](model,"multistart")
def initialize():
    global fits
    r.initialize_synthetic()
    fits=(r.combined(r.model),diagnostic(r.model))
def task(i):
    results=[]
    for rep in range(200):
        rng=np.random.default_rng([32022,i,rep])
        results.append(r.syn.run_case(r.syn.CASES[i],fits,r.model,rng))
    return i,results
def main():
    initialize()
    no=[r.syn.run_case(c,fits,r.model) for c in r.syn.CASES]
    # Control with the same short early pulses but alpha=1.
    c=r.syn.CASES[4]
    no.append(r.syn.run_case(("proportional short early pulses",1.,*c[2:]),fits,r.model))
    for v in no:print("noiseless",v["case"],v["target"],v["ratio"],flush=True)
    records=[]
    with ProcessPoolExecutor(3,initializer=initialize) as pool:
        for i,res in pool.map(task,(2,5,6)):
            v=np.array([x["ratio"] for x in res]);target=res[0]["target"];err=abs(v-target);d=err[:,1]-err[:,0]
            out=dict(case=r.syn.CASES[i][0],target=target,mean_ci=r.mean_ci(v),central95=np.quantile(v,[.025,.975],axis=0).T.tolist(),paired_abs_error_change_ci=r.mean_ci(d),improved_tied_worse=[int((d<0).sum()),int((d==0).sum()),int((d>0).sum())])
            records.append(out);print(out,flush=True)
    (r.HERE/"anchor-diagnostic.json").write_text(json.dumps(dict(noiseless=no,sampled=records),indent=2)+"\n",encoding="utf8")
if __name__=="__main__":main()

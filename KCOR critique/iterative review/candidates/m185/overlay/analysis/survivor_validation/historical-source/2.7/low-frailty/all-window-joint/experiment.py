"""Isolated common-k all-window experiment; no production edits."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import sys, inspect, io, subprocess, json
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/"1.3"),str(ROOT/"1.3/multiwindow-coverage")]
import table8_bootstrap as table
import study
import fallback_stress as stress

def candidate(model, mode):
    source=inspect.getsource(model.fit_theta0_gompertz)
    a=source.index("        # 3c:")
    b=source.index("        except Exception as e:",a)
    source=source[:a]+'''        # Trial: jointly update the common k and theta.
        def _residual_joint(params):
            k, th = np.exp(params[0]), params[1]
            offset = Delta
            if PROFILE:
                offset = np.zeros_like(h_arr)
                cumulative = np.zeros_like(h_arr)
                cumulative[post_idx] = np.r_[0., np.cumsum(np.maximum(np.nan_to_num(h_arr[post_idx]),0.))[:-1]]
                inv = np.expm1(th*cumulative)/th if th else cumulative
                prev = 0.
                for start in reentry_indices:
                    raw = inv[start] - k*unit_start[start]
                    offset[start:] += max(raw-prev,0.)
                    prev = raw
            pred = weekly_gamma_hazard(k*unit_start[fit_mask_theta]+offset[fit_mask_theta], k*unit_week[fit_mask_theta], th)
            return h_fit-pred
        try:
            r = least_squares(_residual_joint, [np.log(k_fixed), min(max(theta,0.),theta0_max)],
                bounds=([np.log(1e-8),0.],[np.log(.1),theta0_max]),
                ftol=1e-14, xtol=1e-14, gtol=1e-14)
            if MULTISTART:
                low = least_squares(_residual_joint, [np.log(k_fixed), 0.5],
                    bounds=([np.log(1e-8),0.],[np.log(.1),theta0_max]),
                    ftol=1e-14, xtol=1e-14, gtol=1e-14)
                if low.cost < r.cost:
                    r = low
            old_k = k_fixed
            k_fixed = float(np.exp(r.x[0]))
            H_gom_full = k_fixed*unit_start
            week_fit = k_fixed*unit_week[fit_mask_theta]
''' +source[b:]
    source=source.replace("theta_new = float(r.x[0])","theta_new = float(r.x[1])")
    source=source.replace("if abs(theta_new - theta) < convergence_tol:", "if (PROFILE and r.success) or (abs(theta_new - theta) < convergence_tol and abs(np.log(k_fixed/old_k)) < convergence_tol):")
    ns=dict(vars(model),PROFILE=(mode in ("profile","multistart")),MULTISTART=(mode=="multistart"))
    exec(compile(source,"<experimental all-window joint fit>","exec"),ns)
    return partial(ns["fit_theta0_gompertz"],joint_seed=True)

def table_init(pop,mode):
    table.initialize(pop,"entry-moment")
    table.compare.joint_fit=candidate(table.K,mode)

def table_task(task):
    i,j=task
    return i,j,table.estimate(i,j)

def synth_init(mode):
    stress.initialize()
    trial=candidate(study.model,mode)
    study.compare.production_fit=study.compare.joint_fit
    study.compare.joint_fit=trial

def synth_task(rep):
    batch,_=study.samples("nonlinear_wave",rep,99)
    weights=study.pooling_weights(batch)
    return rep,np.array([study.estimate(c,weights,"nonlinear_wave") for c in batch])

def main():
    mode=sys.argv[1]
    book=io.BytesIO(subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=ROOT))
    pop=[table.read_population(book,s) for s in table.SHEETS]
    table_init(pop,mode)
    print(mode,"table points", [table.estimate(i,0) for i in range(2)],flush=True)
    synth_init(mode)
    probs,_=study.population("nonlinear_wave")
    exact=probs*study.POPULATION/4
    print(mode,"exact",study.estimate(exact,study.pooling_weights(exact[None]),"nonlinear_wave"),flush=True)
    if len(sys.argv)>2 and sys.argv[2]=="pilot":
        values=synth_task(34)[1]
        np.savez_compressed(HERE/(mode+"-pilot.npz"),points=values)
        print("stress 34",values[0],np.quantile(values[1:,:,0],[.025,.975],axis=0),flush=True)
        return
    data=np.empty((2,1000,3)); flags=np.empty((2,1000,2))
    with ProcessPoolExecutor(7,initializer=table_init,initargs=(pop,mode)) as pool:
        for n,(i,j,(v,f,e)) in enumerate(pool.map(table_task,[(i,j) for j in range(1000) for i in range(2)],chunksize=5),1):
            data[i,j]=v; flags[i,j]=f,e
            if n%200==0: print("table",n,flush=True)
    np.savez_compressed(HERE/(mode+"-table.npz"),points=data,flags=flags)
    print("table summary",data[:,0],np.quantile(data[:,1:],[.025,.975],axis=1),flush=True)
    with ProcessPoolExecutor(7,initializer=synth_init,initargs=(mode,)) as pool:
        data=[]
        for rep,v in pool.map(synth_task,range(50)):
            data.append(v)
            if (rep+1)%5==0: print("stress",rep+1,flush=True)
    np.savez_compressed(HERE/(mode+"-stress.npz"),points=data)
    print(json.dumps(study.summarize(np.array(data),1.),indent=2),flush=True)
if __name__=="__main__": main()


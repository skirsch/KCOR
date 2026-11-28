"""Paired numerical-rescaling checks: exact hazards and original zero-contamination draws."""
from pathlib import Path
import os,sys,importlib.util,json,argparse
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="identified_only")
import numpy as np,pandas as pd
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"test/quiet_window_contamination")]
import KCOR as new
import run_contamination_test as study
def load_old(path):
    spec=importlib.util.spec_from_file_location("old_KCOR",path/"code/KCOR.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def ci(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    m=x.mean();d=student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return [float(m),float(m-d),float(m+d)]
def main():
    parser=argparse.ArgumentParser();parser.add_argument("previous_candidate",type=Path);args=parser.parse_args()
    old=load_old(args.previous_candidate)
    gamma=.085/52
    cases=[]
    for k in (1e-6,1e-5,1e-4,.001):
        for th in (.5,4.,50.):
            for n in (12,52):
                t=np.arange(n,dtype=float)
                H=k*np.expm1(gamma*np.arange(n+1))/gamma
                h=np.diff(np.log1p(th*H)/th)
                row=dict(k_true=k,theta_true=th,weeks=n)
                for label,K in (("before",old),("after",new)):
                    (kf,tf),dg=K.fit_theta0_gompertz(h,t,np.ones(n,bool),4,gamma)
                    row.update({label+"_k":kf,label+"_theta":tf,label+"_status":dg["status"],label+"_rmse_hazard":dg["rmse_hazard"]})
                cases.append(row)
    exact=pd.DataFrame(cases)
    exact.to_csv(HERE/"noiseless-grid.csv",index=False)
    sample=exact[(exact.k_true==1e-5)&(exact.theta_true==50)&(exact.weeks==12)].iloc[0]
    assert abs(sample.after_theta-50)<1e-5
    print("Rare-rate counterexample",sample.to_dict(),flush=True)
    print("Maximum exact-grid absolute theta error",abs(exact.before_theta-exact.theta_true).max(),abs(exact.after_theta-exact.theta_true).max(),flush=True)
    # Original zero-contamination generator, seed stream, and historical
    # experiment anchoring convention; current and previous estimators see
    # exactly the same histories. This is not a newly designed experiment.
    k=.00045616902672594684;skip=new.DYNAMIC_HVE_SKIP_WEEKS
    weeks,rebased,quiet=study.build_time_arrays(study.N_WEEKS,skip)
    anchors=new.get_gompertz_theta_fit_config()["k_anchor_weeks"]
    rows=[]
    for size in (50000,500000,5000000):
        for rep in range(200):
            rng=np.random.default_rng(np.random.SeedSequence([10042,2,0,0,rep]))
            histories=[study.simulate_one_cohort(rng,size,study.N_WEEKS,k,gamma,theta,skip,None) for theta in (.5,4.)]
            for label,K in (("before",old),("after",new)):
                rec=dict(size=size,rep=rep,mode=label)
                growth=[]
                for arm,(alive,dead) in zip("ab",histories):
                    h=K.hazard_from_mr(dead/alive);eff=np.where(weeks>=skip,h,0.)
                    (kf,tf),d=K.fit_theta0_gompertz(eff,rebased,quiet,anchors,gamma,deaths_arr=dead,risk_arr=alive)
                    rec.update({f"theta_raw_{arm}":float(d.get("theta0_raw",tf)),f"theta_applied_{arm}":float(tf),f"k_{arm}":float(kf),f"status_{arm}":d["status"],f"success_{arm}":bool(d["success"]),f"flag_{arm}":bool(d.get("insufficient_signal"))})
                    applied=tf if d["success"] and np.isfinite(tf) else 0.
                    H=K.invert_gamma_frailty(np.cumsum(eff),applied)
                    growth.append(H[-1]/H[skip+4])
                rec["KCOR"]=growth[0]/growth[1]
                rows.append(rec)
        print("Completed size",size,flush=True)
        pd.DataFrame(rows).to_csv(HERE/"zero-contamination-paired.csv",index=False)
    f=pd.DataFrame(rows);summary=[]
    for (size,mode),g in f.groupby(["size","mode"]):
        summary.append(dict(size=int(size),mode=mode,datasets=len(g),KCOR_mean_ci=ci(g.KCOR),KCOR_sample_quantiles=np.quantile(g.KCOR,[.025,.975]).tolist(),theta_a_mean_ci=ci(g.theta_raw_a),theta_a_sample_quantiles=np.quantile(g.theta_raw_a,[.025,.975]).tolist(),theta_b_mean_ci=ci(g.theta_raw_b),theta_b_sample_quantiles=np.quantile(g.theta_raw_b,[.025,.975]).tolist(),failed=int((~g.success_a|~g.success_b).sum()),flagged=int((g.flag_a|g.flag_b).sum())))
    (HERE/"low-frailty-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()

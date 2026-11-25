"""Paired replay of existing contamination experiment, unchanged generator and anchor."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import t,ttest_ind
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from compare_fitters import study,edition,STUDY
from comorbidity_sweep import summarize
from diagnose_null import RecordProbabilities
sys.path.insert(0,str(PARENT/"low-frailty/all-window-joint"))
from experiment import candidate
N=200
SIZES=(50000,500000,5000000)

def setup():
    edition.initialize("entry-moment")
    original=study.K.fit_theta0_gompertz
    current=edition.joint_fit
    trial=candidate(edition.adapter.K,"multistart")
    meta=pd.read_csv(STUDY/"out/quiet_contamination_meta.csv",index_col=0)["value"]
    k,gamma=map(float,(meta.k_simulator,meta.gamma_per_week))
    skip=study.K.DYNAMIC_HVE_SKIP_WEEKS
    weeks,rebased,quiet=study.build_time_arrays(study.N_WEEKS,skip)
    anchors=study.K.get_gompertz_theta_fit_config()["k_anchor_weeks"]
    return original,current,trial,k,gamma,skip,weeks,rebased,quiet,anchors

def run(job):
    size,level,start,end=job
    original,current,trial,k,gamma,skip,weeks,rebased,quiet,anchors=setup()
    fits={"current_joint":current,"all_window":trial}
    if size==SIZES[0]:fits["original"]=original
    truth=(study.THETA0_A,study.THETA0_B)
    def contamination(week):
        return level if study.QUIET_T_REBASED_MIN<=week<=study.QUIET_T_REBASED_MAX else 0.
    def generate(rng,theta,fn):
        return study.simulate_one_cohort(rng,size,study.N_WEEKS,k,gamma,theta,skip,fn)
    reference=[]
    for theta,fn in zip(truth,(contamination,None)):
        stream=RecordProbabilities()
        generate(stream,theta,fn)
        prob=np.array(stream.probabilities)
        alive=size*np.r_[1.,np.cumprod(1.-prob[:-1])]
        reference.append((alive,alive*prob))
    exact_ref=[]
    for (a,d),theta in zip(reference,truth):
        h=study.K.hazard_from_mr(d/a)
        H=np.cumsum(np.where(weeks>=skip,h,0.))
        exact_ref.append(study.K.invert_gamma_frailty(H,theta))
    pop_anchor=exact_ref[0][skip+4]/exact_ref[1][skip+4]
    out=[]
    for rep in range(start,end):
        if rep<0:
            samples=reference
        else:
            code=round(float(level)*1e12)%(2**31)
            rng=np.random.default_rng(np.random.SeedSequence([10042,2,0,code,rep]))
            samples=[generate(rng,theta,fn) for theta,fn in zip(truth,(contamination,None))]
        for label,fit in fits.items():
            with patch.object(study.K,"fit_theta0_gompertz",fit):
                fitted=[study.cohort_pipeline(d,a,weeks,rebased,quiet,anchors,gamma) for a,d in samples]
            curves=[(study.K.invert_gamma_frailty(v[3],theta),v[4]) for v,theta in zip(fitted,truth)]
            r=dict(size=size,epsilon_comorb=level,rep=rep,fitter=label)
            for arm,v in zip("ab",fitted):
                r.update({f"k_{arm}":v[0],f"theta_{arm}":v[1],f"status_{arm}":v[2]["status"],f"success_{arm}":v[2]["success"],f"flag_{arm}":v[2].get("insufficient_signal",False)})
            for kind,i,j in (("oracle",0,0),("fit_a",1,0),("fit_b",0,1),("fitted",1,1)):
                a,b=curves[0][i],curves[1][j]
                r[kind+"_sample_anchor"]=study.kcor_asymptote_value(study.anchored_kcor(a,b,skip),b)
                r[kind+"_population_anchor"]=a[-1]/b[-1]/pop_anchor
            r["KCOR_asymptote"]=r["fitted_sample_anchor"] if r["success_a"] and r["success_b"] else np.nan
            out.append(r)
    return out

def ci(x):
    x=np.asarray(x);m=x.mean();d=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return m,m-d,m+d

def main():
    jobs=[(size,level,start,min(start+25,N)) for size in SIZES for level in study.EPSILON_COMORB for start in range(0,N,25)]
    jobs+=[(size,level,-1,0) for size in SIZES for level in study.EPSILON_COMORB]
    out=[]
    with ProcessPoolExecutor(7) as pool:
        for i,rows in enumerate(pool.map(run,jobs),1):
            out.extend(rows)
            if i%15==0: print(f"{i}/{len(jobs)} batches",flush=True)
    frame=pd.DataFrame(out)
    frame.to_csv(HERE/"runs.csv",index=False)
    r=frame[frame.rep>=0].copy()
    keys=["epsilon_comorb","rep"]
    archived=pd.read_csv(PARENT/"comorbidity_200_runs.csv").set_index(keys).sort_index()
    previous=pd.read_csv(PARENT/"fitter_comparison_runs.csv")
    orig=r[(r["size"]==SIZES[0])&(r.fitter=="original")].set_index(keys).sort_index()
    curr=r[(r["size"]==SIZES[0])&(r.fitter=="current_joint")].set_index(keys).sort_index()
    np.testing.assert_allclose(orig.KCOR_asymptote,archived.KCOR_asymptote,rtol=0,atol=1e-8)
    jc=previous[previous.fitter=="revised_joint"].set_index(keys).sort_index()
    np.testing.assert_allclose(curr.KCOR_asymptote,jc.KCOR_asymptote,rtol=0,atol=1e-8)
    print("Archived original and previous joint endpoints reproduced.",flush=True)
    assert r.KCOR_asymptote.notna().all()
    stats=[];pars=[];null=[];paired=[]
    for (size,fitter),g in r.groupby(["size","fitter"]):
        stats.append(summarize(g).assign(size=size,fitter=fitter))
        for level,h in g.groupby("epsilon_comorb"):
            for name in ("theta_a","theta_b","k_a","k_b"):
                x=h[name];m,lo,hi=ci(x)
                qlo,qhi=np.quantile(x,[.025,.975])
                pars.append(dict(size=size,fitter=fitter,level=level,parameter=name,mean=m,lower95=lo,upper95=hi,median=x.median(),q025=qlo,q975=qhi,near_zero=int((x<1e-6).sum())))
            if level==0:
                for key in ("oracle_sample_anchor","oracle_population_anchor","fit_a_sample_anchor","fit_b_sample_anchor","fitted_sample_anchor","fitted_population_anchor"):
                    m,lo,hi=ci(h[key]);null.append(dict(size=size,fitter=fitter,measure=key,mean=m,lower95=lo,upper95=hi))
                for key,delta in (("anchor_oracle",h.oracle_sample_anchor-h.oracle_population_anchor),("fit_a_effect",h.fit_a_sample_anchor-h.oracle_sample_anchor),("fit_b_effect",h.fit_b_sample_anchor-h.oracle_sample_anchor),("both_fits_effect",h.fitted_sample_anchor-h.oracle_sample_anchor)):
                    m,lo,hi=ci(delta);null.append(dict(size=size,fitter=fitter,measure=key,mean=m,lower95=lo,upper95=hi))
    for (size,level),g in r.groupby(["size","epsilon_comorb"]):
        piv=g.pivot(index="rep",columns="fitter",values="KCOR_asymptote")
        for prior in ("original","current_joint"):
            if prior in piv:
                m,lo,hi=ci(piv.all_window-piv[prior])
                paired.append(dict(size=size,level=level,prior=prior,mean=m,lower95=lo,upper95=hi))
    s=pd.concat(stats,ignore_index=True)
    s.to_csv(HERE/"endpoint-summary.csv",index=False)
    pd.DataFrame(pars).to_csv(HERE/"parameter-summary.csv",index=False)
    pd.DataFrame(null).to_csv(HERE/"null-decomposition.csv",index=False)
    pd.DataFrame(paired).to_csv(HERE/"paired-changes.csv",index=False)
    print(s[["size","fitter","added_weekly_hazard","mean","lower95","upper95","difference","difference_lower95","difference_upper95"]].round(5).to_string(index=False))
if __name__=="__main__":main()

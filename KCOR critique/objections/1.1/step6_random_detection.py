"""Finite-count residual detection for the existing constant-harm case only.

An exploratory gate is fixed here before alternative runs. Its threshold is
calibrated on separate no-effect simulations. It is NOT the manuscript's full
diagnostic protocol. No production or prior audit files are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step1_contamination_audit import ROOT, SOURCE, load_experiment


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wilson(count, n):
    if not n:
        return [None, None]
    z=1.959963984540054
    p=count/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-width,center+width]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step5",type=Path,required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--calibration-reps",type=int,default=200)
    parser.add_argument("--evaluation-reps",type=int,default=200)
    parser.add_argument("--seed",type=int,default=20261109)
    args=parser.parse_args()
    assert min(args.calibration_reps,args.evaluation_reps)>=100
    prior=json.loads((args.step5/"manifest.json").read_text())
    for path,checksum in prior["source_hashes"].items():
        assert sha(ROOT/path)==checksum,f"Prior input changed: {path}"
    E=load_experiment()
    K=E.K
    d=prior["original_design"]
    k,gamma=d["mortality_scale"],d["gamma_per_week"]
    n0,nweeks,skip=d["n_initial"],d["n_weeks"],d["skip_weeks"]
    cfg=K.get_gompertz_theta_fit_config()
    t,tr,_=E.build_time_arrays(nweeks,skip)
    quiet=(tr>=d["quiet_start_rebased"])&(tr<=d["quiet_end_rebased"])
    b=k*np.exp(gamma*np.maximum(0,tr))
    Bpost=np.where(tr>=0,b,0.).sum()
    amount=float(E.EPSILON_COMORB_POS.min())
    cases={"no_effect":None,"baseline_clean":5,"baseline_affected":0}
    targets={name:1+(53*amount/Bpost if start is not None else 0) for name,start in cases.items()}
    sources=[SOURCE,ROOT/"code/KCOR.py",ROOT/"data/Czech/Czech.yaml",Path(__file__).with_name("step1_contamination_audit.py"),
             args.step5/"manifest.json",args.step5/"baseline_results.csv"]
    design=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,
                calibration_reps=args.calibration_reps,evaluation_reps=args.evaluation_reps,
                cohort_size=n0,n_weeks=nweeks,quiet_rebased=[d["quiet_start_rebased"],d["quiet_end_rebased"]],
                profiles={name:{"start_rebased":start,"duration":53 if start is not None else 0,"added_weekly_hazard":amount if start is not None else 0} for name,start in cases.items()},
                theta_A=d["theta_A"],theta_B=d["theta_B"],targets=targets,
                statistic="For quiet-week binomial standardized residuals z: max(mean(z^2), (sum(z)/sqrt(n))^2, (dot(centered time,z)/norm(centered time))^2), maximized across both cohorts.",
                threshold_rule="95th empirical quantile (higher order statistic) of separate no-effect calibration scores; frozen before evaluation.",
                pass_rule="Both fitters report success and no identifiability/insufficient-signal/deaths/bad-fit flag, plus score <= fixed threshold.",
                large_error_rule="Absolute relative error >10% in raw final contrast, reported both among accepted and among all runs; an exploratory reporting threshold, not a manuscript cutoff.",
                pairing="Independent calibration/evaluation seed branches. Evaluation cases share per-repetition cohort seeds; comparator draws/fits are identical across cases.",
                limits=["This is only an exploratory residual gate, not full manuscript diagnostic acceptance.",
                        "Calibration is specific to this known synthetic null, cohort size, and nuisance parameters.",
                        "No parameter-window/skip/gamma perturbations inside each random run; no formal low-theta gate.",
                        "Only the inherited constant effect and its two placements are evaluated; plausibility is not established.",
                        "Wilson intervals describe Monte Carlo rate uncertainty conditional on the chosen threshold, not clinical confidence intervals or threshold-calibration uncertainty."],
                source_hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in sources},script_sha256=sha(Path(__file__)),
                versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__})
    args.outdir.mkdir(parents=True,exist_ok=False)
    # Written before any calibration or evaluation outcomes are generated.
    (args.outdir/"design.json").write_text(json.dumps(design,indent=2),encoding="utf-8")

    def estimate(alive,dead):
        kh,th,diag,Hobs,H0=E.cohort_pipeline(dead,alive,t,tr,quiet,int(cfg["k_anchor_weeks"]),gamma)
        # Use the unmodified fitter's parameters for its actual hazard prediction.
        model_theta=float(diag.get("theta0_raw",th))
        Hgom=np.zeros_like(t)
        post=tr>=0
        Hgom[post]=np.r_[0.,np.cumsum(kh*np.exp(gamma*tr[post]))[:-1]]
        assert not diag.get("delta_applied_values")
        prediction=kh*np.exp(gamma*tr)/(1+model_theta*Hgom)
        p=-np.expm1(-prediction[quiet])
        var=alive[quiet]*p*(1-p)
        z=(dead[quiet]-alive[quiet]*p)/np.sqrt(np.maximum(var,1e-12))
        u=tr[quiet]-tr[quiet].mean()
        scatter=float(np.mean(z*z))
        level=float(np.sum(z)**2/len(z))
        trend=float(np.dot(u,z)**2/np.dot(u,u))
        eligible=bool(diag.get("success",False)) and all(not bool(diag.get(flag,False)) for flag in ["identifiability_flag","insufficient_signal","insufficient_deaths","bad_fit"])
        return dict(H0=H0,theta=th,k=kh,status=diag.get("status"),eligible=eligible,
                    score=max(scatter,level,trend),scatter=scatter,level=level,trend=trend,
                    relative_rmse=diag.get("relRMSE_hazard_fit"))

    def counts(theta,start,phase,rep,cohort):
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,phase,rep,cohort]))
        fn=None if start is None else lambda week:amount if start<=week<start+53 else 0.
        return E.simulate_one_cohort(rng,n0,nweeks,k,gamma,theta,skip,fn)

    # Reproduce deterministic historical endpoints before random trials.
    old=pd.read_csv(args.step5/"baseline_results.csv")
    Bpre=np.r_[0.,np.cumsum(b)[:-1]]
    def expected(theta,start):
        extra=np.zeros_like(t) if start is None else np.where((tr>=start)&(tr<start+53),amount,0.)
        h=b/(1+theta*Bpre)+extra
        alive=n0*np.exp(-np.r_[0.,np.cumsum(h)[:-1]])
        return alive,alive*-np.expm1(-h)
    control=estimate(*expected(d["theta_B"],None))
    checks=[]
    for name,start in cases.items():
        a=estimate(*expected(d["theta_A"],start))
        ref=old[(old.profile==("none" if start is None else "constant"))&(old.placement==name)].iloc[0]
        raw=a["H0"][-1]/control["H0"][-1]
        np.testing.assert_allclose(raw,ref.kcor_raw_final,rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(targets[name],ref.reference_raw_final,rtol=1e-12)
        checks.append(dict(case=name,matched=True,raw_final=raw,target=targets[name]))
    pd.DataFrame(checks).to_csv(args.outdir/"step5_replay_verification.csv",index=False)

    calibration=[]
    for rep in range(args.calibration_reps):
        a=estimate(*counts(d["theta_A"],None,0,rep,0))
        c=estimate(*counts(d["theta_B"],None,0,rep,1))
        eligible=a["eligible"] and c["eligible"]
        calibration.append(dict(rep=rep,eligible=eligible,score=max(a["score"],c["score"]),theta_A=a["theta"],theta_B=c["theta"]))
        if (rep+1)%50==0:
            print(f"Calibration {rep+1}/{args.calibration_reps}",flush=True)
    cal=pd.DataFrame(calibration)
    # Ineligible fits are rejected independently; include +infinity in the
    # calibration ordering so the nominal 5% budget includes those failures.
    scores=np.where(cal.eligible,cal.score,np.inf)
    threshold=float(np.quantile(scores,.95,method="higher"))
    if not np.isfinite(threshold):
        raise RuntimeError("At least 5% calibration failures: residual threshold cannot supply the planned overall gate")
    cal.to_csv(args.outdir/"calibration.csv",index=False)
    (args.outdir/"frozen_threshold.json").write_text(json.dumps(dict(score_cutoff=threshold,calibration_passes=int(np.sum(cal.eligible&(cal.score<=threshold))),calibration_total=len(cal)),indent=2),encoding="utf-8")
    print(f"Threshold frozen at {threshold:.6g}; starting independent evaluation",flush=True)

    rows=[]
    for rep in range(args.evaluation_reps):
        c=estimate(*counts(d["theta_B"],None,1,rep,1))
        for name,start in cases.items():
            a=estimate(*counts(d["theta_A"],start,1,rep,0))
            eligible=a["eligible"] and c["eligible"]
            score=max(a["score"],c["score"])
            accepted=eligible and score<=threshold
            raw=float(a["H0"][-1]/c["H0"][-1])
            anchor=skip+4
            anchored=raw/(a["H0"][anchor]/c["H0"][anchor])
            error=raw/targets[name]-1
            assert np.isfinite(raw) and raw>0
            rows.append(dict(case=name,rep=rep,target=targets[name],raw_final=raw,anchored_final=anchored,
                             relative_error_percent=100*error,absolute_error_above_10pct=abs(error)>.10,
                             eligible=eligible,score=score,accepted=accepted,
                             theta_A=a["theta"],theta_B=c["theta"],k_A=a["k"],k_B=c["k"],
                             status_A=a["status"],status_B=c["status"],relative_rmse_A=a["relative_rmse"],
                             residual_scatter_A=a["scatter"],residual_level_A=a["level"],residual_trend_A=a["trend"],
                             residual_scatter_B=c["scatter"],residual_level_B=c["level"],residual_trend_B=c["trend"]))
        if (rep+1)%50==0:
            print(f"Evaluation {rep+1}/{args.evaluation_reps} paired repetitions",flush=True)
    runs=pd.DataFrame(rows)
    runs.to_csv(args.outdir/"evaluation_runs.csv",index=False)
    summaries=[]
    for name,g in runs.groupby("case",sort=False):
        passed=g[g.accepted]
        n=len(g); na=len(passed)
        nb=int(passed.absolute_error_above_10pct.sum())
        summaries.append(dict(case=name,n=n,accepted=na,acceptance_percent=100*na/n,
                              acceptance_wilson95_low=wilson(na,n)[0],acceptance_wilson95_high=wilson(na,n)[1],
                              target=targets[name],raw_median=g.raw_final.median(),raw_p05=g.raw_final.quantile(.05),raw_p95=g.raw_final.quantile(.95),
                              accepted_raw_median=passed.raw_final.median(),accepted_raw_p05=passed.raw_final.quantile(.05),accepted_raw_p95=passed.raw_final.quantile(.95),
                              accepted_median_relative_error_percent=passed.relative_error_percent.median(),
                              accepted_large_error_count=nb,accepted_large_error_percent=100*nb/na if na else np.nan,
                              accepted_large_error_wilson95_low=wilson(nb,na)[0],accepted_large_error_wilson95_high=wilson(nb,na)[1],
                              joint_accept_and_large_error_percent=100*nb/n,
                              median_theta_A=g.theta_A.median(),near_zero_theta_A_percent=100*np.mean(g.theta_A<1e-8)))
    summary=pd.DataFrame(summaries)
    summary.to_csv(args.outdir/"summary.csv",index=False)
    # Paired comparisons retained separately: the comparator is identical within
    # each repetition, so timing differences are not comparator sampling noise.
    wide=runs.pivot(index="rep",columns="case",values="raw_final")
    wide["affected_minus_clean"]=wide.baseline_affected-wide.baseline_clean
    wide["clean_minus_null"]=wide.baseline_clean-wide.no_effect
    wide["affected_minus_null"]=wide.baseline_affected-wide.no_effect
    wide.to_csv(args.outdir/"paired_contrasts.csv")
    colors={"no_effect":"#777777","baseline_clean":"#157f60","baseline_affected":"#9348a6"}
    labels={"no_effect":"No added effect","baseline_clean":"Effect after baseline weeks","baseline_affected":"Effect in baseline weeks"}
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for name,g in runs.groupby("case",sort=False):
        ordered=np.sort(g.score)
        axes[0].step(ordered,np.arange(1,len(g)+1)/len(g),where="post",color=colors[name],label=labels[name])
        passed=g[g.accepted]
        axes[1].hist(passed.raw_final/targets[name],bins=25,histtype="step",density=True,color=colors[name],lw=1.8,label=labels[name])
    axes[0].axvline(threshold,color="black",ls="--",lw=1,label="Frozen residual cutoff")
    axes[0].set_xlabel("Residual-check score (higher is worse)")
    axes[0].set_ylabel("Fraction of evaluation runs")
    axes[0].set_title("Independent evaluation of the residual check")
    axes[0].legend(fontsize=7)
    axes[1].axvline(1,color="black",ls="--",lw=1)
    axes[1].set_xlabel("Estimated raw contrast / reference")
    axes[1].set_ylabel("Density among accepted runs")
    axes[1].set_title("Accuracy among runs accepted by this check")
    for ax in axes: ax.grid(axis="y",alpha=.2)
    fig.suptitle("Exploratory residual gate only; not the full manuscript diagnostic protocol",fontsize=11)
    fig.savefig(args.outdir/"random_detection.png",dpi=160)
    plt.close(fig)
    for name,checksum in design["source_hashes"].items():
        assert sha(ROOT/name)==checksum
    print(summary[["case","accepted","n","acceptance_percent","target","raw_median","accepted_raw_median","accepted_large_error_count","accepted_large_error_percent"]].to_string(index=False))
    print("All source hashes unchanged; deterministic replays and random runs completed",flush=True)


if __name__=="__main__": main()

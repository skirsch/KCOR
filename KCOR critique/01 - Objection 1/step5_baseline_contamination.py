"""Shift the existing 53-week profiles into early baseline estimation.

Compare post-skip effect weeks 0-52 with saved Step 3 weeks 5-57, retaining the
same 58-110 quiet window, 200-week horizon, and profile vector. The quiet-window
data and final integrated mortality are identical; the early baseline differs.
Original code and earlier artifacts remain read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step1_contamination_audit import ROOT, SOURCE, load_experiment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step3", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    meta=json.loads((args.step3/"manifest.json").read_text())
    for path,checksum in meta["source_hashes"].items():
        assert sha(ROOT/path)==checksum, f"Step 3 source changed: {path}"
    E=load_experiment()
    K=E.K
    design=meta["design"]
    old=pd.read_csv(args.step3/"timing_results.csv")
    old_series=pd.read_csv(args.step3/"timing_trajectories.csv")
    cfg=K.get_gompertz_theta_fit_config()
    k,gamma=design["mortality_scale"],design["gamma_per_week"]
    skip=design["skip_weeks"]
    t=np.arange(design["n_weeks"],dtype=float)
    tr=t-skip
    post=tr>=0
    anchor=skip+4
    b=k*np.exp(gamma*np.maximum(0,tr))
    Bpre=np.r_[0.,np.cumsum(b)[:-1]]
    B=np.cumsum(np.where(post,b,0.))
    baseline_mask=(tr>=0)&(tr<int(cfg["k_anchor_weeks"]))
    quiet=(tr>=design["quiet_start_rebased"])&(tr<=design["quiet_end_rebased"])

    def run(theta,g,start_shift=0,end_shift=0):
        h=b/(1+theta*Bpre)+g
        alive=design["n_initial"]*np.exp(-np.r_[0.,np.cumsum(h)[:-1]])
        dead=alive*-np.expm1(-h)
        q=(tr>=design["quiet_start_rebased"]+start_shift)&(tr<=design["quiet_end_rebased"]+end_shift)
        kh,th,diag,Hobs,H0=E.cohort_pipeline(dead,alive,t,tr,q,int(cfg["k_anchor_weeks"]),gamma)
        np.testing.assert_allclose(Hobs,np.cumsum(np.where(post,h,0.)),rtol=1e-10,atol=1e-14)
        Hgom=np.zeros_like(t)
        Hgom[post]=np.r_[0.,np.cumsum(kh*np.exp(gamma*tr[post]))[:-1]]
        assert not diag.get("delta_applied_values")
        pred=kh*np.exp(gamma*tr)/(1+th*Hgom)
        mask=np.asarray(diag["fit_mask_theta"],dtype=bool)
        residual=np.where(post,h,0.)-pred
        err=np.sqrt(np.mean((residual[mask]/(np.where(post,h,0.)[mask]+1e-12))**2))
        np.testing.assert_allclose(err,diag["relRMSE_hazard_fit"],rtol=1e-9,atol=1e-12)
        return dict(k=kh,theta=th,diag=diag,Hobs=Hobs,H0=H0,alive=alive,dead=dead,h=h,pred=pred,residual=residual,quiet=q)

    zero=np.zeros_like(t)
    control=run(design["theta_B"],zero)
    null=run(design["theta_A"],zero)
    rows,trajectories,window_runs,replays=[],[],[],[]
    early_results={}
    for profile in ["none","fading","constant","combined"]:
        if profile=="none":
            saved_block=old_series[(old_series.profile=="none")].sort_values("week")
            saved_row=old[old.profile=="none"].iloc[0]
        else:
            saved_block=old_series[(old_series.profile==profile)&(old_series.position=="before")].sort_values("week")
            saved_row=old[(old.profile==profile)&(old.position=="before")].iloc[0]
        clean_g=saved_block.added_hazard.to_numpy()
        clean=run(design["theta_A"],clean_g)
        with np.errstate(divide="ignore",invalid="ignore"):
            clean_raw=clean["H0"]/control["H0"]
        np.testing.assert_allclose([clean_raw[-1],clean["theta"],control["theta"]],
                                   [saved_row.kcor_raw_final,saved_row.theta_A,saved_row.theta_B],rtol=1e-8,atol=1e-10)
        replays.append(dict(profile=profile,raw_final_reproduced=float(clean_raw[-1]),matched=True))
        early_g=np.zeros_like(t)
        if profile!="none":
            vector=clean_g[(tr>=5)&(tr<=57)]
            assert len(vector)==53 and np.all(vector>0)
            early_g[(tr>=0)&(tr<=52)]=vector
            np.testing.assert_array_equal(early_g[post][:53],vector)
        early=run(design["theta_A"],early_g)
        np.testing.assert_allclose(early_g.sum(),clean_g.sum(),atol=1e-14,rtol=1e-12)
        assert np.all(early_g[quiet]==0) and np.all(clean_g[quiet]==0)
        # Same data in the fixed quiet window: only early information differs.
        for key in ["h","alive","dead"]:
            np.testing.assert_allclose(early[key][quiet],clean[key][quiet],rtol=1e-11,atol=1e-10)
        np.testing.assert_allclose(early["Hobs"][-1],clean["Hobs"][-1],rtol=1e-11,atol=1e-14)
        variants=[("no_effect",zero,null)] if profile=="none" else [("baseline_clean",clean_g,clean),("baseline_affected",early_g,early)]
        for label,g,a in variants:
            G=np.cumsum(np.where(post,g,0.))
            with np.errstate(divide="ignore",invalid="ignore"):
                ref=(B+G)/B
                raw=a["H0"]/control["H0"]
            anc=E.anchored_kcor(a["H0"],control["H0"],skip)
            ref_anc=ref/ref[anchor]
            np.testing.assert_allclose(anc[post],raw[post]/raw[anchor],rtol=1e-12)
            np.testing.assert_allclose(-np.log(np.exp(-(B[post]+G[post])))/-np.log(np.exp(-B[post])),ref[post],rtol=1e-11)
            extra=design["n_initial"]*(np.exp(-null["h"].sum())-np.exp(-a["h"].sum()))
            rq=a["residual"][quiet]/a["h"][quiet]
            d=a["diag"]
            rows.append(dict(profile=profile,placement=label,
                             effect_start_rebased=0 if label=="baseline_affected" else (5 if label=="baseline_clean" else None),
                             effect_end_rebased=52 if label=="baseline_affected" else (57 if label=="baseline_clean" else None),
                             baseline_added_hazard=g[baseline_mask].sum(),added_cumulative_hazard=G[-1],
                             extra_expected_deaths=extra,k_A=a["k"],k_A_over_clean_k=a["k"]/null["k"],theta_A=a["theta"],theta_B=control["theta"],
                             reference_raw_final=ref[-1],kcor_raw_final=raw[-1],reference_at_anchor=ref[anchor],kcor_at_anchor=raw[anchor],
                             reference_anchored_final=ref_anc[-1],kcor_anchored_final=anc[-1],
                             raw_relative_error_percent=100*(raw[-1]/ref[-1]-1),
                             anchored_relative_error_percent=100*(anc[-1]/ref_anc[-1]-1),
                             pooled_relative_rmse_percent=100*d["relRMSE_hazard_fit"],
                             quiet_relative_rmse_percent=100*np.sqrt(np.mean(rq**2)),
                             quiet_first_relative_residual_percent=100*rq[0],quiet_last_relative_residual_percent=100*rq[-1],
                             fit_status_A=d.get("status"),fit_success_A=d.get("success"),identifiability_flag_A=d.get("identifiability_flag")))
            trajectories.append(pd.DataFrame(dict(profile=profile,placement=label,week=t,rebased_week=tr,
                                                  added_hazard=g,observed_hazard=a["h"],predicted_hazard=a["pred"],
                                                  residual=a["residual"],reference_raw=ref,kcor_raw=raw,
                                                  reference_anchored=ref_anc,kcor_anchored=anc)))
        early_results[profile]=early
        for ds in [-4,0,4]:
            for de in [-4,0,4]:
                a=run(design["theta_A"],early_g,ds,de)
                c=run(design["theta_B"],zero,ds,de)
                assert not np.any(early_g[a["quiet"]]),"Window shift introduced harm into quiet weeks"
                window_runs.append(dict(profile=profile,start_shift=ds,end_shift=de,k_A=a["k"],theta_A=a["theta"],theta_B=c["theta"],
                                        raw_final=a["H0"][-1]/c["H0"][-1],pooled_relative_rmse_percent=100*a["diag"]["relRMSE_hazard_fit"],
                                        fit_status_A=a["diag"].get("status")))

    result=pd.DataFrame(rows)
    trace=pd.concat(trajectories,ignore_index=True)
    for profile,block in result[result.profile!="none"].groupby("profile"):
        for col in ["reference_raw_final","added_cumulative_hazard","extra_expected_deaths"]:
            np.testing.assert_allclose(block[col],block.iloc[0][col],rtol=1e-11,atol=1e-10)
    windows=pd.DataFrame(window_runs)
    args.outdir.mkdir(parents=True,exist_ok=False)
    result.to_csv(args.outdir/"baseline_results.csv",index=False)
    trace.to_csv(args.outdir/"baseline_trajectories.csv",index=False)
    windows.to_csv(args.outdir/"window_perturbations.csv",index=False)
    windows.groupby("profile")[["theta_A","raw_final","pooled_relative_rmse_percent"]].agg(["min","max"]).to_csv(args.outdir/"window_perturbation_ranges.csv")
    pd.DataFrame(replays).to_csv(args.outdir/"step3_replay_verification.csv",index=False)

    fig,axes=plt.subplots(2,3,figsize=(12,6.7),constrained_layout=True)
    for col,profile in enumerate(["fading","constant","combined"]):
        block=trace[(trace.profile==profile)&(trace.placement=="baseline_affected")&(trace.rebased_week>=0)]
        clean=trace[(trace.profile==profile)&(trace.placement=="baseline_clean")&(trace.rebased_week>=0)]
        axes[0,col].plot(block.rebased_week,block.reference_raw,color="#157f60",lw=2,label="Reference, baseline affected")
        axes[0,col].plot(block.rebased_week,block.kcor_raw,color="#9348a6",lw=1.8,ls="--",label="KCOR, baseline affected")
        axes[0,col].plot(clean.rebased_week,clean.kcor_raw,color="#777777",lw=1.5,ls=":",label="KCOR, clean baseline")
        axes[0,col].axvspan(0,4,color="#efd3aa",alpha=.6)
        axes[0,col].axvspan(58,111,color="#dfe8ef",alpha=.7)
        axes[0,col].set_title(profile.capitalize()+" effect")
        axes[0,col].set_xlabel("Weeks after initial two-week skip")
        axes[0,col].set_ylim(.95,2.75)
        w=block[(block.rebased_week>=58)&(block.rebased_week<=110)]
        axes[1,col].plot(w.rebased_week,100*w.residual/w.observed_hazard,color="#c06b25",lw=2)
        axes[1,col].axhline(0,color="black",lw=.8)
        axes[1,col].set_ylim(-25,20)
        axes[1,col].set_xlabel("Weeks within unchanged quiet window")
        for ax in axes[:,col]:
            ax.grid(axis="y",alpha=.2)
    axes[0,0].set_ylabel("Unanchored cumulative hazard contrast")
    axes[1,0].set_ylabel("Residual / simulated hazard (%)")
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Moving the same effect into early baseline weeks amplifies the fitted correction",fontsize=12)
    fig.savefig(args.outdir/"baseline_contamination_comparison.png",dpi=160)
    plt.close(fig)
    inputs=[SOURCE,ROOT/"code/KCOR.py",ROOT/"data/Czech/Czech.yaml",Path(__file__).with_name("step1_contamination_audit.py"),
            args.step3/"manifest.json",args.step3/"timing_results.csv",args.step3/"timing_trajectories.csv"]
    outmeta=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in inputs},
                 script_sha256=sha(Path(__file__)),original_design=design,
                 new_effect_start_rebased=0,new_effect_end_rebased=52,
                 reference="(B+G)/B and identically anchored version; same discrete generator and accumulation origin",
                 baseline_fit_config=cfg,diagnostic_config=K.get_theta_degenerate_fit_config(),
                 validation="Reproduced four Step 3 cases; identical effect vectors/integrals, final hazards, quiet-window hazards/risk sets/deaths, and final targets verified; residual RMS and anchoring reconstructed; 36 +/-4-week boundary fits retain uncontaminated quiet windows.",
                 limitations=["Same three existing effect strengths only; no plausibility claim.",
                              "Expected counts only, not statistical diagnostic rejection rates.",
                              "Only early baseline weeks added to scope; effects still end before quiet window.",
                              "Anchor is affected in the early case, so raw and anchored references differ.",
                              "No complete manuscript-diagnostic pass is inferred from fitter status."])
    (args.outdir/"manifest.json").write_text(json.dumps(outmeta,indent=2),encoding="utf-8")
    print(result[["profile","placement","reference_raw_final","kcor_raw_final","reference_anchored_final","kcor_anchored_final","k_A_over_clean_k","theta_A","pooled_relative_rmse_percent","fit_status_A"]].to_string(index=False))
    print(windows.groupby("profile")[["theta_A","raw_final"]].agg(["min","max"]).to_string())
    print(f"Saved {len(result)} nominal cases and {len(windows)} boundary checks to {args.outdir}")


if __name__=="__main__":
    main()

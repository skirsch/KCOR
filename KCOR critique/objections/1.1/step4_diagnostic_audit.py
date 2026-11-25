"""Apply observable diagnostics to saved Step 3 schedules, without new effects.

Residual and linearity metrics are descriptive. No post-hoc threshold is used to
declare full manuscript acceptance. Window +/-4 is specified on PDF page 68;
skip 0/4 and gamma +/-5% are explicitly exploratory choices for other checks.
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

from step1_contamination_audit import ROOT, load_experiment


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step3", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.step3 / "manifest.json").read_text())
    for name, checksum in meta["source_hashes"].items():
        assert sha(ROOT/name) == checksum, f"Step 3 input changed: {name}"
    E = load_experiment()
    K = E.K
    design = meta["design"]
    k = design["mortality_scale"]
    gamma = design["gamma_per_week"]
    original_skip = design["skip_weeks"]
    saved = pd.read_csv(args.step3 / "timing_results.csv")
    series = pd.read_csv(args.step3 / "timing_trajectories.csv")
    quiet_start_abs = design["quiet_start_rebased"] + original_skip
    quiet_end_abs = design["quiet_end_rebased"] + original_skip
    fit_cfg = K.get_gompertz_theta_fit_config()

    def linear_r2(x, y):
        pred = np.polyval(np.polyfit(x, y, 1), x)
        ss = np.sum((y-y.mean())**2)
        return float(1-np.sum((y-pred)**2)/ss)

    def examine(block, start_shift=0, end_shift=0, skip=original_skip, gamma_scale=1.):
        block = block.sort_values("week")
        t = block.week.to_numpy()
        original_tr = block.rebased_week.to_numpy()
        tr = t - skip
        # Physical data stay fixed when fitting windows, skip or assumed gamma change.
        b = k*np.exp(gamma*np.maximum(0, original_tr))
        Bpre = np.r_[0., np.cumsum(b)[:-1]]
        g = block.added_hazard.to_numpy()
        quiet = (t >= quiet_start_abs+start_shift) & (t <= quiet_end_abs+end_shift)
        fit_gamma = gamma*gamma_scale
        fits = []
        for theta, extra in [(design["theta_A"],g),(design["theta_B"],np.zeros_like(g))]:
            h = b/(1+theta*Bpre)+extra
            alive = design["n_initial"]*np.exp(-np.r_[0.,np.cumsum(h)[:-1]])
            dead = alive * -np.expm1(-h)
            obs = K.hazard_from_mr(dead/alive)
            eff = np.where(t>=skip, obs, 0.)
            (kh,th),d = K.fit_theta0_gompertz(eff,tr,quiet,int(fit_cfg["k_anchor_weeks"]),fit_gamma,deaths_arr=dead)
            assert d["success"] and np.isfinite(th)
            H0 = K.invert_gamma_frailty(np.cumsum(eff),th)
            post = tr >= 0
            Hgom = np.zeros_like(tr)
            Hgom[post] = np.r_[0., np.cumsum(kh*np.exp(fit_gamma*tr[post]))[:-1]]
            # Each fit has one contiguous quiet window, hence no delta offsets.
            assert len(d.get("delta_applied_values",[])) == 0
            pred = kh*np.exp(fit_gamma*tr)/(1+th*Hgom)
            mask = np.asarray(d["fit_mask_theta"],dtype=bool)
            residual = eff-pred
            recomputed = float(np.sqrt(np.mean((residual[mask]/(eff[mask]+1e-12))**2)))
            np.testing.assert_allclose(recomputed,d["relRMSE_hazard_fit"],rtol=1e-10,atol=1e-13)
            fits.append(dict(theta=th,k=kh,diag=d,H0=H0,observed=eff,predicted=pred,residual=residual,fit_mask=mask))
        a,c = fits
        with np.errstate(divide="ignore",invalid="ignore"):
            raw = a["H0"]/c["H0"]
        anc = raw/raw[skip+4]
        r=a["residual"][quiet]
        y=a["observed"][quiet]
        # A sign tolerance only avoids machine-rounding signs. It is NOT a
        # diagnostic rejection threshold or a significance test.
        rtol=1e-12
        row=dict(start_shift=start_shift,end_shift=end_shift,skip_weeks=skip,gamma_scale=gamma_scale,
                 theta_A=a["theta"],theta_B=c["theta"],kcor_raw_final=raw[-1],kcor_anchored_final=anc[-1],
                 status_A=a["diag"]["status"],success_A=bool(a["diag"]["success"]),
                 identifiability_flag_A=bool(a["diag"].get("identifiability_flag",False)),
                 pooled_relative_rmse_percent=100*a["diag"]["relRMSE_hazard_fit"],
                 quiet_relative_rmse_percent=100*np.sqrt(np.mean((r/y)**2)),
                 quiet_positive_residual_weeks=int(np.sum(r>rtol)),quiet_negative_residual_weeks=int(np.sum(r < -rtol)),
                 quiet_weeks=int(quiet.sum()), quiet_mean_relative_residual_percent=100*np.mean(r/y),
                 quiet_first_relative_residual_percent=100*r[0]/y[0],
                 quiet_last_relative_residual_percent=100*r[-1]/y[-1],
                 quiet_residual_lag1_correlation=float(np.corrcoef(r[:-1],r[1:])[0,1]) if np.std(r)>rtol else np.nan,
                 ordinary_time_linearity_r2=linear_r2(tr[quiet],a["H0"][quiet]),
                 gompertz_time_linearity_r2=linear_r2(np.expm1(fit_gamma*tr[quiet])/fit_gamma,a["H0"][quiet]),
                 quiet_expected_deaths=a["diag"]["total_quiet_deaths"])
        diagnostic_series=pd.DataFrame(dict(week=t,rebased_week=original_tr,in_quiet=quiet,
                                            in_fit=a["fit_mask"],observed_hazard=a["observed"],
                                            fitted_hazard=a["predicted"],residual=a["residual"]))
        return row,diagnostic_series

    nominal, residuals, perturbations = [],[],[]
    for (profile,position),block in series.groupby(["profile","position"],sort=False):
        row,rs=examine(block)
        expected=saved[(saved.profile==profile)&(saved.position==position)].iloc[0]
        np.testing.assert_allclose([row["theta_A"],row["theta_B"],row["kcor_raw_final"],row["kcor_anchored_final"]],
                                   [expected.theta_A,expected.theta_B,expected.kcor_raw_final,expected.kcor_anchored_final],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(row["pooled_relative_rmse_percent"],100*expected.relative_fit_rmse_A,rtol=1e-8,atol=1e-10)
        nominal.append(dict(profile=profile,position=position,**row))
        rs["profile"],rs["position"]=profile,position
        residuals.append(rs)
        # Target the distorted cases and a no-effect control; leave effect
        # schedules unchanged. No made-up boundary sensitivity tolerance.
        if position=="during" or profile=="none":
            variants=[("window_bounds",a,b,original_skip,1.) for a in [-4,0,4] for b in [-4,0,4]]
            variants += [("skip_exploratory",0,0,s,1.) for s in [0,4]]
            variants += [("gamma_exploratory",0,0,original_skip,q) for q in [.95,1.05]]
            for family,ds,de,skip,gs in variants:
                rr,_=examine(block,ds,de,skip,gs)
                perturbations.append(dict(profile=profile,position=position,check=family,**rr))
    nominal=pd.DataFrame(nominal)
    residuals=pd.concat(residuals,ignore_index=True)
    perturbations=pd.DataFrame(perturbations)
    args.outdir.mkdir(parents=True,exist_ok=True)
    assert not (args.outdir/"diagnostic_metrics.csv").exists(),"Do not overwrite earlier audit"
    nominal.to_csv(args.outdir/"diagnostic_metrics.csv",index=False)
    residuals.to_csv(args.outdir/"hazard_residuals.csv",index=False)
    perturbations.to_csv(args.outdir/"diagnostic_perturbations.csv",index=False)
    perturbations.groupby(["profile","check"])[["theta_A","kcor_raw_final","pooled_relative_rmse_percent"]].agg(["min","max"]).to_csv(args.outdir/"perturbation_ranges.csv")

    fig,axes=plt.subplots(2,3,figsize=(12,6.4),constrained_layout=True)
    for col,profile in enumerate(["fading","constant","combined"]):
        block=residuals[(residuals.profile==profile)&(residuals.position=="during")&residuals.in_quiet]
        t=block.rebased_week
        axes[0,col].plot(t,1000*block.observed_hazard,lw=2,color="#157f60",label="Simulated hazard")
        axes[0,col].plot(t,1000*block.fitted_hazard,lw=1.8,ls="--",color="#9348a6",label="Fitted hazard")
        axes[0,col].set_title(profile.capitalize()+" effect during estimation")
        axes[1,col].plot(t,100*block.residual/block.observed_hazard,lw=2,color="#c06b25")
        axes[1,col].axhline(0,color="black",lw=.8)
        axes[1,col].set_ylim(-2,60)
        axes[1,col].set_xlabel("Weeks after initial two-week skip")
        for ax in axes[:,col]:
            ax.grid(axis="y",alpha=.2)
    axes[0,0].set_ylabel("Weekly hazard x 1,000")
    axes[1,0].set_ylabel("Residual / simulated hazard (%)")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("All three overlapping cases have persistent positive fitting residuals",fontsize=12)
    fig.savefig(args.outdir/"overlap_residuals.png",dpi=160)
    plt.close(fig)

    inputs=[ROOT/"KCOR critique/KCOR.pdf", ROOT/"code/KCOR.py", ROOT/"data/Czech/Czech.yaml",
            ROOT/"test/quiet_window/code/quiet_window_scan_theta_czech_2021_24.py",
            args.step3/"manifest.json",args.step3/"timing_results.csv",args.step3/"timing_trajectories.csv"]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in inputs},
                  script_sha256=sha(Path(__file__)),pdf_pages_examined=[7,8,12,14,38,56,58,59,67,68,75],
                  nominal_cases_reproduced=len(nominal),perturbation_fits=len(perturbations),
                  production_diagnostic_config=K.get_theta_degenerate_fit_config(),
                  limits=["No formal full-pass label assigned: manuscript qualitative criteria lack complete numeric thresholds.",
                          "Residual metrics describe expected-count schedules; no random-noise rejection rates or significance tests.",
                          "Near-zero estimate is a warning under manuscript wording, not by itself proof of statistical non-identification.",
                          "Only one quiet window: omit-one-window and inter-window offset checks cannot be evaluated.",
                          "Optional epidemic module inactive: its checks do not apply.",
                          "Skip 0/4 and gamma +/-5% chosen for exploratory sensitivity, not claimed manuscript-prespecified magnitudes.",
                          "Separate empirical scan's R2/percentile thresholds use a different fitter and are not imported as universal rules."])
    (args.outdir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(nominal[["profile","position","pooled_relative_rmse_percent","quiet_positive_residual_weeks","quiet_weeks","quiet_first_relative_residual_percent","quiet_last_relative_residual_percent","ordinary_time_linearity_r2","gompertz_time_linearity_r2","theta_A","status_A"]].to_string(index=False))
    print(perturbations.groupby(["profile","check"])[["theta_A","kcor_raw_final"]].agg(["min","max"]).to_string())
    print(f"Verified {len(nominal)} nominal cases; ran {len(perturbations)} diagnostic perturbations")


if __name__=="__main__":
    main()

"""Controlled timing extension of the existing additive contamination scenarios.

Only expected counts, three original smallest nonzero profiles, and one null.
Reuses original KCOR fitting/inversion functions through the Step 1 loader.
Production files and previous audit outputs are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
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


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    old_meta = json.loads((args.step1 / "manifest.json").read_text())
    for name, checksum in old_meta["sources"].items():
        assert sha(ROOT / name) == checksum, f"Input changed since Step 1: {name}"
    E = load_experiment()
    K = E.K
    k, gamma = old_meta["mortality_scale"], old_meta["gamma_per_week"]
    skip = int(K.DYNAMIC_HVE_SKIP_WEEKS)
    fit_cfg = K.get_gompertz_theta_fit_config()
    assert np.isclose(fit_cfg["gamma_per_week"], gamma, atol=1e-16, rtol=0)
    profiles = {"fading": (float(E.EPSILON_DYNAMIC_POS.min()), 0.0),
                "constant": (0.0, float(E.EPSILON_COMORB_POS.min())),
                "combined": (float(E.EPSILON_DYNAMIC_POS.min()), float(E.EPSILON_COMORB_POS.min()))}

    def evaluate(n_weeks, quiet_start, quiet_end, start, ed, ec):
        t, tr, _ = E.build_time_arrays(n_weeks, skip)
        quiet = (tr >= quiet_start) & (tr <= quiet_end)
        post = tr >= 0
        b = k * np.exp(gamma * np.maximum(0, tr))
        Bpre = np.r_[0.0, np.cumsum(b)[:-1]]
        duration = E.QUIET_T_REBASED_MAX - E.QUIET_T_REBASED_MIN + 1
        offset = tr - start
        active = (offset >= 0) & (offset < duration)
        # Shift the existing vector: at onset it equals ed*exp(-lambda*4)+ec.
        g = np.where(active, ed * np.exp(-E.LAMBDA_DECAY * (offset + E.QUIET_T_REBASED_MIN)) + ec, 0.0)
        B = np.cumsum(np.where(post, b, 0))
        G = np.cumsum(np.where(post, g, 0))
        results = []
        for theta, added in [(E.THETA0_A, g), (E.THETA0_B, np.zeros_like(g))]:
            h = b / (1 + theta * Bpre) + added
            alive = E.N_INITIAL * np.exp(-np.r_[0.0, np.cumsum(h)[:-1]])
            dead = alive * -np.expm1(-h)
            np.testing.assert_allclose(dead.sum(), E.N_INITIAL * -np.expm1(-h.sum()), rtol=1e-12)
            kh, th, diag, Hobs, H0 = E.cohort_pipeline(dead, alive, t, tr, quiet,
                                                     int(fit_cfg["k_anchor_weeks"]), gamma)
            np.testing.assert_allclose(Hobs, np.cumsum(np.where(post, h, 0)), rtol=1e-10, atol=1e-14)
            results.append(dict(k=kh, theta=th, diag=diag, Hobs=Hobs, H0=H0, h=h))
        a, control = results
        anchor = skip + 4
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = a["H0"] / control["H0"]
            reference = (B + G) / B
        anchored = E.anchored_kcor(a["H0"], control["H0"], skip)
        ref_anchored = reference / reference[anchor]
        np.testing.assert_allclose(anchored[post], raw[post] / raw[anchor], rtol=1e-12)
        np.testing.assert_allclose(-np.log(np.exp(-(B[post]+G[post]))) / -np.log(np.exp(-B[post])), reference[post], rtol=1e-11)
        return dict(t=t, tr=tr, quiet=quiet, post=post, b=b, g=g, B=B, G=G,
                    a=a, control=control, raw=raw, anchored=anchored,
                    reference=reference, ref_anchored=ref_anchored, anchor=anchor)

    # Verify this wrapper reproduces saved expected-count cases before changing
    # any timings. The duplicate null settings are equivalent.
    old = pd.read_csv(args.step1 / "expected_count_results.csv")
    replay = []
    for mode, ed, ec in [("dynamic", 0., 0.), ("dynamic", *profiles["fading"]),
                         ("comorbidity", *profiles["constant"]), ("combined", *profiles["combined"])]:
        x = evaluate(E.N_WEEKS, E.QUIET_T_REBASED_MIN, E.QUIET_T_REBASED_MAX,
                     E.QUIET_T_REBASED_MIN, ed, ec)
        row = old[(old["mode"] == mode) & (old.epsilon_dyn == ed) & (old.epsilon_constant == ec)].iloc[0]
        actual = [x["raw"][-1], x["anchored"][-1], x["a"]["theta"], x["control"]["theta"]]
        expected = [row.raw_final, row.anchored_final, row.theta_A, row.theta_B]
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
        replay.append(dict(mode=mode, epsilon_dyn=ed, epsilon_constant=ec,
                           max_abs_difference=float(np.max(np.abs(np.array(actual)-expected))), matched=True))

    # All positions use identical baseline, fitted window, follow-up, anchor,
    # amplitude vector, and duration. Early baseline/anchor remain clean.
    # Relative week 0 is the first week after the initial two-week skip.
    n_weeks, q_start, q_end = 200, 58, 110
    positions = {"before": 5, "during": 58, "after": 111}
    null = evaluate(n_weeks, q_start, q_end, positions["before"], 0., 0.)
    assert null["reference"][null["post"]].min() == 1
    cases = [("none", "none", positions["before"], 0., 0.)]
    cases += [(name, position, start, ed, ec) for name, (ed, ec) in profiles.items()
              for position, start in positions.items()]
    rows, trajectories, diags = [], [], []
    vectors = {}
    for profile, position, start, ed, ec in cases:
        x = null if profile == "none" else evaluate(n_weeks, q_start, q_end, start, ed, ec)
        a, control = x["a"], x["control"]
        g, anchor, post = x["g"], x["anchor"], x["post"]
        assert np.all(g[:anchor+1] == 0), "Baseline or anchor contaminated"
        assert np.all(g >= 0)
        np.testing.assert_allclose(x["reference"][anchor], 1, atol=1e-14)
        np.testing.assert_allclose(x["ref_anchored"][post], x["reference"][post], rtol=1e-12)
        np.testing.assert_allclose(a["k"], null["a"]["k"], rtol=1e-12)
        np.testing.assert_allclose(control["H0"], null["control"]["H0"], rtol=1e-12)
        if profile != "none":
            active = (x["tr"] >= start) & (x["tr"] <= start+52)
            assert active.sum() == 53
            assert np.count_nonzero(g) == 53
            vectors.setdefault(profile, g[active])
            np.testing.assert_array_equal(g[active], vectors[profile])
            overlap = int(np.count_nonzero(active & x["quiet"]))
            assert overlap == (53 if position == "during" else 0)
        else:
            overlap = 0
        extra_deaths = E.N_INITIAL * (np.exp(-null["a"]["h"].sum()) - np.exp(-a["h"].sum()))
        np.testing.assert_allclose(np.exp(-a["h"].sum()) / np.exp(-null["a"]["h"].sum()), np.exp(-g.sum()), rtol=1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            fixed_raw = K.invert_gamma_frailty(a["Hobs"], null["a"]["theta"]) / control["H0"]
        raw, anc, ref, ref_anc = x["raw"], x["anchored"], x["reference"], x["ref_anchored"]
        effect = ref[-1] - 1
        d = a["diag"]
        rows.append(dict(profile=profile, position=position,
                         start_rebased=start if profile != "none" else None,
                         end_rebased=start+52 if profile != "none" else None,
                         epsilon_dyn=ed, epsilon_constant=ec, overlapping_fit_weeks=overlap,
                         added_cumulative_hazard=g.sum(), peak_added_hazard_over_href=g.max()/k,
                         extra_expected_deaths=extra_deaths,
                         target_raw_final=ref[-1], kcor_raw_final=raw[-1],
                         target_anchored_final=ref_anc[-1], kcor_anchored_final=anc[-1],
                         raw_relative_error_percent=100*(raw[-1]/ref[-1]-1),
                         anchored_relative_error_percent=100*(anc[-1]/ref_anc[-1]-1),
                         raw_excess_recovery_percent=100*(raw[-1]-1)/effect if effect > 0 else np.nan,
                         paired_added_raw_contrast_recovery_percent=100*(raw[-1]-null["raw"][-1])/effect if effect > 0 else np.nan,
                         max_abs_raw_relative_error_percent=100*np.max(np.abs(raw[post]/ref[post]-1)),
                         theta_A=a["theta"], theta_B=control["theta"],
                         fixed_clean_fit_raw_final=fixed_raw[-1],
                         refitting_raw_shift=raw[-1]-fixed_raw[-1],
                         fit_status_A=d.get("status"), fit_success_A=d.get("success"),
                         relative_fit_rmse_A=d.get("relRMSE_hazard_fit"),
                         identifiability_flag_A=d.get("identifiability_flag"),
                         insufficient_signal_A=d.get("insufficient_signal"), bad_fit_A=d.get("bad_fit")))
        diags.append(dict(profile=profile, position=position, A=d, B=control["diag"]))
        trajectories.append(pd.DataFrame(dict(profile=profile, position=position,
                                              week=x["t"], rebased_week=x["tr"], added_hazard=g,
                                              reference=ref, kcor_raw=raw, reference_anchored=ref_anc,
                                              kcor_anchored=anc, fixed_clean_fit_raw=fixed_raw)))
    result = pd.DataFrame(rows)
    for profile, block in result[result.profile != "none"].groupby("profile"):
        # Equal integrated harm and common endpoint ensure a common true final
        # contrast and total expected excess deaths across positions.
        for col in ["added_cumulative_hazard", "target_raw_final", "extra_expected_deaths"]:
            np.testing.assert_allclose(block[col], block.iloc[0][col], rtol=1e-11, atol=1e-10)
        for col in ["target_anchored_final", "kcor_raw_final", "kcor_anchored_final"]:
            assert np.all(np.isfinite(block[col]))
    trajectories = pd.concat(trajectories, ignore_index=True)

    args.outdir.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.outdir / "timing_results.csv", index=False)
    trajectories.to_csv(args.outdir / "timing_trajectories.csv", index=False)
    pd.DataFrame(replay).to_csv(args.outdir / "step1_replay_verification.csv", index=False)
    (args.outdir / "fit_diagnostics.json").write_text(json.dumps(clean_json(diags), indent=2, allow_nan=False), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), constrained_layout=True)
    for i, position in enumerate(positions):
        block = trajectories[(trajectories.profile == "fading") & (trajectories.position == position)
                             & (trajectories.rebased_week >= 0)]
        for ax in axes[:, i]:
            ax.axvspan(q_start, q_end+1, color="#dfe8ef", alpha=0.8, label="Estimation window")
            ax.grid(axis="y", alpha=0.2)
        axes[0, i].plot(block.rebased_week, 100*block.added_hazard/k, color="#c06b25", lw=2)
        axes[0, i].set_title(f"Effect {position} estimation")
        axes[0, i].set_ylim(-2, 100)
        axes[1, i].axhline(1, color="#888888", lw=0.8)
        axes[1, i].plot(block.rebased_week, block.reference, color="#157f60", lw=2.2, label="Defined reference")
        axes[1, i].plot(block.rebased_week, block.kcor_raw, color="#9348a6", ls="--", lw=1.8, label="KCOR before anchoring")
        axes[1, i].set_ylim(0.98, 1.56)
        axes[1, i].set_xlabel("Weeks after initial two-week skip")
    axes[0, 0].set_ylabel("Added hazard / initial baseline (%)")
    axes[1, 0].set_ylabel("Cumulative hazard contrast")
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Same fading effect, three positions; baseline and anchor remain uncontaminated", fontsize=12)
    fig.savefig(args.outdir / "fading_timing_comparison.png", dpi=160)
    plt.close(fig)

    inputs = [SOURCE, ROOT/"code/KCOR.py", ROOT/"data/Czech/Czech.yaml",
              Path(__file__).with_name("step1_contamination_audit.py"),
              args.step1/"manifest.json", args.step1/"expected_count_results.csv"]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__,
                    design=dict(n_initial=E.N_INITIAL, n_weeks=n_weeks, skip_weeks=skip,
                                quiet_start_rebased=q_start, quiet_end_rebased=q_end,
                                anchor_rebased=4, effect_starts_rebased=positions, duration_weeks=53,
                                theta_A=E.THETA0_A, theta_B=E.THETA0_B, profiles=profiles,
                                mortality_scale=k, gamma_per_week=gamma),
                    baseline_fit_config=fit_cfg, diagnostic_config=K.get_theta_degenerate_fit_config(),
                    target="Remove baseline frailty depletion while preserving additive hazard: (B+G)/B; same skip and anchor.",
                    source_hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in inputs},
                    script_sha256=sha(Path(__file__)),
                    checks="Four saved Step 1 cases replayed; counts/hazards, positive additive effect, clean baseline/anchor, equal shifted effect vectors, equal final targets/excess deaths, and anchoring identities verified.",
                    limitations=["Expected counts only; no sampling/coverage claim.",
                                 "Original discrete generator and unequal frailty levels retained.",
                                 "New common 200-week follow-up and middle estimation window allow three non-overlapping placements.",
                                 "No effects in early baseline-estimation weeks or anchor are tested.",
                                 "No individual frailty-dependent treatment or epidemic mechanism is added.",
                                 "Fitter status is not a demonstration of all manuscript diagnostics passing.",
                                 "Effect sizes inherited from existing grid; no empirical plausibility claim."])
    (args.outdir / "manifest.json").write_text(json.dumps(clean_json(manifest), indent=2, allow_nan=False), encoding="utf-8")
    for p in inputs:
        assert sha(p) == manifest["source_hashes"][str(p.resolve().relative_to(ROOT))]
    print(result[["profile","position","target_raw_final","kcor_raw_final","kcor_anchored_final",
                  "paired_added_raw_contrast_recovery_percent","theta_A","relative_fit_rmse_A","fit_status_A"]].to_string(index=False))
    print(f"Saved {len(result)} cases to {args.outdir}")


if __name__ == "__main__":
    main()

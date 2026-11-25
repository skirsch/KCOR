"""Define a selection-removed, added-hazard-retained reference for Step 1.

For the original discrete generator h_j=b_j/(1+theta*Bpre_j)+g_j,
the chosen intervention removes the baseline depletion denominator, leaving
b_j+g_j. The reference contrast is (B_j+G_j)/B_j, accumulated over the same
post-skip weeks as KCOR. This is a mechanistic benchmark, not a mortality-risk
ratio or a claim that every off-model KCOR output must have this interpretation.

Reads Step 1 results; does not alter production code or regenerate scenarios.
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

ROOT = Path(__file__).resolve().parents[3]
KEYS = ["mode", "epsilon_dyn", "epsilon_constant"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.step1 / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in meta["sources"].items():
        assert digest(ROOT / name) == sha, f"Step 1 source changed: {name}"
    trajectories = pd.read_csv(args.step1 / "expected_count_trajectories.csv")
    summary = pd.read_csv(args.step1 / "expected_count_results.csv")
    k, gamma = meta["mortality_scale"], meta["gamma_per_week"]
    anchor = int(meta["anchor_original_week"])
    null = trajectories[(trajectories["mode"] == "dynamic")
                        & (trajectories.epsilon_dyn == 0)
                        & (trajectories.epsilon_constant == 0)].sort_values("week")
    null_raw, null_anchored = null.raw.to_numpy(), null.anchored.to_numpy()
    rows, series = [], []

    for values, block in trajectories.groupby(KEYS, sort=False):
        mode, ed, ec = values
        block = block.sort_values("week").copy()
        t = block.week.to_numpy()
        tr = block.rebased_week.to_numpy()
        post = tr >= 0
        b = k * np.exp(gamma * np.maximum(0.0, tr))
        g = block.added_hazard.to_numpy()
        B = np.cumsum(np.where(post, b, 0))
        G = np.cumsum(np.where(post, g, 0))
        # These are the specified selection-free cumulative hazards, not a
        # gamma inversion applied to the whole contaminated cumulative hazard.
        with np.errstate(divide="ignore", invalid="ignore"):
            target = (B + G) / B
        at_anchor = np.flatnonzero(t == anchor).item()
        target_anchored = target / target[at_anchor]
        actual = block.raw.to_numpy()
        actual_anchored = block.anchored.to_numpy()
        fixed = block.fixed_clean_fit_raw.to_numpy()
        fixed_anchored = fixed / fixed[at_anchor]

        # Independent arithmetic check: the reference also equals the ratio of
        # cumulative -log survival in the two specified selection-free schedules.
        s_control = np.exp(-B)
        s_treated = np.exp(-(B + G))
        np.testing.assert_allclose(-np.log(s_treated[post]) / -np.log(s_control[post]), target[post], rtol=1e-11)
        np.testing.assert_allclose(actual_anchored[post], actual[post] / actual[at_anchor], rtol=1e-12)
        np.testing.assert_allclose(target_anchored[at_anchor], 1, rtol=1e-12)
        np.testing.assert_allclose(actual - target, (actual - fixed) + (fixed - target), equal_nan=True, rtol=1e-12, atol=1e-14)
        assert np.all(target[post] >= 1)
        if ed == 0 and ec == 0:
            np.testing.assert_allclose(target[post], 1)
            np.testing.assert_allclose(target_anchored[post], 1)

        old = summary[(summary["mode"] == mode)
                      & np.isclose(summary.epsilon_dyn, ed, atol=1e-14, rtol=0)
                      & np.isclose(summary.epsilon_constant, ec, atol=1e-14, rtol=0)].iloc[0]
        np.testing.assert_allclose([actual[-1], actual_anchored[-1]], [old.raw_final, old.anchored_final], rtol=1e-12)
        excess = target[-1] - 1
        # Both are reported. Whole-ratio error can conceal a much larger error
        # relative to a small added-effect contrast. The paired response also
        # accounts for the estimator's nonzero no-effect baseline discrepancy.
        unpaired_recovery = 100 * (actual[-1] - 1) / excess if excess > 0 else np.nan
        paired_recovery = 100 * (actual[-1] - null_raw[-1]) / excess if excess > 0 else np.nan
        raw_error = actual / target - 1
        anchored_error = actual_anchored / target_anchored - 1
        rows.append(dict(mode=mode, epsilon_dyn=ed, epsilon_constant=ec,
                         target_raw_final=target[-1], kcor_raw_final=actual[-1],
                         target_at_anchor=target[at_anchor], kcor_at_anchor=actual[at_anchor],
                         target_anchored_final=target_anchored[-1], kcor_anchored_final=actual_anchored[-1],
                         raw_relative_error_percent=100 * raw_error[-1],
                         anchored_relative_error_percent=100 * anchored_error[-1],
                         max_abs_raw_relative_error_percent=100 * np.max(np.abs(raw_error[post])),
                         max_abs_anchored_relative_error_percent=100 * np.max(np.abs(anchored_error[post])),
                         target_added_raw_contrast=excess,
                         kcor_raw_excess_above_one=actual[-1]-1,
                         kcor_paired_added_raw_contrast=actual[-1]-null_raw[-1],
                         raw_excess_recovery_percent=unpaired_recovery,
                         paired_added_contrast_recovery_percent=paired_recovery,
                         raw_refitting_shift=actual[-1]-fixed[-1],
                         raw_fixed_fit_minus_target=fixed[-1]-target[-1],
                         anchored_refitting_shift=actual_anchored[-1]-fixed_anchored[-1],
                         anchored_fixed_fit_minus_target=fixed_anchored[-1]-target_anchored[-1],
                         theta_A=old.theta_A, peak_added_hazard_over_href=old.peak_added_hazard_over_href,
                         fit_status_A=old.fit_status_A))
        block["reference_raw"] = target
        block["reference_anchored"] = target_anchored
        block["baseline_cumulative_post_skip"] = B
        block["added_cumulative_post_skip"] = G
        block["raw_relative_error_percent"] = 100 * raw_error
        block["anchored_relative_error_percent"] = 100 * anchored_error
        block["paired_raw_shift"] = actual - null_raw
        block["paired_anchored_shift"] = actual_anchored - null_anchored
        series.append(block)

    result = pd.DataFrame(rows)
    series = pd.concat(series, ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.outdir / "reference_comparison.csv", index=False)
    series.to_csv(args.outdir / "reference_trajectories.csv", index=False)
    selected = result[((result["mode"] == "dynamic") & (result.epsilon_dyn == 0.0005))
                      | ((result["mode"] == "comorbidity") & (result.epsilon_constant == 0.0001))
                      | ((result["mode"] == "combined") & (result.epsilon_dyn == 0.0005)
                         & (result.epsilon_constant == 0.0001))]
    # Plot only the existing smallest fading case; keep the wider grid in tables.
    example = series[(series["mode"] == "dynamic") & (series.epsilon_dyn == 0.0005)]
    example = example[example.rebased_week >= 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for ax, actual, reference, title in zip(axes, ["raw", "anchored"],
                                           ["reference_raw", "reference_anchored"],
                                           ["Before anchoring", "After identical anchoring"]):
        ax.axvspan(4, 56, color="#dfe8ef", alpha=0.7, label="Added mortality / fitting window")
        ax.axhline(1, color="#777777", linewidth=0.8)
        ax.plot(example.rebased_week, example[reference], color="#157f60", lw=2.2, label="Defined reference")
        ax.plot(example.rebased_week, example[actual], color="#9348a6", lw=1.8, ls="--", label="KCOR")
        ax.axvline(4, color="#777777", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Weeks after initial two-week skip")
        ax.set_ylabel("Cumulative hazard contrast")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Existing fading-effect case: the reference itself falls below 1 after anchoring", fontsize=11)
    fig.savefig(args.outdir / "fading_reference_comparison.png", dpi=160)
    plt.close(fig)

    inputs = [args.step1 / n for n in ["manifest.json", "expected_count_results.csv", "expected_count_trajectories.csv"]]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    reference_definition="Remove baseline depletion denominator; retain additive hazard. Q=(B+G)/B. Q_anchored=Q/Q(anchor). B,G start after the same skip as the experiment.",
                    reference_scope="Specified discrete generator only; not a ratio of death risks or a model-free causal estimate.",
                    no_effect_raw_kcor=float(null_raw[-1]), no_effect_anchored_kcor=float(null_anchored[-1]),
                    inputs={str(p.resolve().relative_to(ROOT)): digest(p) for p in inputs},
                    script_sha256=digest(Path(__file__)),
                    validation="All 26 saved Step 1 settings read and verified; survival/log-hazard, null, anchoring, decomposition and endpoint checks passed. Original Step 1 sources match recorded hashes.")
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    columns = KEYS + ["target_raw_final", "kcor_raw_final", "target_anchored_final", "kcor_anchored_final", "raw_excess_recovery_percent", "paired_added_contrast_recovery_percent"]
    print(selected[columns].to_string(index=False))
    nonzero = result[(result.epsilon_dyn > 0) | (result.epsilon_constant > 0)]
    print(nonzero[["raw_relative_error_percent", "anchored_relative_error_percent", "raw_excess_recovery_percent", "paired_added_contrast_recovery_percent", "max_abs_raw_relative_error_percent", "max_abs_anchored_relative_error_percent"]].agg(["min", "max"]).to_string())
    print(f"Saved {len(result)} settings to {args.outdir}")


if __name__ == "__main__":
    main()

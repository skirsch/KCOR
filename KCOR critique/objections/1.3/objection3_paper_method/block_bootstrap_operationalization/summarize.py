"""Audit saved aggregate-block draws and summarize empirical interval coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from run import HERE, PAPER, STARTS, arrays, block_counts, configure, estimate, sample_dataset


def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - radius), float(center + radius)]


def summarize(phase):
    out = HERE / phase
    assert (out / "COMPLETE.json").exists(), "Run is incomplete"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == phase
    for rel, wanted in manifest["hashes"].items():
        assert hashlib.sha256((PAPER.parent / rel).read_bytes()).hexdigest() == wanted, rel
    n, b = manifest["n"], manifest["b"]
    first, last = manifest["design"]["quiet_window_inclusive"]
    records = []
    point = []
    center_shift = []
    width = []
    covered = 0
    lower_miss = 0
    upper_miss = 0
    point_outside_interval = 0
    failed_draws = 0
    warm_start_mismatches = 0
    theta_near_zero = np.zeros(2, int)
    original_near_zero = np.zeros(2, int)
    both_original_theta_above_point01 = 0
    covered_among_both_above_point01 = 0
    fit_rmse_by_coverage = {"covered": [], "missed": []}
    for i in range(n):
        row = json.loads((out / f"{i:04d}.json").read_text(encoding="utf-8"))
        with np.load(out / f"{i:04d}.npz") as z:
            draws = z["draws"]
            theta = z["theta"]
        assert row["dataset"] == i and row["truth"] == manifest["design"]["truth"]
        assert draws.shape == (b,) and theta.shape == (b, 2)
        valid = draws[np.isfinite(draws)]
        assert len(valid) == row["valid_draws"] and len(valid) >= .99 * b
        lo, hi = np.quantile(valid, [.025, .975])
        assert np.allclose([lo, hi], row["interval"], atol=1e-12, rtol=0)
        assert row["covered"] == bool(lo <= row["truth"] <= hi)
        assert row["theta_near_zero_draws"] == np.sum(theta < .01, axis=0).tolist()
        assert row["bootstrap_median"] == float(np.median(valid))
        records.append(row)
        point.append(row["point"])
        center_shift.append(row["bootstrap_median"] - row["point"])
        width.append(hi - lo)
        covered += row["covered"]
        lower_miss += lo > row["truth"]
        upper_miss += hi < row["truth"]
        point_outside_interval += not (lo <= row["point"] <= hi)
        failed_draws += row["failed_draws"]
        warm_start_mismatches += row["warm_start_check_mismatches"]
        theta_near_zero += np.sum(theta < .01, axis=0)
        original_near_zero += [f["theta"] < .01 for f in row["fits"]]
        both_above = all(f["theta"] >= .01 for f in row["fits"])
        both_original_theta_above_point01 += both_above
        covered_among_both_above_point01 += both_above and row["covered"]
        fit_rmse_by_coverage["covered" if row["covered"] else "missed"].append(
            float(np.mean([np.sqrt(f["objective"] / (last - first + 1)) for f in row["fits"]])))
    cfg = configure(manifest["scenario"]["windows"])
    for i in sorted({0, n // 2, n - 1}):
        row = records[i]
        counts, truth = sample_dataset(manifest["scenario"], np.random.default_rng(row["source_seed"]))
        assert hashlib.sha256(counts.tobytes()).hexdigest() == row["counts_sha256"]
        assert truth == row["truth"]
        rng = np.random.default_rng(row["bootstrap_seed"])
        boot = block_counts(counts, rng, manifest["block_length"], manifest["design"]["source_segments_inclusive"])
        deaths, risk = arrays(boot)
        assert np.array_equal(risk[:, 1:], risk[:, :-1] - deaths[:, :-1])
        warm = tuple((f["theta"],) for f in row["fits"])
        replayed, _ = estimate(boot, cfg, first, last, manifest["design"]["target_week"], warm)
        if row["warm_start_check_mismatches"]:
            replayed, _ = estimate(boot, cfg, first, last, manifest["design"]["target_week"],
                                   (STARTS, STARTS))
        with np.load(out / f"{i:04d}.npz") as z:
            assert abs(replayed - z["draws"][0]) < 1e-12
    point = np.asarray(point)
    center_shift = np.asarray(center_shift)
    width = np.asarray(width)
    report = dict(phase=phase, datasets=n, draws_per_dataset=b, block_length=manifest["block_length"],
        true_value=manifest["design"]["truth"], nominal_coverage=.95,
        covered=int(covered), coverage=float(covered / n), coverage_wilson_95=wilson(covered, n),
        one_sided_binomial_p_vs_95=float(binomtest(covered, n, .95, alternative="less").pvalue),
        lower_endpoint_above_truth=int(lower_miss), upper_endpoint_below_truth=int(upper_miss),
        point_outside_own_interval=int(point_outside_interval),
        point_mean=float(point.mean()), point_sd=float(point.std(ddof=1)),
        point_rmse=float(np.sqrt(np.mean((point - 1) ** 2))),
        mean_bootstrap_median_minus_point=float(center_shift.mean()),
        median_bootstrap_median_minus_point=float(np.median(center_shift)),
        mean_interval_width=float(width.mean()),
        failed_draws=int(failed_draws), warm_start_check_mismatches=int(warm_start_mismatches),
        original_theta_near_zero_by_arm=original_near_zero.tolist(),
        both_original_theta_at_least_point01=int(both_original_theta_above_point01),
        coverage_among_both_original_theta_at_least_point01=(
            int(covered_among_both_above_point01) / both_original_theta_above_point01
            if both_original_theta_above_point01 else None),
        mean_fit_rmse_by_coverage={k: float(np.mean(v)) if v else None
                                   for k, v in fit_rmse_by_coverage.items()},
        bootstrap_theta_near_zero_by_arm=theta_near_zero.tolist(),
        total_bootstrap_draws=n*b,
        scope="One explicit aggregate-block implementation of the manuscript outline, one window and one stratum. Full manuscript diagnostic-pass coverage is unavailable.")
    audit = dict(passed=True, files=n, checked_draw_files=n, source_hashes_verified=True,
                 replayed_dataset_indices=sorted({0, n // 2, n - 1}),
                 risk_set_recurrence_verified=True, intervals_reconstructed=True)
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (out / "audit.json").open("x", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "pilot", "main", "sensitivity"))
    summarize(parser.parse_args().phase)

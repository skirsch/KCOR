"""Audit paired paper-fit intervals under person and aggregate-block resampling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from run_person import HERE, PAPER, BLOCK, CAL, STARTS, configure, estimate, sample_dataset, bootstrap


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [float(center-radius), float(center+radius)]


def summarize(phase):
    out = HERE / phase
    assert (out / "COMPLETE.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "person" and manifest["phase"] == phase
    for rel, digest in manifest["hashes"].items():
        assert hashlib.sha256((PAPER.parent / rel).read_bytes()).hexdigest() == digest, rel
    block_dir = BLOCK / "main"
    block_manifest = json.loads((block_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n"] <= block_manifest["n"] and manifest["b"] <= block_manifest["b"]
    n, b = manifest["n"], manifest["b"]
    k_person = k_block = both = person_only = block_only = neither = 0
    prefix_person_covered = prefix_block_covered = 0
    person_lower_above = person_upper_below = point_outside = 0
    failed = warm_mismatch = 0
    shifts, widths, points = [], [], []
    original_near_zero = 0
    both_theta_above = both_theta_above_covered = 0
    paired_rows = []
    for i in range(n):
        row = json.loads((out / f"{i:04d}.json").read_text(encoding="utf-8"))
        block = json.loads((block_dir / f"{i:04d}.json").read_text(encoding="utf-8"))
        with np.load(out / f"{i:04d}.npz") as z:
            draws = z["draws"]
            theta = z["theta"]
        if b >= 199:
            with np.load(block_dir / f"{i:04d}.npz") as z:
                block_draws = z["draws"]
            person_prefix = draws[:99][np.isfinite(draws[:99])]
            block_prefix = block_draws[:99][np.isfinite(block_draws[:99])]
            assert len(person_prefix) >= 95 and len(block_prefix) >= 95
            p_lo, p_hi = np.quantile(person_prefix, [.025, .975])
            b_lo, b_hi = np.quantile(block_prefix, [.025, .975])
            prefix_person_covered += p_lo <= 1.0 <= p_hi
            prefix_block_covered += b_lo <= 1.0 <= b_hi
        assert row["dataset"] == block["dataset"] == i
        assert row["source_seed"] == block["source_seed"]
        assert row["counts_sha256"] == block["counts_sha256"]
        assert row["truth"] == block["truth"] == 1.0
        assert row["point"] == block["point"]
        assert row["paired_block_interval"] == block["interval"]
        assert row["paired_block_covered"] == block["covered"]
        assert draws.shape == (b,) and theta.shape == (b, 2)
        valid = draws[np.isfinite(draws)]
        assert len(valid) == row["valid_draws"] and len(valid) >= .99*b
        lo, hi = np.quantile(valid, [.025, .975])
        assert np.allclose([lo, hi], row["interval"], atol=1e-12, rtol=0)
        assert bool(lo <= 1.0 <= hi) == row["covered"]
        assert row["theta_near_zero_draws"] == np.sum(theta < .01, axis=0).tolist()
        assert row["bootstrap_median"] == float(np.median(valid))
        person = bool(row["covered"])
        block_hit = bool(block["covered"])
        k_person += person
        k_block += block_hit
        both += person and block_hit
        person_only += person and not block_hit
        block_only += block_hit and not person
        neither += not person and not block_hit
        person_lower_above += lo > 1.0
        person_upper_below += hi < 1.0
        point_outside += not (lo <= row["point"] <= hi)
        failed += row["failed_draws"]
        warm_mismatch += row["warm_start_check_mismatches"]
        shifts.append(row["bootstrap_median"] - row["point"])
        widths.append(hi-lo)
        points.append(row["point"])
        original_near_zero += any(f["theta"] < .01 for f in row["fits"])
        above = all(f["theta"] >= .01 for f in row["fits"])
        both_theta_above += above
        both_theta_above_covered += above and person
        paired_rows.append(dict(dataset=i, point=row["point"],
            person_lower=float(lo), person_upper=float(hi), person_covered=person,
            block_lower=block["interval"][0], block_upper=block["interval"][1],
            block_covered=block_hit))
    cfg = configure(manifest["scenario"]["windows"])
    first, last = block_manifest["design"]["quiet_window_inclusive"]
    for i in sorted({0, n//2, n-1}):
        row = json.loads((out / f"{i:04d}.json").read_text())
        counts, truth = sample_dataset(manifest["scenario"], np.random.default_rng(row["source_seed"]))
        assert truth == row["truth"]
        assert hashlib.sha256(counts.tobytes()).hexdigest() == row["counts_sha256"]
        rng = np.random.default_rng(row["bootstrap_seed"])
        sample = bootstrap(counts, rng)
        assert int(sample.sum()) == int(counts.sum())
        warm = tuple((f["theta"],) for f in row["fits"])
        replayed, _ = estimate(sample, cfg, first, last, block_manifest["design"]["target_week"], warm)
        if row["warm_start_check_mismatches"]:
            replayed, _ = estimate(sample, cfg, first, last,
                                   block_manifest["design"]["target_week"], (STARTS, STARTS))
        with np.load(out / f"{i:04d}.npz") as z:
            assert abs(replayed-z["draws"][0]) < 1e-12
    points = np.asarray(points)
    report = dict(scope="Paper joint fit held fixed; same outer datasets; person versus one explicit aggregate-block rule",
        datasets=n, draws_per_dataset=b, truth=1.0, nominal_coverage=.95,
        person_covered=int(k_person), person_coverage=k_person/n,
        person_coverage_wilson_95=wilson(k_person,n),
        person_one_sided_binomial_p_vs_95=float(binomtest(k_person,n,.95,alternative="less").pvalue),
        block_covered_on_same_datasets=int(k_block), block_coverage_on_same_datasets=k_block/n,
        first_99_draw_coverage=(dict(person=int(prefix_person_covered),
            block=int(prefix_block_covered)) if b >= 199 else None),
        paired_outcomes=dict(both=int(both),person_only=int(person_only),
            block_only=int(block_only),neither=int(neither)),
        person_lower_endpoint_above_truth=int(person_lower_above),
        person_upper_endpoint_below_truth=int(person_upper_below),
        person_point_outside_own_interval=int(point_outside),
        mean_point=float(points.mean()), point_sd=float(points.std(ddof=1)),
        person_mean_bootstrap_median_minus_point=float(np.mean(shifts)),
        person_mean_interval_width=float(np.mean(widths)),
        person_failed_draws=int(failed),person_warm_start_check_mismatches=int(warm_mismatch),
        original_any_theta_below_point01=int(original_near_zero),
        both_original_theta_at_least_point01=int(both_theta_above),
        person_coverage_among_both_theta_at_least_point01=(
            both_theta_above_covered/both_theta_above if both_theta_above else None))
    audit = dict(passed=True, source_hashes_verified=True, paired_datasets=n,
        intervals_reconstructed=n, replayed_datasets=sorted({0,n//2,n-1}),
        first_person_bootstrap_draws_replayed=True,
        person_resampling_source="corrected_validation/simulation.py:bootstrap")
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    with (out / "audit.json").open("x", encoding="utf-8") as f:
        json.dump(audit,f,indent=2)
    with (out / "paired_intervals.json").open("x", encoding="utf-8") as f:
        json.dump(paired_rows,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("smoke","main"))
    summarize(parser.parse_args().phase)

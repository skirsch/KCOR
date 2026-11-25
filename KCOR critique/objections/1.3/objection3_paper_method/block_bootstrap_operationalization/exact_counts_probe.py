"""Post-pilot diagnosis: block bootstrap on exact expected cohort counts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from run import HERE, CAL, STARTS, block_counts, configure, estimate
from simulation import population


def main():
    design = json.loads((HERE / "design.json").read_text(encoding="utf-8"))
    all_scenarios = json.loads((CAL / "design.json").read_text(encoding="utf-8"))
    scenario = all_scenarios["common"] | next(s for s in all_scenarios["scenarios"]
                                             if s["id"] == design["primary_scenario"])
    probabilities, truth = population(scenario)
    counts = probabilities * np.asarray(scenario["arm_weights"])[:, None] * scenario["n"]
    cfg = configure(scenario["windows"])
    first, last = design["quiet_window_inclusive"]
    point, fits = estimate(counts, cfg, first, last, design["target_week"], (STARTS, STARTS))
    reports = []
    for length in (4, 8):
        rng = np.random.default_rng(2026111409 + length)
        draws = []
        for _ in range(199):
            boot = block_counts(counts, rng, length, design["source_segments_inclusive"])
            warm = tuple((f["theta"],) for f in fits)
            draw, _ = estimate(boot, cfg, first, last, design["target_week"], warm)
            draws.append(draw)
        draws = np.asarray(draws)
        reports.append(dict(block_length=length, bootstrap_median=float(np.median(draws)),
                            bootstrap_mean=float(np.mean(draws)),
                            percentile_interval=np.quantile(draws, [.025, .975]).tolist(),
                            median_minus_point=float(np.median(draws) - point)))
    report = dict(scope="Post-pilot diagnostic, not a prespecified coverage result",
                  exact_expected_count_point=point, truth=truth, block_results=reports,
                  interpretation="Any nonzero center shift here comes from block rearrangement, not random sampling of persons or deaths.")
    with (HERE / "exact_counts_block_check.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

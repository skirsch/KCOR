"""Post-hoc expected-count check of source-time placement choices."""
from __future__ import annotations
import json
import numpy as np
from run import HERE, CAL, STARTS, block_counts, configure, estimate
from simulation import population


def main():
    design = json.loads((HERE / "design.json").read_text())
    all_scenarios = json.loads((CAL / "design.json").read_text())
    scenario = all_scenarios["common"] | next(s for s in all_scenarios["scenarios"]
                                             if s["id"] == design["primary_scenario"])
    probs, truth = population(scenario)
    counts = probs * np.asarray(scenario["arm_weights"])[:, None] * scenario["n"]
    cfg = configure(scenario["windows"])
    first, last = design["quiet_window_inclusive"]
    original, fits = estimate(counts, cfg, first, last, design["target_week"], (STARTS, STARTS))
    choices = {
        "primary_segments": design["source_segments_inclusive"],
        "target_horizon_segment": [[0, 5], [6, 40], [41, 52], [53, 99]],
        "roughly_quarterly_segments": [[0, 5], [6, 17], [18, 29], [30, 40],
                                        [41, 52], [53, 64], [65, 76], [77, 88], [89, 99]],
    }
    results = []
    for name, segments in choices.items():
        rng = np.random.default_rng(2026111419)
        draws = []
        for _ in range(199):
            boot = block_counts(counts, rng, 4, segments)
            warm = tuple((f["theta"],) for f in fits)
            draw, _ = estimate(boot, cfg, first, last, design["target_week"], warm)
            draws.append(draw)
        results.append(dict(choice=name, segments=segments,
                            median=float(np.median(draws)),
                            percentile_interval=np.quantile(draws, [.025, .975]).tolist(),
                            median_minus_point=float(np.median(draws) - original)))
    report = dict(scope="Post-hoc diagnostic of an unspecified paper choice, not prespecified coverage",
                  truth=truth, exact_expected_count_point=original, results=results)
    with (HERE / "exact_counts_placement_check.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

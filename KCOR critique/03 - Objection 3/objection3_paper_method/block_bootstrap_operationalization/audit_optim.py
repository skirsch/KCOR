"""Check warm-start bootstrap fits against the paper fit's four-start optimum."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from run import HERE, CAL, STARTS, block_counts, configure, estimate, sample_dataset


def audit(phase):
    out = HERE / phase
    assert (out / "COMPLETE.json").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    cfg = configure(manifest["scenario"]["windows"])
    first, last = manifest["design"]["quiet_window_inclusive"]
    target = manifest["design"]["target_week"]
    checked = []
    for i in sorted({0, manifest["n"] // 2, manifest["n"] - 1}):
        row = json.loads((out / f"{i:04d}.json").read_text())
        counts, _ = sample_dataset(manifest["scenario"], np.random.default_rng(row["source_seed"]))
        rng = np.random.default_rng(row["bootstrap_seed"])
        selected = {0, manifest["b"] // 2, manifest["b"] - 1}
        with np.load(out / f"{i:04d}.npz") as z:
            saved = z["draws"].copy()
        for j in range(manifest["b"]):
            boot = block_counts(counts, rng, manifest["block_length"],
                                manifest["design"]["source_segments_inclusive"])
            if j not in selected:
                continue
            warm = tuple((f["theta"],) for f in row["fits"])
            p_warm, f_warm = estimate(boot, cfg, first, last, target, warm)
            p_full, f_full = estimate(boot, cfg, first, last, target, (STARTS, STARTS))
            if row["warm_start_check_mismatches"] and j == 0:
                assert abs(saved[j] - p_full) < 1e-12
            else:
                assert abs(saved[j] - p_warm) < 1e-12
            checked.append(dict(dataset=i, draw=j, warm_point=p_warm, full_point=p_full,
                max_objective_difference=max(abs(a["objective"] - b["objective"])
                                             for a, b in zip(f_warm, f_full)),
                point_difference=abs(p_warm - p_full)))
    report = dict(phase=phase, checked=checked,
                  max_point_difference=max(x["point_difference"] for x in checked),
                  max_objective_difference=max(x["max_objective_difference"] for x in checked))
    with (out / "optimization_audit.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "checked"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "main", "sensitivity"))
    audit(parser.parse_args().phase)

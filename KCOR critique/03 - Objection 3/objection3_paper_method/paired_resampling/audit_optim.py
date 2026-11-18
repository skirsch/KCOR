"""Replay selected person-bootstrap draws and check joint-fit starting values."""
from __future__ import annotations
import json
import numpy as np
from run_person import HERE, BLOCK, STARTS, configure, estimate, sample_dataset, bootstrap


def main():
    out = HERE / "main"
    assert (out / "COMPLETE.json").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    block_manifest = json.loads((BLOCK / "main/manifest.json").read_text())
    cfg = configure(manifest["scenario"]["windows"])
    first, last = block_manifest["design"]["quiet_window_inclusive"]
    target = block_manifest["design"]["target_week"]
    checked = []
    for i in sorted({0, manifest["n"] // 2, manifest["n"] - 1}):
        row = json.loads((out / f"{i:04d}.json").read_text())
        counts, _ = sample_dataset(manifest["scenario"], np.random.default_rng(row["source_seed"]))
        rng = np.random.default_rng(row["bootstrap_seed"])
        selected = {0, manifest["b"] // 2, manifest["b"] - 1}
        with np.load(out / f"{i:04d}.npz") as z:
            draws = z["draws"].copy()
        for j in range(manifest["b"]):
            sample = bootstrap(counts, rng)
            if j not in selected:
                continue
            warm = tuple((f["theta"],) for f in row["fits"])
            warm_point, warm_fits = estimate(sample, cfg, first, last, target, warm)
            full_point, full_fits = estimate(sample, cfg, first, last, target, (STARTS, STARTS))
            saved_match = full_point if j == 0 and row["warm_start_check_mismatches"] else warm_point
            assert abs(saved_match - draws[j]) < 1e-12
            checked.append(dict(dataset=i, draw=j,
                point_difference=abs(warm_point-full_point),
                max_objective_difference=max(abs(a["objective"]-b["objective"])
                    for a,b in zip(warm_fits,full_fits))))
    report = dict(checked=checked,
        max_point_difference=max(c["point_difference"] for c in checked),
        max_objective_difference=max(c["max_objective_difference"] for c in checked))
    with (out / "optimization_audit.json").open("x", encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!="checked"},indent=2))


if __name__ == "__main__":
    main()

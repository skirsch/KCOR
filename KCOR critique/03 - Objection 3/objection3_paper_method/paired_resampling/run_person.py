"""Apply actual person-resampling Monte Carlo to the paper joint fit, paired by dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
BLOCK = PAPER / "block_bootstrap_operationalization"
sys.path.insert(0, str(BLOCK))
from run import CAL, STARTS, configure, estimate, sample_dataset  # noqa: E402
from simulation import bootstrap  # noqa: E402


def run(phase):
    design = json.loads((HERE / "design.json").read_text(encoding="utf-8"))
    reference = BLOCK / "main"
    main_manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    assert (reference / "COMPLETE.json").exists()
    assert main_manifest["n"] == design["outer_datasets"]
    assert main_manifest["b"] == design["person_draws_per_dataset"]
    n = 2 if phase == "smoke" else design["outer_datasets"]
    b = 19 if phase == "smoke" else design["person_draws_per_dataset"]
    scenario = main_manifest["scenario"]
    cfg = configure(scenario["windows"])
    first, last = main_manifest["design"]["quiet_window_inclusive"]
    target = main_manifest["design"]["target_week"]
    source = [HERE / "run_person.py", HERE / "design.json", BLOCK / "run.py",
              BLOCK / "design.json", reference / "manifest.json", PAPER / "paper_step1.py",
              CAL / "simulation.py", CAL / "production_adapter.py", CAL / "design.json"]
    hashes = {str(p.relative_to(PAPER.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    out = HERE / phase
    out.mkdir(exist_ok=False)
    (out / "manifest.json").write_text(json.dumps(dict(phase=phase, n=n, b=b, method="person",
        paired_block_run=str(reference.relative_to(PAPER)), design=design, scenario=scenario,
        hashes=hashes, python=platform.python_version(), numpy=np.__version__,
        created_utc=datetime.now(timezone.utc).isoformat(), workers=1), indent=2), encoding="utf-8")
    if os.name == "nt":
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except (ImportError, OSError):
            pass
    begun = time.perf_counter()
    for i in range(n):
        paired = json.loads((reference / f"{i:04d}.json").read_text(encoding="utf-8"))
        counts, truth = sample_dataset(scenario, np.random.default_rng(paired["source_seed"]))
        counts_sha = hashlib.sha256(counts.tobytes()).hexdigest()
        assert counts_sha == paired["counts_sha256"] and truth == paired["truth"]
        point, fits = estimate(counts, cfg, first, last, target, (STARTS, STARTS))
        assert abs(point - paired["point"]) < 1e-12
        boot_seed = design["bootstrap_seed_namespace"] + i
        rng = np.random.default_rng(boot_seed)
        draws = np.empty(b)
        theta = np.empty((b, 2))
        failed = 0
        start_mismatches = 0
        for j in range(b):
            sample = bootstrap(counts, rng)
            warm = tuple((f["theta"],) for f in fits)
            value, fitted = estimate(sample, cfg, first, last, target, warm)
            if j == 0:
                checked, full_fits = estimate(sample, cfg, first, last, target, (STARTS, STARTS))
                if any(abs(a["objective"] - c["objective"]) > 1e-12
                       for a, c in zip(fitted, full_fits)):
                    start_mismatches += 1
                    value, fitted = checked, full_fits
            elif not np.isfinite(value) or not all(f["success"] for f in fitted):
                value, fitted = estimate(sample, cfg, first, last, target, (STARTS, STARTS))
            failed += int(not np.isfinite(value) or not all(f["success"] for f in fitted))
            draws[j] = value
            theta[j] = [f["theta"] for f in fitted]
        valid = draws[np.isfinite(draws)]
        lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        row = dict(dataset=i, source_seed=paired["source_seed"], bootstrap_seed=boot_seed,
            counts_sha256=counts_sha, truth=truth, point=point, fits=fits,
            interval=[float(lo), float(hi)], covered=bool(lo <= truth <= hi),
            valid_draws=len(valid), failed_draws=failed,
            warm_start_check_mismatches=start_mismatches,
            theta_near_zero_draws=np.sum(theta < .01, axis=0).tolist(),
            bootstrap_median=float(np.median(valid)) if len(valid) else None,
            paired_block_interval=paired["interval"], paired_block_covered=paired["covered"])
        (out / f"{i:04d}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        np.savez_compressed(out / f"{i:04d}.npz", draws=draws, theta=theta)
        if (i + 1) % max(1, min(10, n // 5)) == 0:
            elapsed = time.perf_counter() - begun
            print(f"{phase}: {i+1}/{n}; {elapsed:.1f}s; ETA {elapsed/(i+1)*(n-i-1):.1f}s", flush=True)
    assert all(hashlib.sha256((PAPER.parent / rel).read_bytes()).hexdigest() == want
               for rel, want in hashes.items())
    (out / "COMPLETE.json").write_text(json.dumps(dict(datasets=n, draws=b,
        seconds=time.perf_counter()-begun, finished_utc=datetime.now(timezone.utc).isoformat()),
        indent=2), encoding="utf-8")
    print(f"Completed {phase} in {time.perf_counter()-begun:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "main"))
    run(parser.parse_args().phase)

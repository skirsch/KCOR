"""Reproducible one-worker aggregate-block coverage study for paper Step 1."""
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
CAL = PAPER.parent / "objection3_corrected_validation"
sys.path.insert(0, str(CAL))
sys.path.insert(0, str(PAPER))
from production_adapter import arrays, configure  # noqa: E402
from simulation import sample_dataset  # noqa: E402
from paper_step1 import fit_step1  # noqa: E402

STARTS = (0.01, 0.5, 4.0, 20.0)


def block_counts(counts, rng, length, segments):
    """Resample source death fractions in runs; join at reconstructed risk sets."""
    deaths, risk = arrays(counts)
    n0 = counts.sum(axis=1).astype(np.int64)
    weeks = deaths.shape[1]
    output = np.zeros((2, weeks + 1), dtype=np.int64)
    for arm in (0, 1):
        current = int(n0[arm])
        for low, high in segments:
            assert 0 <= low <= high < weeks
            width = min(length, high - low + 1)
            dest = low
            while dest <= high:
                source = int(rng.integers(low, high - width + 2))
                take = min(width, high - dest + 1)
                for offset in range(take):
                    fraction = float(deaths[arm, source + offset] / risk[arm, source + offset])
                    count = min(current, max(0, int(np.rint(current * fraction))))
                    output[arm, dest + offset] = count
                    current -= count
                dest += take
        output[arm, -1] = current
    assert np.array_equal(output.sum(axis=1), n0)
    return output


def estimate(counts, cfg, first, last, target, starts):
    deaths, risk = arrays(counts)
    h = -np.log1p(-deaths / risk)
    t = np.arange(first, last + 1) - cfg["skip"]
    gamma = cfg["gompertz"]["gamma_per_week"]
    theta = []
    fits = []
    for arm in (0, 1):
        choices = [fit_step1(h[arm, first:last + 1], t, gamma, s) for s in starts[arm]]
        chosen = min(choices, key=lambda f: f["objective"])
        fits.append(chosen)
        theta.append(chosen["theta"])
    H = np.cumsum(np.where(np.arange(h.shape[1]) >= cfg["skip"], h, 0.0), axis=1)
    inv = np.asarray([np.expm1(th * H[arm]) / th if th > 1e-8 else H[arm]
                      for arm, th in enumerate(theta)])
    ratio = np.divide(inv[1], inv[0], out=np.full(inv.shape[1], np.nan), where=inv[0] > 0)
    point = float(ratio[target] / ratio[cfg["anchor"]])
    return point, fits


def run(phase, pilot_n=None, pilot_b=None):
    design = json.loads((HERE / "design.json").read_text(encoding="utf-8"))
    scenarios = json.loads((CAL / "design.json").read_text(encoding="utf-8"))
    scenario = scenarios["common"] | next(s for s in scenarios["scenarios"]
                                          if s["id"] == design["primary_scenario"])
    cfg = configure(scenario["windows"])
    first, last = design["quiet_window_inclusive"]
    target = design["target_week"]
    assert scenario["target"] == target and cfg["anchor"] == design["anchor_week"]
    assert cfg["skip"] == design["post_skip_week"]
    assert scenario["weeks"] == design["source_segments_inclusive"][-1][-1] + 1
    if phase == "smoke":
        n, b, length, offset = pilot_n or 2, pilot_b or 19, 4, 50000
    elif phase == "pilot":
        n, b, length, offset = design["outer_datasets_pilot"], design["bootstrap_draws_pilot"], 4, 0
    elif phase == "main":
        n, b, length, offset = design["outer_datasets_main"], design["bootstrap_draws_main"], 4, 10000
    elif phase == "sensitivity":
        n, b, length, offset = design["outer_datasets_sensitivity"], design["bootstrap_draws_sensitivity"], 8, 10000
    else:
        raise ValueError(phase)
    assert length == design[f"block_length_{'sensitivity' if phase == 'sensitivity' else 'primary'}_weeks"]
    assert n > 0 and b >= 19
    source = [HERE / "run.py", HERE / "design.json", PAPER / "paper_step1.py",
              CAL / "simulation.py", CAL / "production_adapter.py", CAL / "design.json",
              PAPER.parent / "objection3_resolution/source_snapshot/documentation/preprint/paper.md"]
    hashes = {str(p.relative_to(PAPER.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    out = HERE / phase
    out.mkdir(exist_ok=False)
    (out / "manifest.json").write_text(json.dumps(dict(phase=phase, n=n, b=b, block_length=length,
        seed_offset=offset, design=design, scenario=scenario, hashes=hashes,
        python=platform.python_version(), numpy=np.__version__, workers=1,
        created_utc=datetime.now(timezone.utc).isoformat()), indent=2), encoding="utf-8")
    if os.name == "nt":
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except (ImportError, OSError):
            pass
    begun = time.perf_counter()
    for i in range(n):
        source_seed = design["seed_namespace"] + offset + i
        boot_seed = design["seed_namespace"] + 100000 * length + offset + i
        counts, truth = sample_dataset(scenario, np.random.default_rng(source_seed))
        assert abs(truth - design["truth"]) < 1e-12
        point, fits = estimate(counts, cfg, first, last, target, (STARTS, STARTS))
        rng = np.random.default_rng(boot_seed)
        draws = np.empty(b)
        theta = np.empty((b, 2))
        failed = 0
        start_mismatches = 0
        for j in range(b):
            boot = block_counts(counts, rng, length, design["source_segments_inclusive"])
            warm = ((fits[0]["theta"],), (fits[1]["theta"],))
            draw, fitted = estimate(boot, cfg, first, last, target, warm)
            if j == 0:
                multi, check = estimate(boot, cfg, first, last, target, (STARTS, STARTS))
                if any(abs(a["objective"] - c["objective"]) > 1e-12
                       for a, c in zip(fitted, check)):
                    start_mismatches += 1
                    draw, fitted = multi, check
            elif not all(f["success"] for f in fitted) or not np.isfinite(draw):
                draw, fitted = estimate(boot, cfg, first, last, target, (STARTS, STARTS))
            failed += int(not all(f["success"] for f in fitted) or not np.isfinite(draw))
            draws[j] = draw
            theta[j] = [f["theta"] for f in fitted]
        valid = draws[np.isfinite(draws)]
        lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        record = dict(dataset=i, source_seed=source_seed, bootstrap_seed=boot_seed,
            counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest(), truth=truth,
            point=point, fits=fits, interval=[float(lo), float(hi)],
            covered=bool(lo <= truth <= hi), valid_draws=len(valid), failed_draws=failed,
            warm_start_check_mismatches=start_mismatches,
            theta_near_zero_draws=np.sum(theta < .01, axis=0).tolist(),
            bootstrap_median=float(np.median(valid)) if len(valid) else None)
        (out / f"{i:04d}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        np.savez_compressed(out / f"{i:04d}.npz", draws=draws, theta=theta)
        if (i + 1) % max(1, min(10, n // 5)) == 0:
            elapsed = time.perf_counter() - begun
            print(f"{phase}: {i+1}/{n}; {elapsed:.1f}s; ETA {elapsed/(i+1)*(n-i-1):.1f}s", flush=True)
    assert all(hashlib.sha256((PAPER.parent / rel).read_bytes()).hexdigest() == want
               for rel, want in hashes.items())
    (out / "COMPLETE.json").write_text(json.dumps(dict(datasets=n, bootstrap_draws=b,
        seconds=time.perf_counter()-begun, finished_utc=datetime.now(timezone.utc).isoformat()), indent=2),
        encoding="utf-8")
    print(f"Completed {phase} in {time.perf_counter()-begun:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "pilot", "main", "sensitivity"))
    parser.add_argument("--pilot-n", type=int)
    parser.add_argument("--pilot-b", type=int)
    args = parser.parse_args()
    run(args.phase, args.pilot_n, args.pilot_b)

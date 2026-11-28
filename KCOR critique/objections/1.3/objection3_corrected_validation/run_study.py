"""Run/resume the fixed study; all artifacts remain inside this module."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import scipy
from production_adapter import ROOT, SOURCE_ROOT, configure, estimate, intervals
from simulation import sample_dataset, bootstrap, population

HERE = Path(__file__).resolve().parent


def hashes():
    paths = [SOURCE_ROOT/'code/KCOR.py', SOURCE_ROOT/'code/KCOR_CMR.py', SOURCE_ROOT/'data/Czech/Czech.yaml']
    paths += sorted(HERE.glob('*.py')) + [HERE/'design.json', HERE/'DESIGN.md']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def job(task):
    scenario, si, rep, namespace, B, prefix, out = task
    windows, target = scenario['windows'], scenario['target']
    data_rng = np.random.default_rng(np.random.SeedSequence([namespace, si, rep, 0]))
    boot_rng = np.random.default_rng(np.random.SeedSequence([namespace, si, rep, 1]))
    counts, truth = sample_dataset(scenario, data_rng)
    original = estimate(counts, windows, target, iteration=0, total_iterations=B+1)
    draws = np.empty(B)
    fit_failures = accepted_boot = 0
    flags = {}
    for b in range(B):
        result = estimate(bootstrap(counts, boot_rng), windows, target,
                          iteration=b+1, total_iterations=B+1)
        draws[b] = result['point']
        accepted_boot += int(result['accepted'])
        fit_failures += int(any(not d['success'] for d in result['diagnostics']))
        for d in result['diagnostics']:
            key = d['status']
            flags[key] = flags.get(key, 0) + 1
    result = dict(scenario=scenario['id'], repetition=rep,
                  seed_components=[namespace, si, rep], truth=truth,
                  point=original['point'], accepted=original['accepted'],
                  diagnostics=original['diagnostics'],
                  intervals=intervals(original['point'], draws),
                  prefix_intervals=intervals(original['point'], draws[:min(prefix, B)]),
                  bootstrap_fit_failures=fit_failures, accepted_bootstrap=accepted_boot,
                  bootstrap_status_counts=flags,
                  counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())
    path = Path(out)/scenario['id']
    path.mkdir(exist_ok=True)
    # Different jobs own different names; fail if results already exist.
    npz = path/f'{rep:04d}.npz'
    with npz.open('xb') as f:
        np.savez_compressed(f, counts=counts, draws=draws)
    with (path/f'{rep:04d}.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    return scenario['id'], rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['development', 'final'], required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError('Output must stay inside objection3_corrected_validation')
    design = json.loads((HERE/'design.json').read_text())
    R, B = (design['outer_repetitions'], design['bootstrap_draws']) if args.phase == 'final' else (3, 49)
    namespace = design[f'{args.phase}_seed_namespace']
    manifest_path = out/'manifest.json'
    source_hashes = hashes()
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        assert manifest['hashes'] == source_hashes, 'Source/config changed: cannot resume'
        assert manifest['phase'] == args.phase
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(phase=args.phase, created=datetime.now(timezone.utc).isoformat(),
                        repetitions=R, draws=B, namespace=namespace, design=design,
                        hashes=source_hashes, python=platform.python_version(),
                        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                        production_config=configure(design['common']['windows']),
                        command=sys.argv, workers=args.workers)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    tasks = []
    for si, specific in enumerate(design['scenarios']):
        scenario = design['common'] | specific
        for rep in range(R):
            complete = out/scenario['id']/f'{rep:04d}.json'
            if complete.exists():
                assert complete.with_suffix('.npz').exists()
                continue
            assert not complete.with_suffix('.npz').exists(), 'Partial job needs diagnosis; do not overwrite'
            tasks.append((scenario, si, rep, namespace, B, design['precision_prefix_draws'], str(out)))
    started = time.perf_counter()
    print(f'{args.phase}: {len(tasks)} pending jobs, {B} bootstrap draws each', flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            sid, rep = future.result()
            if i % 50 == 0 or i == len(tasks):
                print(f'{i}/{len(tasks)} complete; {sid}/{rep}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
    assert hashes() == source_hashes, 'Sources changed during run'
    with (out/'COMPLETE.json').open('x', encoding='utf-8') as f:
        json.dump(dict(completed=datetime.now(timezone.utc).isoformat(), jobs=R*len(design['scenarios']),
                       elapsed_seconds=time.perf_counter()-started), f, indent=2)


if __name__ == '__main__':
    main()

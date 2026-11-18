"""Audit retained study results and report prespecified calibration measures."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from production_adapter import intervals, estimate
from simulation import sample_dataset, bootstrap


def wilson(successes, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    args = p.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run/'manifest.json').read_text())
    assert (run/'COMPLETE.json').exists()
    R, B, design = manifest['repetitions'], manifest['draws'], manifest['design']
    summaries, details, replay = [], [], []
    hashes_seen = set()
    for si, specific in enumerate(design['scenarios']):
        scenario = design['common'] | specific
        rows = []
        prefix_deltas = []
        for rep in range(R):
            path = run/scenario['id']/f'{rep:04d}.json'
            row = json.loads(path.read_text())
            saved = np.load(path.with_suffix('.npz'))
            draws, counts = saved['draws'], saved['counts']
            assert len(draws) == B
            digest = hashlib.sha256(counts.tobytes()).hexdigest()
            assert digest == row['counts_sha256'] and digest not in hashes_seen
            hashes_seen.add(digest)
            for key in ('percentile', 'basic'):
                np.testing.assert_allclose(intervals(row['point'], draws)[key], row['intervals'][key], equal_nan=True)
            if rep in (0, R-1):
                ns = manifest['namespace']
                regen, truth = sample_dataset(scenario, np.random.default_rng(np.random.SeedSequence([ns,si,rep,0])))
                np.testing.assert_array_equal(regen, counts)
                rng = np.random.default_rng(np.random.SeedSequence([ns,si,rep,1]))
                for b in range(8):
                    r = estimate(bootstrap(counts, rng), scenario['windows'], scenario['target'],
                                 iteration=b+1, total_iterations=B+1)
                    np.testing.assert_allclose(r['point'], draws[b], rtol=1e-12, equal_nan=True)
                point = estimate(counts, scenario['windows'], scenario['target'], total_iterations=B+1)
                np.testing.assert_allclose(point['point'], row['point'], rtol=1e-12, equal_nan=True)
                assert point['accepted'] == row['accepted']
                replay.append(dict(scenario=scenario['id'], repetition=rep, checked_draws=8))
            lo, hi = row['intervals']['percentile']
            plo, phi = row['prefix_intervals']['percentile']
            prefix_deltas.append(max(abs(lo-plo), abs(hi-phi)))
            rows.append(row)
            details.append(dict(scenario=scenario['id'], repetition=rep, truth=row['truth'],
                                point=row['point'], accepted=row['accepted'],
                                percentile_low=lo, percentile_high=hi,
                                basic_low=row['intervals']['basic'][0], basic_high=row['intervals']['basic'][1],
                                bootstrap_fit_failures=row['bootstrap_fit_failures']))
        accepted = np.array([r['accepted'] for r in rows])
        error = np.array([r['point']-r['truth'] for r in rows])
        for kind in ('percentile', 'basic'):
            ci = np.array([r['intervals'][kind] for r in rows])
            valid = np.isfinite(ci).all(axis=1) & (ci[:,0] <= ci[:,1])
            truths = np.array([r['truth'] for r in rows])
            covered = valid & (ci[:,0] <= truths) & (truths <= ci[:,1])
            pc = np.array([r['prefix_intervals'][kind] for r in rows])
            prefix_covered = np.isfinite(pc).all(axis=1) & (pc[:,0] <= truths) & (truths <= pc[:,1])
            coverage = float(covered.mean())
            bounds = wilson(int(covered.sum()), R)
            lower, upper = design['acceptable_coverage_range']
            status = ('undercoverage' if bounds[1] < lower else
                      'adequate_in_tested_scenario' if bounds[0] >= lower and bounds[1] <= upper
                      and float((~valid).mean()) <= design['maximum_interval_failure_rate'] else 'inconclusive_or_conservative')
            conditional_n = int((accepted & valid).sum())
            conditional_success = int((accepted & covered).sum())
            summaries.append(dict(scenario=scenario['id'], scope=scenario['scope'], interval=kind,
                repetitions=R, truth=rows[0]['truth'], coverage=coverage, coverage_wilson=bounds,
                coverage_among_produced=float(covered[valid].mean()) if valid.any() else None,
                coverage_status=status, accepted=int(accepted.sum()), accepted_produced=conditional_n,
                conditional_coverage=conditional_success/conditional_n if conditional_n else None,
                conditional_wilson=wilson(conditional_success, conditional_n),
                conditional_status='insufficient_accepted_analyses' if conditional_n < design['minimum_accepted_repetitions'] else 'assess_wilson_interval',
                interval_failure_rate=float((~valid).mean()),
                original_fit_failure_rate=float(np.mean([any(not d['success'] for d in r['diagnostics']) for r in rows])),
                bootstrap_fit_failure_rate=float(sum(r['bootstrap_fit_failures'] for r in rows)/(R*B)),
                mean_width=float(np.mean((ci[:,1]-ci[:,0])[valid])) if valid.any() else None,
                mean_error=float(np.mean(error)), rmse=float(np.sqrt(np.mean(error**2))),
                prefix_coverage=float(prefix_covered.mean()),
                precision_coverage_change=float(coverage-prefix_covered.mean()),
                precision_stable=bool(abs(coverage-prefix_covered.mean()) <= design['maximum_precision_coverage_change']),
                percentile_prefix_endpoint_change_median=float(np.nanmedian(prefix_deltas))))
    dest = run/'assessment'
    dest.mkdir(exist_ok=False)
    (dest/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    pd.DataFrame(details).to_csv(dest/'replicates.csv', index=False)
    (dest/'audit.json').write_text(json.dumps(dict(checks='passed', independent_datasets=len(hashes_seen),
        retained_draws=len(hashes_seen)*B, all_intervals_reconstructed=True, seed_replays=replay), indent=2), encoding='utf-8')
    lines = ['# Calibration results', '', '| Scenario | Interval | Coverage (95% Wilson interval) | Accepted | Mean width | Mean error |',
             '|---|---|---:|---:|---:|---:|']
    for r in summaries:
        a,b = r['coverage_wilson']
        lines.append(f"| {r['scenario']} | {r['interval']} | {r['coverage']:.1%} ({a:.1%}–{b:.1%}) | {r['accepted']}/{R} | {r['mean_width']:.4f} | {r['mean_error']:.4f} |")
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

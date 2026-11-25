"""Verify completion evidence without rerunning or changing the fixed study.

Run only after the 1,000-sample age-specific production comparison is verified. The numerical
audit has already reconstructed every retained interval; this checks its scope,
source integrity, production equivalence, and reporting denominators.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(HERE)
    run = HERE/'final_2026-11-10'
    m = read(run/'manifest.json')
    end = read(run/'COMPLETE.json')
    numeric = read(run/'assessment/audit.json')
    summary = read(run/'assessment/summary.json')
    expected = m['repetitions']*len(m['design']['scenarios'])
    assert m['phase'] == 'final' and expected == 4800 and m['draws'] == 999
    assert end['jobs'] == expected
    assert numeric['checks'] == 'passed' and numeric['independent_datasets'] == expected
    assert numeric['retained_draws'] == expected*m['draws']
    assert numeric['all_intervals_reconstructed']
    source_checks = []
    for rel, want in m['hashes'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == want, rel
        source_checks.append(rel)
    snapshot = read(HERE/'source_snapshot/manifest.json')
    for rel, want in snapshot['source_hashes'].items():
        for base in (ROOT, HERE/'source_snapshot'):
            assert hashlib.sha256((base/rel).read_bytes()).hexdigest() == want, str(base/rel)
    production = []
    for name, size in [('reproduction_single_iteration', 1), ('reproduction_04', 4),
                       ('reproduction_10_iterations', 10), ('reproduction_final_geometry', 10)]:
        p = read(HERE/'development'/name/'verification.json')
        assert p['checks'] == 'passed'
        assert len(p['max_curve_errors']) == size
        assert max(p['max_curve_errors']) < 1e-10
        for rel, want in p['source_sha256'].items():
            assert m['hashes'][rel] == want
        production.append(dict(fixture=name, iterations=size, max_error=max(p['max_curve_errors'])))
    full_size = HERE/'development/reproduction_1000_iterations'
    captured = read(full_size/'verification_age_scope.json')
    capture = read(full_size/'production_capture.json')
    disposition = read(full_size/'run_disposition.json')
    assert captured['checks'] == 'passed' and captured['rows'] == 100000
    assert captured['total_iterations'] == 1000 and len(captured['max_curve_errors']) == 1000
    assert max(captured['max_curve_errors']) < 1e-10 and captured['theta_columns_verified']
    assert captured['production_source_sha256'] == hashlib.sha256((ROOT/'code/KCOR.py').read_bytes()).hexdigest()
    assert capture['csv_sha256'] == hashlib.sha256((full_size/'production_age_rows.csv').read_bytes()).hexdigest()
    assert not captured['full_workbook_export_completed']
    assert disposition['status'] == 'intentionally_stopped_after_requested_calculation_verified'
    production.append(dict(fixture='reproduction_1000_iterations', iterations=1000,
                           max_error=captured['max_curve_error'], rows=100000,
                           scope=captured['scope'], full_workbook_export_completed=False))
    for name in ('person_aggregation.json', 'model_checks.json',
                 'layout_and_status_evidence.json', 'final_batch_time_geometry.json'):
        assert read(HERE/'development'/name)['checks'] == 'passed'
    model = read(HERE/'development/model_checks.json')
    for case in model['scenarios']:
        np.testing.assert_allclose(case['truth'], case['exact_inversion_reference'], rtol=1e-12, atol=1e-12)
    rows = pd.read_csv(run/'assessment/replicates.csv')
    assert len(rows) == expected
    assert not rows[['scenario', 'repetition']].duplicated().any()
    assert len(summary) == 2*len(m['design']['scenarios'])
    for specific in m['design']['scenarios']:
        sid = specific['id']
        assert len(list((run/sid).glob('*.json'))) == m['repetitions']
        assert len(list((run/sid).glob('*.npz'))) == m['repetitions']
        replayed = [r for r in numeric['seed_replays'] if r['scenario'] == sid]
        assert sorted(r['repetition'] for r in replayed) == [0, m['repetitions']-1]
        assert all(r['checked_draws'] == 8 for r in replayed)
        subset = rows[rows.scenario == sid]
        assert len(subset) == m['repetitions']
        for result in (r for r in summary if r['scenario'] == sid):
            kind = result['interval']
            valid = np.isfinite(subset[[kind+'_low', kind+'_high']]).all(axis=1)
            covered = valid & (subset[kind+'_low'] <= subset.truth) & (subset.truth <= subset[kind+'_high'])
            np.testing.assert_allclose(covered.mean(), result['coverage'], atol=1e-14)
            assert int(subset.accepted.sum()) == result['accepted']
            if result['accepted'] == 0:
                assert result['conditional_coverage'] is None
                assert result['conditional_wilson'] == [None, None]
            assert result['scope'] == specific['scope']
            assert result['precision_stable'] == (abs(result['precision_coverage_change']) <= m['design']['maximum_precision_coverage_change'])
    evidence = dict(checks='passed', completed=datetime.now(timezone.utc).isoformat(),
                    independent_datasets=expected, retained_draws=numeric['retained_draws'],
                    dataset_seed_replays=len(numeric['seed_replays']),
                    bootstrap_seed_replays=sum(r['checked_draws'] for r in numeric['seed_replays']),
                    frozen_hashes_checked=source_checks, source_snapshot_files=len(snapshot['source_hashes']),
                    production_comparisons=production, report_coverage_recomputed_from_replicates=True,
                    conditional_denominators_checked=True, development_final_separation=(m['namespace'] != m['design']['development_seed_namespace']))
    assert evidence['development_final_separation']
    sensitivity = read(run/'assessment/target_sensitivity.json')['results']
    assert len(sensitivity) == 48
    targets = {r['scenario']: r for r in model['scenarios']}
    for result in sensitivity:
        subset = rows[rows.scenario == result['scenario']]
        target = targets[result['scenario']][result['target_kind']]
        assert result['target'] == target
        kind = result['interval']
        covered = (subset[kind+'_low'] <= target) & (target <= subset[kind+'_high'])
        np.testing.assert_allclose(covered.mean(), result['coverage'], atol=1e-14)
    evidence['target_sensitivity_verified'] = True
    evidence['final_module_file_hashes'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(list(HERE.glob('*.py')) + list(HERE.glob('*.md')))
    }
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(evidence, f, indent=2)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

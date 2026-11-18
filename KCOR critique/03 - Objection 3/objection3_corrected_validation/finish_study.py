"""Audit and compare the corrected run after summarize.py has reconstructed it."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from production_adapter import ROOT
from run_study import hashes
from summarize import wilson

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'objection3_resolution'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    run = Path(sys.argv[1]).resolve()
    assert run.is_relative_to(HERE)
    manifest = read(run/'manifest.json')
    design = manifest['design']
    assert manifest['phase'] == 'final'
    assert manifest['hashes'] == hashes(), 'Frozen execution sources changed'
    assert manifest['namespace'] != read(OLD/'design.json')['final_seed_namespace']
    for key in ['outer_repetitions', 'bootstrap_draws', 'precision_prefix_draws', 'nominal_coverage',
                'acceptable_coverage_range', 'minimum_accepted_repetitions',
                'maximum_interval_failure_rate', 'maximum_precision_coverage_change', 'scenarios', 'common']:
        assert design[key] == read(OLD/'design.json')[key], key
    n = design['outer_repetitions']
    B = design['bootstrap_draws']
    expected = n*len(design['scenarios'])
    assert read(run/'COMPLETE.json')['jobs'] == expected == 4800
    numeric = read(run/'assessment/audit.json')
    assert numeric['checks'] == 'passed' and numeric['independent_datasets'] == expected
    assert numeric['retained_draws'] == expected*B == 4795200
    assert numeric['all_intervals_reconstructed']
    assert len(numeric['seed_replays']) == 16
    assert sum(r['checked_draws'] for r in numeric['seed_replays']) == 128
    p = read(HERE/'development/production_10/verification.json')
    assert p['checks'] == 'passed' and p['total_iterations'] == 10
    assert max(p['max_curve_errors']) < 1e-10
    for rel, want in p['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == want
    provenance = read(HERE/'provenance.json')
    for name, want in provenance['reused_source_sha256'].items():
        assert hashlib.sha256((OLD/name).read_bytes()).hexdigest() == want, name
    for rel, want in provenance['source_snapshot_sha256'].items():
        assert hashlib.sha256((HERE/rel).read_bytes()).hexdigest() == want, rel
    cmr = read(OLD/'development/person_aggregation.json')
    assert cmr['checks'] == 'passed'
    assert cmr['source_sha256'] == hashlib.sha256((HERE/'source_snapshot/code/KCOR_CMR.py').read_bytes()).hexdigest()
    # The earlier final execution inputs remain frozen; never reinterpret its results.
    for rel, want in read(OLD/'final_2026-11-10/manifest.json')['hashes'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == want, rel
    model = read(HERE/'development/model_checks.json')
    assert model['checks'] == 'passed'
    targets = {r['scenario']:r for r in model['scenarios']}
    for r in targets.values():
        np.testing.assert_allclose(r['truth'], r['exact_inversion_reference'], atol=1e-12, rtol=1e-12)
        assert r['single_population_point'] == r['production_population_point']
    current = read(run/'assessment/summary.json')
    old = {(r['scenario'], r['interval']):r for r in read(OLD/'final_2026-11-10/assessment/summary.json')}
    comparison, sensitivity = [], []
    lower, upper = design['acceptable_coverage_range']
    def status(bounds, size, failure, stable):
        if size < design['minimum_accepted_repetitions']:
            return 'insufficient_accepted_analyses'
        if not stable:
            return 'bootstrap_precision_inconclusive'
        if bounds[1] < lower:
            return 'undercoverage'
        if bounds[0] >= lower and bounds[1] <= upper and failure <= design['maximum_interval_failure_rate']:
            return 'adequate_in_tested_scenario'
        return 'inconclusive_or_conservative'
    for specific in design['scenarios']:
        sid = specific['id']
        rows = [read(run/sid/f'{rep:04d}.json') for rep in range(n)]
        assert len(list((run/sid).glob('*.json'))) == n
        assert len(list((run/sid).glob('*.npz'))) == n
        accepted = np.array([r['accepted'] for r in rows])
        truth = np.array([r['truth'] for r in rows])
        for row in rows:
            gate = all(d['success'] and not d['degenerate'] and not any(d['flags'].values()) for d in row['diagnostics']) and np.isfinite(row['point'])
            assert gate == row['accepted']
        for result in [r for r in current if r['scenario'] == sid]:
            kind = result['interval']
            ci = np.array([r['intervals'][kind] for r in rows])
            valid = np.isfinite(ci).all(axis=1) & (ci[:,0] <= ci[:,1])
            covered = valid & (ci[:,0] <= truth) & (truth <= ci[:,1])
            prefix = np.array([r['prefix_intervals'][kind] for r in rows])
            prefix_valid = np.isfinite(prefix).all(axis=1) & (prefix[:,0] <= prefix[:,1])
            prefix_hit = prefix_valid & (prefix[:,0] <= truth) & (truth <= prefix[:,1])
            np.testing.assert_allclose(covered.mean(), result['coverage'], atol=1e-14)
            assert int(accepted.sum()) == result['accepted']
            assert int((accepted & valid).sum()) == result['accepted_produced']
            num, den = int((accepted & covered).sum()), int(accepted.sum())
            conditional = num/den if den else None
            bounds = wilson(num, den)
            failure = float((~valid)[accepted].mean()) if den else None
            delta = conditional-float(prefix_hit[accepted].mean()) if den else None
            stable = abs(delta) <= design['maximum_precision_coverage_change'] if den else False
            item = dict(result, previous_coverage=old[(sid,kind)]['coverage'],
                previous_accepted=old[(sid,kind)]['accepted'],
                conditional_all_accepted_coverage=conditional,
                conditional_all_accepted_wilson=bounds, conditional_interval_failure_rate=failure,
                conditional_precision_change=delta, conditional_precision_stable=stable,
                conditional_assessment=status(bounds,den,failure,stable))
            comparison.append(item)
            fitted_target = targets[sid]['production_population_point']
            fitted_hit = valid & (ci[:,0] <= fitted_target) & (fitted_target <= ci[:,1])
            sensitivity.append(dict(scenario=sid, interval=kind, target=fitted_target,
                target_kind='fitted_population_output_secondary_only',
                coverage=float(fitted_hit.mean()), coverage_wilson=wilson(int(fitted_hit.sum()), n),
                conditional_coverage=float(fitted_hit[accepted].mean()) if den else None,
                accepted=den))
    dest = run/'assessment'
    for filename, data in [('comparison.json', comparison), ('target_sensitivity.json', sensitivity)]:
        with (dest/filename).open('x',encoding='utf-8') as f:
            json.dump(data,f,indent=2)
    lines = ['# Corrected versus original calibration', '',
        'Fresh random seeds; identical scenarios and statistical rules. These are scenario-level comparisons, not paired datasets.', '',
        '| Scenario | Interval | Old coverage | Corrected coverage | Accepted | Conditional coverage | Conditional assessment |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in comparison:
        cc = r['conditional_all_accepted_coverage']
        conditional_text = f'{cc:.1%}' if cc is not None else 'undefined'
        lines.append(f"| {r['scenario']} | {r['interval']} | {r['previous_coverage']:.1%} | {r['coverage']:.1%} | {r['accepted']}/{n} | {conditional_text} | {r['conditional_assessment']} |")
    lines += ['', 'Quiet-window contamination is a separately labelled assumption-violation stress. Boundary and low-information cases are identified in the JSON scope fields.',
        '', 'Coverage against the fitted population output is a secondary diagnostic only; it does not replace the model-defined target.',
        '', 'This run does not resolve the separate manuscript-versus-code resampling description or identify historical published-output provenance.']
    with (dest/'COMPARISON.md').open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    audit = dict(checks='passed', datasets=expected, retained_bootstrap_draws=expected*B,
        seed_replays=16, replayed_bootstrap_estimates=128, execution_hashes_checked=len(manifest['hashes']),
        original_execution_hashes_unchanged=True, original_module_sources_unchanged=True,
        scenarios_and_statistical_rules_unchanged=True, fresh_final_seeds=True,
        production_comparison_samples=10, production_max_curve_error=max(p['max_curve_errors']),
        unchanged_person_bootstrap_verification_reused=True,
        conditional_gate_and_denominators_rechecked=True, all_intervals_reconstructed=True)
    with (dest/'completion_audit.json').open('x',encoding='utf-8') as f:
        json.dump(audit,f,indent=2)
    print('\n'.join(lines))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()

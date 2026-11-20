"""Compare all completed production age curves at the actual final batch size."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from production_adapter import ROOT, K, estimate, intervals
from simulation import sample_dataset, bootstrap


def main():
    base = Path(sys.argv[1]).resolve()
    capture = json.loads((base/'production_capture.json').read_text())
    assert capture['csv_sha256'] == hashlib.sha256((base/'production_age_rows.csv').read_bytes()).hexdigest()
    assert capture['production_source_sha256'] == hashlib.sha256((ROOT/'code/KCOR.py').read_bytes()).hexdigest()
    design = json.loads((Path(__file__).parent/'design.json').read_text())
    scenario = design['common'] | design['scenarios'][1]
    rng = np.random.default_rng(415001)
    original, truth = sample_dataset(scenario, rng)
    actual = pd.read_csv(base/'production_age_rows.csv', parse_dates=['Date'], float_precision='round_trip')
    assert len(actual) == 100000 and set(actual.mc_id) == set(range(1000))
    errors, values, records = [], [], []
    for i in range(1000):
        counts = original if i == 0 else bootstrap(original, rng)
        expected = estimate(counts, scenario['windows'], scenario['target'],
                            iteration=i, total_iterations=1000, return_curves=True)
        rows = actual[actual.mc_id == i].sort_values('Date')
        assert len(rows) == 100
        np.testing.assert_allclose(rows.KCOR, expected['curve'], rtol=1e-10, atol=1e-12, equal_nan=True)
        for d, col in [(0, 'theta_den'), (1, 'theta_num')]:
            np.testing.assert_allclose(rows[col], expected['diagnostics'][d]['theta'], rtol=1e-10, atol=1e-12)
        errors.append(float(np.nanmax(np.abs(rows.KCOR.to_numpy()-expected['curve']))))
        point = rows.iloc[scenario['target']]
        values.append(float(point.KCOR))
        records.append(dict(Iteration=i, Dose_num=1, Dose_den=0, YearOfBirth=1950,
                            Date=point.Date, KCOR=float(point.KCOR)))
    # Use production's real summary helper on the actual production values.
    lines = []
    K.create_mc_summary(records, lines.append, enrollment_label='2021_24')
    ci = intervals(values[0], values[1:])
    for label, key in [('Percentile 95% CI', 'percentile'), ('Basic bootstrap 95% CI', 'basic')]:
        lo, hi = ci[key]
        assert any(f'{label}: [{lo:.4f}, {hi:.4f}]' in s for s in lines)
    with (base/'captured_production_interval_summary.log').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    result = dict(checks='passed', scope='all completed age-specific production trajectories and actual-value interval summary',
                  full_workbook_export_completed=False, total_iterations=1000, rows=100000,
                  max_curve_errors=errors, max_curve_error=max(errors), truth=truth,
                  original_point=values[0], intervals=ci, theta_columns_verified=True,
                  production_source_sha256=capture['production_source_sha256'],
                  production_capture_sha256=hashlib.sha256((base/'production_capture.json').read_bytes()).hexdigest(),
                  note='Unused aggregate output was still being computed during capture; these values are not a substitute fitter.')
    with (base/'verification_age_scope.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print('Passed: 1000 complete production age trajectories (100000 points), theta columns, and interval summary.')
    print('Maximum absolute curve error:', max(errors))


if __name__ == '__main__':
    main()

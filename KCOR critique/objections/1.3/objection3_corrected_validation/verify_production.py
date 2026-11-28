"""Development equivalence checks against the actual production workbook path."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from production_adapter import K, ROOT, SOURCE_ROOT, START, configure, estimate, intervals, workbook_frame
from simulation import sample_dataset, bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--design-case', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    scenario = dict(n=100000, theta=[4., .5], k=.001, weeks=100, target=80)
    if args.design_case:
        design = json.loads((Path(__file__).parent/'design.json').read_text())
        scenario = design['common'] | design['scenarios'][1]
    windows = scenario.get('windows', ((6, 55),))
    rng = np.random.default_rng(415001)
    original, truth = sample_dataset(scenario, rng)
    counts = [original] + [bootstrap(original, rng) for _ in range(args.draws)]
    estimates = [estimate(c, windows, scenario['target'], return_curves=True, iteration=i, total_iterations=len(counts))
                 for i, c in enumerate(counts)]
    src, dst = args.out / 'input.xlsx', args.out / 'production.xlsx'
    with pd.ExcelWriter(src) as writer:
        pd.concat([workbook_frame(c, i) for i, c in enumerate(counts)]).to_excel(
            writer, sheet_name='2021_24', index=False)
    configure(windows)
    with (args.out / 'production_stdout.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            K.process_workbook(str(src), str(dst))
    xls = pd.ExcelFile(dst)
    print('Production sheets:', xls.sheet_names)
    output = pd.concat([pd.read_excel(xls, s) for s in xls.sheet_names], ignore_index=True)
    print('Production columns:', list(output.columns))
    output.to_json(args.out / 'production_rows.json', orient='records', indent=2, date_format='iso')
    # MC output schemas retain either Iteration or mc_id; inspect actual result.
    it_col = next(c for c in ('mc_id', 'Iteration') if c in output)
    differences = []
    for i, result in enumerate(estimates):
        rows = output[(output[it_col] == i) & (output['YearOfBirth'] == 1950) &
                      (output['Dose_num'] == 1) & (output['Dose_den'] == 0)].sort_values('Date')
        actual = rows['KCOR'].to_numpy()
        np.testing.assert_allclose(actual, result['curve'], rtol=1e-10, atol=1e-12, equal_nan=True)
        differences.append(float(np.nanmax(np.abs(actual - result['curve']))))
    # Direct pandas person resampling followed by category aggregation equals
    # counting the exact same sampled person indices (including arm variation).
    cells = np.repeat(np.arange(original.size), original.ravel())
    people = pd.DataFrame({'cell': cells})
    sampled = people.sample(n=len(people), replace=True, random_state=7)
    indices = np.random.RandomState(7).choice(len(people), size=len(people), replace=True)
    np.testing.assert_array_equal(np.bincount(sampled.cell, minlength=original.size),
                                  np.bincount(cells[indices], minlength=original.size))
    # Quantile construction is separately checked against production log output.
    values = [r['point'] for r in estimates]
    records = [dict(Iteration=i, Dose_num=1, Dose_den=0, YearOfBirth=1950,
                    Date=START+pd.Timedelta(weeks=scenario['target']), KCOR=v) for i, v in enumerate(values)]
    lines = []
    K.create_mc_summary(records, lines.append, enrollment_label='2021_24')
    ci = intervals(values[0], values[1:])
    for label, key in ([('Percentile 95% CI', 'percentile'), ('Basic bootstrap 95% CI', 'basic')] if args.draws else []):
        lo, hi = ci[key]
        assert any(f'{label}: [{lo:.4f}, {hi:.4f}]' in s for s in lines)
    (args.out / 'interval_summary.log').write_text('\n'.join(lines), encoding='utf-8')
    start = time.perf_counter()
    for _ in range(50):
        estimate(bootstrap(original, rng), windows, scenario['target'])
    manifest = dict(checks='passed', max_curve_errors=differences, truth=truth,
                    total_iterations=len(counts), points=values,
                    diagnostics=[e['diagnostics'] for e in estimates],
                    seconds_per_estimate=(time.perf_counter()-start)/50,
                    config=configure(windows), scenario=scenario,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [SOURCE_ROOT/'code/KCOR.py', SOURCE_ROOT/'code/KCOR_CMR.py', SOURCE_ROOT/'data/Czech/Czech.yaml']})
    (args.out / 'verification.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

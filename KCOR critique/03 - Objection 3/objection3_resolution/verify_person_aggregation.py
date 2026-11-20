"""Execute the original CMR function without its unrelated top-level file I/O."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from production_adapter import ROOT, START, workbook_frame
from simulation import sample_dataset


def verify():
    source = ROOT / 'code/KCOR_CMR.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'process_enrollment_data')
    namespace = {'os': os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    counts, _ = sample_dataset(dict(n=12000, theta=[4., .5], k=.003), np.random.default_rng(415002))
    cells = np.repeat(np.arange(counts.size), counts.ravel())
    dose, week = np.divmod(cells, counts.shape[1])
    dates = pd.Series(START + pd.to_timedelta(week*7+2, unit='D'))
    dates.loc[week == counts.shape[1]-1] = pd.NaT
    people = pd.DataFrame(dict(DateOfDeath=dates, WeekOfDeath=dates.dt.strftime('%G-%V'),
                               YearOfBirth=1950, Sex='O', DCCI=0, Date_COVID_death=pd.NaT))
    for name in ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']:
        people[f'Date_{name}Dose'] = pd.NaT
    people.loc[dose == 1, 'Date_FirstDose'] = START-pd.Timedelta(days=14)
    evidence = []
    for seed in (0, 7, 31):
        draw = people.copy() if seed == 0 else people.sample(n=len(people), replace=True, random_state=seed)
        # Retain original person indices to recover each sampled lifetime cell.
        sampled_counts = np.bincount(cells[draw.index], minlength=counts.size).reshape(counts.shape)
        actual, *_ = namespace['process_enrollment_data'](
            draw.reset_index(drop=True), START, '2021-24', max_dose=1, monte_carlo_mode=True)
        expected = workbook_frame(sampled_counts, seed)
        merged = actual.merge(expected, on=['ISOweekDied','YearOfBirth','Dose'], suffixes=('_actual','_expected'))
        assert len(merged) == 2*(counts.shape[1]-1), 'CMR follow-up length differs'
        for col in ('Dead', 'Alive'):
            np.testing.assert_array_equal(merged[f'{col}_actual'], merged[f'{col}_expected'])
        evidence.append(dict(seed=seed, rows=len(merged), arm_totals=sampled_counts.sum(axis=1).tolist()))
    return dict(checks='passed', function='process_enrollment_data',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), comparisons=evidence)


if __name__ == '__main__':
    result = verify()
    destination = Path(sys.argv[1])
    with destination.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

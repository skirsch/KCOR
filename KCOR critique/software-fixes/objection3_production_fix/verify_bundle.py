"""Verify red/green regressions and replay retained evidence in the fixed copy."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    out = HERE/'verification'
    out.mkdir(exist_ok=False)
    manifest = json.loads((HERE/'manifest.json').read_text())
    before = hashlib.sha256((ROOT/'code/KCOR.py').read_bytes()).hexdigest()
    assert before == manifest['original_sha256']
    tests = HERE/'submission/test/monte_carlo/test_sample_isolation.py'
    test_results = {}
    for label, source, expected in [('original', ROOT/'code/KCOR.py', 1),
                                     ('fixed', HERE/'submission/code/KCOR.py', 0)]:
        env = os.environ.copy()
        env['KCOR_TEST_SOURCE'] = str(source)
        env['KCOR_FIX_RESULTS'] = str(out/label)
        with (out/f'{label}.log').open('w', encoding='utf-8') as log:
            run = subprocess.run([sys.executable, str(tests)], env=env, stdout=log, stderr=subprocess.STDOUT)
        assert run.returncode == expected, f'{label}: unexpected exit {run.returncode}'
        text = (out/f'{label}.log').read_text(encoding='utf-8')
        if label == 'original':
            assert 'FAILED (failures=2)' in text and 'Ran 3 tests' in text
        else:
            assert 'Ran 3 tests' in text and '\nOK\n' in text
        test_results[label] = dict(exit_code=run.returncode, expected=expected)
        print(label, 'regression result confirmed:', run.returncode, flush=True)
    for key,value in dict(DATASET='Czech', MONTE_CARLO='1', MC_INCLUDE_AGGREGATES='0',
                           MC_ENROLLMENT_DATE='2021_24', MC_YOB_START='1950', MC_YOB_END='1950').items():
        os.environ[key] = value
    source = HERE/'submission/code/KCOR.py'
    spec = importlib.util.spec_from_file_location('kcor_fixed_replay', source)
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    K._load_dataset_dose_pairs_config()
    K._DATASET_COVID_CORRECTION_CACHE = {}
    K._DATASET_DOSE_PAIRS_CACHE = {'2021_24':[(1,0)], 'default':[(1,0)]}
    K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE = [dict(start_iso='2021-30', end_iso='2022-27',
        start_int=202130, end_int=202227)]
    artifacts = ROOT/'KCOR critique/objections/1.3/objection3_resolution/development'
    replay = out/'historical_replay'
    replay.mkdir()
    old_input = artifacts/'reproduction_10_iterations/input.xlsx'
    old_input_hash = hashlib.sha256(old_input.read_bytes()).hexdigest()
    with (replay/'console.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            K.process_workbook(str(old_input), str(replay/'fixed.xlsx'))
    fixed = pd.read_excel(replay/'fixed.xlsx')
    single = pd.read_excel(artifacts/'reproduction_single_iteration/production.xlsx')
    buggy = pd.read_excel(artifacts/'reproduction_10_iterations/production.xlsx')
    def original_curve(frame):
        return frame[(frame.mc_id == 0) & (frame.YearOfBirth == 1950)].sort_values('Date').reset_index(drop=True)
    f,s,b = map(original_curve,(fixed,single,buggy))
    np.testing.assert_allclose(f.KCOR, s.KCOR, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert f.theta0_status_num.eq('OK').all() and f.theta0_status_den.eq('OK').all()
    assert b.theta0_status_num.eq('NOT_IDENTIFIED').all()
    assert hashlib.sha256(old_input.read_bytes()).hexdigest() == old_input_hash
    original_unchanged = hashlib.sha256((ROOT/'code/KCOR.py').read_bytes()).hexdigest() == before
    assert original_unchanged
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['patched_sha256']
    check = subprocess.run(['git','apply','--check',str(HERE/'KCOR_MC_sample_isolation.patch')],
                           cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    result = dict(checks='passed', tests=test_results, historical_replay=dict(
        original_input_sha256=old_input_hash, points_compared=len(f),
        max_curve_error=float(np.nanmax(np.abs(f.KCOR-s.KCOR))),
        week80_old_joint=float(b.KCOR.iloc[80]), week80_original_alone=float(s.KCOR.iloc[80]),
        week80_fixed_joint=float(f.KCOR.iloc[80]), original_fit_status='OK'),
        original_source_unchanged=original_unchanged, patch_applies=True,
        coverage_revalidated=False,
        note='Regression and historical replay establish these software fixes. Fresh coverage validation is separate work.')
    (out/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

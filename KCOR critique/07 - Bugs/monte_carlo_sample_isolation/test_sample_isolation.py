"""Regression tests for independent MC clocks and iteration-aware diagnostics.

Run directly or with unittest/pytest. KCOR_TEST_SOURCE optionally selects an
unpatched source for a red-test replay. KCOR_FIX_RESULTS retains test artifacts.
"""
from __future__ import annotations
import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

for key, value in dict(DATASET='Czech', MONTE_CARLO='1', MC_INCLUDE_AGGREGATES='0',
                       MC_ENROLLMENT_DATE='2021_24', MC_YOB_START='1950', MC_YOB_END='1950').items():
    os.environ[key] = value


class SampleIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        default = Path(__file__).resolve().parents[2]/'code/KCOR.py'
        path = Path(os.environ.get('KCOR_TEST_SOURCE', str(default)))
        spec = importlib.util.spec_from_file_location('kcor_regression', path)
        cls.K = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.K)
        cls.base = Path(os.environ.get('KCOR_FIX_RESULTS') or tempfile.mkdtemp(prefix='kcor_mc_test_'))
        cls.base.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        K = self.K
        K.MONTE_CARLO_MODE = True
        K._load_dataset_dose_pairs_config()
        K._DATASET_COVID_CORRECTION_CACHE = {}
        K._DATASET_DOSE_PAIRS_CACHE = {'2021_24': [(1, 0)], 'default': [(1, 0)]}
        start = pd.Timestamp('2021-06-14')
        labels = [(start+pd.Timedelta(weeks=w)).strftime('%G-%V') for w in (6,40)]
        K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE = [dict(
            start_iso=labels[0], end_iso=labels[1], start_int=K.iso_label_to_int(labels[0]), end_int=K.iso_label_to_int(labels[1]))]
        self.out = self.base/self._testMethodName
        self.out.mkdir(exist_ok=False)

    @staticmethod
    def frames():
        dates = pd.date_range('2021-06-14', periods=53, freq='7D')
        t = np.arange(53)
        deaths = np.array([100-(.6*t).astype(int), 100+(.1*t).astype(int)])
        cells = np.c_[deaths, 100000-deaths.sum(axis=1)]
        rng = np.random.default_rng(81442)
        frames = []
        for i in range(6):
            draw = cells if i == 0 else rng.multinomial(int(cells.sum()), cells.ravel()/cells.sum()).reshape(cells.shape)
            for dose in (0,1):
                dead = draw[dose,:-1]
                alive = draw[dose].sum()-np.r_[0,np.cumsum(dead)[:-1]]
                frames.append(pd.DataFrame(dict(DateDied=dates, ISOweekDied=dates.strftime('%G-%V'),
                    YearOfBirth=1950, Sex='O', Dose=dose, Dead=dead, Alive=alive, mc_id=i)))
        return pd.concat(frames, ignore_index=True)

    def workbook(self, frame, name):
        directory = self.out/name
        directory.mkdir()
        source, target = directory/'input.xlsx', directory/'output.xlsx'
        frame.to_excel(source, sheet_name='2021_24', index=False)
        with (directory/'console.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                self.K.process_workbook(str(source), str(target))
        result = pd.read_excel(target)
        return result[result.YearOfBirth == 1950].sort_values(['mc_id','Date']).reset_index(drop=True)

    def test_point_and_curves_do_not_depend_on_other_samples(self):
        frame = self.frames()
        single = self.workbook(frame[frame.mc_id == 0], 'single')
        joint = self.workbook(frame, 'joint')
        columns = ['KCOR','CI_lower','CI_upper','theta_num','theta_den',
                   'adj_cum_hazard_num','adj_cum_hazard_den','t']
        np.testing.assert_allclose(joint[joint.mc_id == 0][columns], single[columns],
                                   rtol=1e-10, atol=1e-12, equal_nan=True)
        mapping = {0:0, 1:7, 2:19, 3:25, 4:99, 5:105}
        renamed = frame.assign(mc_id=frame.mc_id.map(mapping)).sample(frac=1, random_state=38)
        permuted = self.workbook(renamed, 'permuted_and_relabelled')
        for before, after in mapping.items():
            np.testing.assert_allclose(joint[joint.mc_id == before][columns],
                permuted[permuted.mc_id == after][columns], rtol=1e-10, atol=1e-12, equal_nan=True)
        # Normal mode retains the same per-cohort clock and estimate.
        self.K.MONTE_CARLO_MODE = False
        normal_frame = frame[frame.mc_id == 0].drop(columns='mc_id')
        directory = self.out/'normal'
        directory.mkdir()
        normal_frame.to_excel(directory/'input.xlsx', sheet_name='2021_24', index=False)
        with (directory/'console.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                self.K.process_workbook(str(directory/'input.xlsx'), str(directory/'output.xlsx'))
        normal = pd.read_excel(directory/'output.xlsx')
        normal = normal[normal.YearOfBirth == 1950].sort_values('Date')
        np.testing.assert_allclose(normal[columns], single[columns], rtol=1e-10, atol=1e-12, equal_nan=True)

    def summary(self, mc):
        self.K.MONTE_CARLO_MODE = mc
        def params(theta, status='OK'):
            return dict(theta0_hat=theta, theta_hat=theta, k_hat=.001 if status=='OK' else np.nan,
                        success=status=='OK', theta0_status=status, theta_fit_status=status,
                        total_quiet_deaths=100, n_quiet_bins_prewave=35)
        records = [dict(EnrollmentDate='2021_24', Date=pd.Timestamp('2022-06-13'), YearOfBirth=1950,
                        Dose_num=1, Dose_den=0, KCOR=1.0, CI_lower=.9, CI_upper=1.1)]
        if mc:
            records = [dict(records[0], mc_id=i) for i in (0,7)]
            mapping = {('2021_24',0,1950,0):params(1), ('2021_24',0,1950,1):params(2),
                       ('2021_24',7,1950,0):params(np.nan,'NOT_IDENTIFIED'),
                       ('2021_24',7,1950,1):params(9),
                       ('2022_06',3,1950,0):params(0,'NOT_IDENTIFIED'), ('bad',):params(0,'NOT_IDENTIFIED')}
        else:
            mapping = {('2021_24',1950,0):params(0,'NOT_IDENTIFIED'), ('2021_24',1950,1):params(2)}
        self.K.create_summary_file(pd.DataFrame(records), str(self.out/'output.xlsx'), lambda *a:None, mapping)
        with pd.ExcelFile(self.out/'KCOR_summary.xlsx') as book:
            return {name:pd.read_excel(book,name) for name in book.sheet_names}

    def test_mc_diagnostics_count_failed_fits_and_preserve_iteration(self):
        sheets = self.summary(True)
        counts = sheets['theta0_status_summary'].iloc[0]
        self.assertEqual(int(counts.NOT_IDENTIFIED),1)
        self.assertEqual(int(counts.OK),3)
        self.assertTrue(bool(counts.any_non_ok_status))
        self.assertIn('mc_id=7',counts.quality_flag_detail)
        rows = sheets['2021_24'].dropna(subset=['mc_id']).set_index('mc_id')
        self.assertEqual(rows.loc[0,'theta0_hat_num'],2)
        self.assertEqual(rows.loc[7,'theta0_hat_num'],9)
        self.assertFalse(bool(rows.loc[0,'Row_Theta0_NonOK']))
        self.assertTrue(bool(rows.loc[7,'Row_Theta0_NonOK']))
        diagnostics = sheets['gamma_frailty_fit']
        self.assertEqual(len(diagnostics),4)
        failed = diagnostics[(diagnostics.mc_id == 7) & (diagnostics.dose == 0)].iloc[0]
        self.assertEqual(failed.theta0_status,'NOT_IDENTIFIED')
        self.assertTrue(pd.isna(failed.theta0_hat))

    def test_normal_diagnostic_keys_keep_their_existing_format(self):
        sheets = self.summary(False)
        counts = sheets['theta0_status_summary'].iloc[0]
        self.assertEqual(int(counts.NOT_IDENTIFIED),1)
        self.assertEqual(int(counts.OK),1)
        self.assertEqual(counts.quality_flag_detail,'(1950,0,NOT_IDENTIFIED)')
        self.assertNotIn('mc_id',sheets['2021_24'].columns)
        self.assertNotIn('mc_id',sheets['gamma_frailty_fit'].columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)

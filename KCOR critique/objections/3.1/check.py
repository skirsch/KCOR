"""Check the gap correction on the five noiseless histories and the ordinary Table 8 workbook."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/"1.3"), str(HERE.parent.parent/"iterative review/evaluations/3")]
import table8_bootstrap as table
from gap_check import history


def check_gap():
    table.compare.initialize("gap-refit")
    model = table.compare.adapter.K
    for n, k, stop, reentry, end in ((90,.002,30,40,90), (60,.001,20,40,60),
                                     (100,.001,30,60,90), (100,.001,40,60,100), (100,.001,24,50,80)):
        week = np.arange(n, dtype=float)
        quiet = (week < stop) | ((week >= reentry) & (week < end))
        def fitted_curve(theta):
            hazard = history(model, week, k, theta)
            (_, estimate), diag = table.compare.joint_fit(hazard, week, quiet, 4, .085/52)
            assert diag["success"] and diag["status"] == "ok"
            for raw, applied in zip(diag["delta_raw_values"], diag["delta_applied_values"]):
                assert applied == max(raw, 0)
            return estimate, diag, model.invert_gamma_frailty(np.cumsum(hazard), estimate)
        estimate, diag, curve = fitted_curve(4)
        np.testing.assert_allclose(estimate, 4, atol=1e-6, rtol=0)
        assert diag["delta_negative_clamped"]
        if n == 90:
            _, _, reference = fitted_curve(0)
            ratio = curve/reference
            anchor = model.KCOR_NORMALIZATION_WEEKS_EFFECTIVE - model.DYNAMIC_HVE_SKIP_WEEKS
            np.testing.assert_allclose(ratio[-1]/ratio[anchor], 1, atol=1e-5, rtol=0)
    print("Five gap cases and anchored ratio passed.")


def check_workbook():
    expected = json.loads((HERE/"table8_bootstrap.json").read_text())["summary"]
    workbook_bytes = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB], cwd=HERE)
    populations = [table.read_population(io.BytesIO(workbook_bytes), sheet) for sheet in table.SHEETS]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, workbook = root/"input.xlsx", root/"output.xlsx"
        source.write_bytes(workbook_bytes)
        # The child releases output workbook handles before we read the pooled values.
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--write-workbook",
                        str(source), str(workbook)], cwd=root, check=True)
        points = table.workbook_points(workbook, populations)
        np.testing.assert_allclose(points, [expected[s]["joint"] for s in table.SHEETS], rtol=1e-10)
    print("All six corrected points matched the ordinary workbook.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", action="store_true")
    parser.add_argument("--write-workbook", nargs=2)
    args = parser.parse_args()
    if args.write_workbook:
        os.environ["MONTE_CARLO"] = "0"
        table.compare.initialize("gap-refit")
        model = table.compare.adapter.K
        model.OVERRIDE_DOSE_PAIRS = list(table.PAIRS)
        model.fit_theta0_gompertz = table.compare.joint_fit
        model.process_workbook(*args.write_workbook)
    else:
        check_gap()
        if args.workbook:
            check_workbook()

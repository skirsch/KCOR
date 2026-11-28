"""Compare the scan with the ordinary workbook entry point for every placement."""
import contextlib
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from functools import partial
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from scan import HERE, KCOR, historical, population


def check_window(args):
    root, window = args
    start = window.start_iso
    last = historical.iso_week_to_monday(*map(int, window.end_iso.split("-"))) - timedelta(weeks=1)
    settings = [dict(start_iso=start, end_iso=last.strftime("%G-%V"),
                     start_int=int(start.replace("-", "")), end_int=int(last.strftime("%G%V")))]
    work = root / start
    work.mkdir()
    fitter = partial(KCOR.fit_theta0_gompertz, joint_seed=True)
    with contextlib.chdir(work), patch.object(KCOR, "fit_theta0_gompertz", fitter), patch.object(
            KCOR, "get_theta_estimation_windows", return_value=settings):
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            KCOR.process_workbook(str(root / "input.xlsx"), str(work / "output.xlsx"))
        with pd.ExcelFile("output.xlsx") as book:
            sheet = pd.read_excel(book, sheet_name="theta0_diagnostics")
    observed = sheet[sheet.YearOfBirth.isin([1940, 1950]) & sheet.Dose.isin([0, 1, 2])].set_index(["YearOfBirth", "Dose"]).sort_index()
    expected = pd.read_csv(HERE / "windows.csv")
    reference = expected[expected.window_start_iso == start].set_index(["yob_decade", "dose"]).sort_index()
    assert observed.index.tolist() == reference.index.tolist()
    np.testing.assert_allclose(observed.k_hat, reference.k, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(observed.theta0_hat, reference.theta_applied, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(observed.status.eq("OK").to_numpy(), reference.passed.to_numpy(bool))


def main():
    frame = population().rename(columns={"yob_decade": "YearOfBirth", "monday": "DateDied"})
    frame["ISOweekDied"] = pd.to_datetime(frame.DateDied).dt.strftime("%G-%V")
    frame["Sex"] = "O"
    with tempfile.TemporaryDirectory(prefix="kcor-n5-") as tmp:
        root = Path(tmp)
        frame.drop(columns="t").to_excel(root / "input.xlsx", sheet_name="2021_24", index=False)
        with ProcessPoolExecutor(4) as pool:
            list(pool.map(check_window, [(root, w) for w in historical.build_windows()[0]]))
    print("All 78 scan fits match the workbook parameters and diagnostic outcomes.")


if __name__ == "__main__":
    main()

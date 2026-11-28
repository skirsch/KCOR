"""Reconstruct dose 1 versus 0 from the frozen m83 quiet-window scan inputs."""
import argparse
from datetime import date
from hashlib import file_digest
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main(edition):
    manifest = json.loads(Path(__file__).with_name("inputs.json").read_text())["files"]["current"]
    for rel in (
        "analysis/quiet_windows/scan.py", "analysis/quiet_windows/windows.csv", "code/KCOR.py",
        "test/quiet_window/code/quiet_window_scan_theta_czech_2021_24.py",
        "data/Czech/reference_without_COVID_correction-1-24-26/KCOR_CMR.xlsx",
    ):
        with (edition / rel).open("rb") as source:
            assert file_digest(source, "sha256").hexdigest() == manifest[rel], rel
    scan_dir = edition / "analysis/quiet_windows"
    sys.path.insert(0, str(scan_dir))
    import scan

    fits = pd.read_csv(scan_dir / "windows.csv")
    assert len(fits) == 78 and (fits.passed == 1).all()
    fits = fits.pivot(index=["yob_decade", "window_start_iso"], columns="dose", values="theta_applied")
    assert len(fits) == 26 and not fits.isna().any().any()
    data = scan.population()
    anchor = date.fromisocalendar(2021, 29, 1)
    endpoint = date.fromisocalendar(2024, 16, 1)
    totals = {}
    for key, group in data[data.Dose.isin([0, 1])].groupby(["yob_decade", "Dose"]):
        group = group[(group.t >= 2) & (group.monday <= endpoint)].sort_values("t")
        assert group.monday.iloc[3] == anchor and group.monday.iloc[-1] == endpoint
        assert np.all(np.diff(group.t) == 1)
        cumulative = -np.log1p(-group.Dead / group.Alive).cumsum()
        totals[key] = cumulative.iloc[[3, -1]].to_numpy()

    print("birth_decade,window_start,unanchored,anchored")
    for (decade, start), fit in fits.iterrows():
        baseline = {}
        for dose in (0, 1):
            H, theta = totals[decade, dose], fit[dose]
            baseline[dose] = H if theta < 1e-10 else np.expm1(theta * H) / theta
            np.testing.assert_allclose(baseline[dose], scan.KCOR.invert_gamma_frailty(H, theta))
        at_anchor, final = baseline[1] / baseline[0]
        print(f"{decade},{start},{final:.9f},{final / at_anchor:.9f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edition", type=Path, help="frozen current edition directory (m83)")
    main(parser.parse_args().edition.resolve())

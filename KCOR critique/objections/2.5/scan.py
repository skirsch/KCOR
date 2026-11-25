"""Replay the Czech quiet-window scan with the iteration 2 joint fitter."""
import hashlib
import io
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent.parent / "iterative review"
ROOT = REVIEW.parents[1]
OLD = ROOT / "test/quiet_window"
FIGURE = REVIEW / "revised/documentation/preprint/figures/fig_si_quiet_window_theta_scan_czech_2021_24.png"
INPUT = "63c60f1f020d0ee59c4c03e6f86d30014cca107f"
sys.path.insert(0, str(OLD / "code"))
import quiet_window_scan_theta_czech_2021_24 as historical
sys.path.insert(0, str(REVIEW / "revised/code"))
import KCOR


def population():
    raw = subprocess.check_output(["git", "cat-file", "blob", INPUT], cwd=ROOT)
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == INPUT
    data = pd.read_excel(io.BytesIO(raw), sheet_name="2021_24",
                         usecols=["YearOfBirth", "Dose", "ISOweekDied", "Alive", "Dead"])
    data["yob_decade"] = (data.YearOfBirth // 10) * 10
    data = data[data.yob_decade.isin(historical.YOB_DECADES) & data.Dose.isin([0, 1, 2])].copy()
    dates = {v: historical.iso_week_to_monday(*historical.parse_iso_week(v))
             for v in data.ISOweekDied.unique()}
    data["monday"] = data.ISOweekDied.map(dates)
    enrollment = historical.iso_week_to_monday(2021, 24)
    data = data[data.monday >= enrollment]
    groups = data.groupby(["yob_decade", "Dose", "monday"], as_index=False)[["Alive", "Dead"]].sum()
    groups["t"] = groups.monday.map(lambda d: historical.abs_week_index(d, enrollment))
    return groups[groups.Alive > 0]


def scan():
    windows, origin = historical.build_windows()
    old = pd.read_csv(OLD / "out/quiet_window_scan_theta_czech_2021_24_yob1940_50.csv")
    keys = ["yob_decade", "dose", "window_start_iso"]
    old = old.set_index(keys, verify_integrity=True)
    cfg = KCOR.get_gompertz_theta_fit_config()
    guard = KCOR.get_theta_degenerate_fit_config()
    rows = []
    for (decade, dose), group in population().groupby(["yob_decade", "Dose"]):
        group = group.sort_values("t")
        time = group.t.to_numpy(float)
        deaths = group.Dead.to_numpy(float)
        alive = group.Alive.to_numpy(float)
        t_rebased = time - KCOR.DYNAMIC_HVE_SKIP_WEEKS
        hazard = KCOR.hazard_from_mr(np.clip(deaths / (alive + KCOR.EPS), 0, .999))
        effective = np.where(t_rebased >= 0, hazard, 0.)
        historical_H = np.cumsum(-np.log1p(-np.clip(deaths / alive, 0, 1-1e-12)))
        weeks = group.monday.map(lambda d: historical.abs_week_index(d, origin)).to_numpy()
        for w in windows:
            quiet = (weeks >= w.start_abs) & (weeks < w.end_abs)
            previous = old.loc[(decade, dose, w.start_iso)]
            assert quiet.sum() == previous.n_points and w.end_iso == previous.window_end_iso
            # Match the saved historical trajectories before changing fitter and diagnostics.
            historical_fit = historical.fit_window(time[quiet], historical_H[quiet])
            np.testing.assert_allclose(historical_fit, previous[["k_hat", "theta_hat", "rmse_H", "r2_postnorm"]].to_numpy(float),
                                       rtol=1e-5, atol=1e-8)
            (k, theta), diag = KCOR.fit_theta0_gompertz(
                effective, t_rebased, quiet, cfg["k_anchor_weeks"], cfg["gamma_per_week"],
                deaths_arr=deaths, joint_seed=True)
            applied, degenerate, _ = KCOR._apply_theta_degenerate_guard(theta, guard)
            passed = bool(diag["success"] and diag["status"] == "ok" and not degenerate)
            rows.append(dict(yob_decade=decade, dose=dose, window_start_iso=w.start_iso,
                             window_end_iso=w.end_iso, midpoint=w.midpoint, quiet_weeks=int(quiet.sum()),
                             fit_weeks=diag["n_fit"], old_pass=int(previous["pass"]), passed=int(passed),
                             k=k, theta_seed=diag["theta0_init"], theta_raw=diag["theta0_raw"],
                             theta_applied=applied, status=diag["status"], success=diag["success"],
                             message=diag["message"], degenerate=degenerate, rmse=diag["rmse_hazard"]))
    results = pd.DataFrame(rows)
    assert len(results) == len(old) == 78 and not results.duplicated(keys).any()
    return results


def figure(rows):
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.7), sharex=True, sharey=True, layout="constrained")
    for (decade, dose), group in rows.groupby(["yob_decade", "dose"]):
        row = 0 if decade == 1940 else 1
        ax = axes[row, dose]
        for value, marker, color in [(1, "o", "#1d5265"), (0, "x", "#983b20")]:
            selected = group[group.passed == value]
            ax.scatter(pd.to_datetime(selected.midpoint), selected.theta_raw, marker=marker, s=21, color=color)
        ax.grid(axis="y", alpha=.23)
        ax.set_axisbelow(True)
        if row == 0:
            ax.set_title(f"Dose {dose}")
        if dose == 0:
            ax.set_ylabel(f"{decade}s births\nFrailty variance")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(labelsize=8)
    fig.supxlabel("Quiet-window midpoint", fontsize=10)
    handles = [Line2D([], [], marker=m, color=c, linestyle="none", label=l) for m,c,l in
               [("o", "#1d5265", "Diagnostic pass"), ("x", "#983b20", "Diagnostic failure")]]
    fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    rows = scan()
    summary = rows.groupby(["yob_decade", "dose"], as_index=False).agg(
        attempted=("passed", "size"), historical_passed=("old_pass", "sum"), revised_passed=("passed", "sum"))
    assert len(summary) == 6 and (summary.attempted == 13).all()
    summary["pass_rate"] = summary.revised_passed / summary.attempted
    rows.to_csv(HERE / "windows.csv", index=False)
    summary.to_csv(HERE / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    figure(rows)

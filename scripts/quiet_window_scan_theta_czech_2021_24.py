import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

THETA_EPS = 1e-8
MIN_POINTS = 30
R2_THRESHOLD = 0.995
THETA_MAX = 100.0
RMSE_PERCENTILE = 75.0

ENROLL_ISO_YEAR = 2021
ENROLL_ISO_WEEK = 24

WINDOW_START_DATE = date(2022, 4, 1)
WINDOW_END_DATE = date(2023, 4, 1)

YOB_DECADES = {1930, 1940, 1950}


@dataclass
class Window:
    start_date: date
    end_date: date
    start_iso: str
    end_iso: str
    start_abs: int
    end_abs: int
    midpoint: date


def add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def parse_iso_week(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        iso = value.isocalendar()
        return iso[0], iso[1]
    s = str(value).strip()
    match = re.search(r"(\d{4})\D*W?(\d{1,2})", s, re.IGNORECASE)
    if not match:
        return None
    year = int(match.group(1))
    week = int(match.group(2))
    if week < 1 or week > 53:
        return None
    return year, week


def iso_week_to_monday(year: int, week: int) -> date:
    return date.fromisocalendar(year, week, 1)


def abs_week_index(week_monday: date, reference_monday: date) -> int:
    return (week_monday - reference_monday).days // 7


def model_cumhaz(t, k, theta):
    if theta < THETA_EPS:
        return k * t
    return np.log1p(theta * k * t) / theta


def fit_window(t, H_obs):
    if len(t) < 2:
        return np.nan, np.nan, np.nan, np.nan
    t = np.asarray(t, dtype=float)
    H_obs = np.asarray(H_obs, dtype=float)
    mask = np.isfinite(t) & np.isfinite(H_obs)
    t = t[mask]
    H_obs = H_obs[mask]
    if len(t) < 2:
        return np.nan, np.nan, np.nan, np.nan
    k0 = max(1e-6, float(H_obs[-1]) / max(t))
    theta0 = 0.1

    def residuals(params):
        k, theta = params
        return H_obs - model_cumhaz(t, k, theta)

    result = least_squares(
        residuals,
        x0=np.array([k0, theta0]),
        bounds=([1e-12, 0.0], [np.inf, np.inf]),
        method="trf",
    )
    k_hat, theta_hat = result.x
    residual = residuals(result.x)
    rmse_h = math.sqrt(np.mean(residual**2))
    if theta_hat < THETA_EPS:
        H_tilde = H_obs
    else:
        H_tilde = (np.expm1(theta_hat * H_obs)) / theta_hat
    r2 = compute_r2(t, H_tilde)
    return k_hat, theta_hat, rmse_h, r2


def compute_r2(x, y):
    if len(x) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot <= 0:
        return np.nan
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    return 1.0 - ss_res / ss_tot


def build_windows():
    windows = []
    ref_monday = date(1900, 1, 1)
    current = WINDOW_START_DATE
    while current <= WINDOW_END_DATE:
        end_date = add_months(current, 12)
        start_iso = current.isocalendar()
        end_iso = end_date.isocalendar()
        start_monday = iso_week_to_monday(start_iso[0], start_iso[1])
        end_monday = iso_week_to_monday(end_iso[0], end_iso[1])
        windows.append(
            Window(
                start_date=current,
                end_date=end_date,
                start_iso=f"{start_iso[0]}-{start_iso[1]:02d}",
                end_iso=f"{end_iso[0]}-{end_iso[1]:02d}",
                start_abs=abs_week_index(start_monday, ref_monday),
                end_abs=abs_week_index(end_monday, ref_monday),
                midpoint=current + (end_date - current) / 2,
            )
        )
        current = add_months(current, 1)
    return windows, ref_monday


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "Czech" / "KCOR_CMR.xlsx"
    out_dir = repo_root / "test" / "quiet_window" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_path, sheet_name="2021_24")

    col_map = {
        "YearOfBirth": ["YearOfBirth", "YoB", "YOB"],
        "Dose": ["Dose"],
        "ISOweekDied": ["ISOweekDied", "ISOWeekDied", "ISOweek", "ISOWeek"],
        "Dead": ["Dead", "Deaths"],
        "Alive": ["Alive"],
    }
    columns = {}
    for key, candidates in col_map.items():
        for name in candidates:
            if name in df.columns:
                columns[key] = name
                break
        if key not in columns:
            raise ValueError(f"Missing required column: {key}")

    df = df.rename(columns={columns[k]: k for k in columns})
    df["YearOfBirth"] = df["YearOfBirth"].astype(int)
    df["yob_decade"] = (df["YearOfBirth"] // 10) * 10
    df = df[df["yob_decade"].isin(YOB_DECADES)].copy()
    df = df[df["Dose"].isin([0, 1, 2])].copy()

    parsed = df["ISOweekDied"].apply(parse_iso_week)
    df["iso_year"] = parsed.apply(lambda x: x[0] if x else np.nan)
    df["iso_week"] = parsed.apply(lambda x: x[1] if x else np.nan)
    df = df.dropna(subset=["iso_year", "iso_week"])
    df["iso_year"] = df["iso_year"].astype(int)
    df["iso_week"] = df["iso_week"].astype(int)

    windows, ref_monday = build_windows()
    enroll_monday = iso_week_to_monday(ENROLL_ISO_YEAR, ENROLL_ISO_WEEK)

    df["week_monday"] = df.apply(
        lambda row: iso_week_to_monday(row["iso_year"], row["iso_week"]), axis=1
    )
    df["week_abs"] = df["week_monday"].apply(lambda d: abs_week_index(d, ref_monday))
    df["t_week"] = df["week_monday"].apply(lambda d: abs_week_index(d, enroll_monday))
    df = df[df["t_week"] >= 0].copy()

    df = (
        df.groupby(["yob_decade", "Dose", "iso_year", "iso_week", "week_abs", "t_week"], as_index=False)
        .agg({"Dead": "sum", "Alive": "sum"})
    )

    df = df[df["Alive"] > 0].copy()
    df["MR"] = df["Dead"] / df["Alive"]
    df["MR"] = df["MR"].clip(lower=0.0, upper=1.0 - 1e-12)
    df["hazard"] = -np.log1p(-df["MR"])

    results = []
    for (yob_decade, dose), group in df.groupby(["yob_decade", "Dose"]):
        group = group.sort_values("t_week")
        group["H_obs"] = group["hazard"].cumsum()
        for window in windows:
            w = group[(group["week_abs"] >= window.start_abs) & (group["week_abs"] < window.end_abs)]
            n_points = len(w)
            if n_points >= 2:
                k_hat, theta_hat, rmse_h, r2 = fit_window(w["t_week"].to_numpy(), w["H_obs"].to_numpy())
            else:
                k_hat, theta_hat, rmse_h, r2 = (np.nan, np.nan, np.nan, np.nan)
            results.append(
                {
                    "window_start_date": window.start_date.isoformat(),
                    "window_end_date": window.end_date.isoformat(),
                    "window_start_iso": window.start_iso,
                    "window_end_iso": window.end_iso,
                    "window_midpoint_date": window.midpoint.isoformat(),
                    "yob_decade": int(yob_decade),
                    "dose": int(dose),
                    "k_hat": k_hat,
                    "theta_hat": theta_hat,
                    "rmse_H": rmse_h,
                    "r2_postnorm": r2,
                    "n_points": int(n_points),
                }
            )

    results_df = pd.DataFrame(results)
    valid_rmse = results_df.loc[results_df["n_points"] >= MIN_POINTS, "rmse_H"].dropna()
    rmse_thresh = np.nan if valid_rmse.empty else np.nanpercentile(valid_rmse, RMSE_PERCENTILE)

    def classify_pass(row):
        if row["n_points"] < MIN_POINTS:
            return 0
        if not np.isfinite(row["rmse_H"]) or not np.isfinite(row["r2_postnorm"]):
            return 0
        if row["rmse_H"] > rmse_thresh:
            return 0
        if row["r2_postnorm"] < R2_THRESHOLD:
            return 0
        if row["theta_hat"] >= THETA_MAX:
            return 0
        return 1

    results_df["pass"] = results_df.apply(classify_pass, axis=1)

    results_path = out_dir / "quiet_window_scan_theta_czech_2021_24_yob1930_40_50.csv"
    results_df.to_csv(results_path, index=False)

    summary = (
        results_df[results_df["pass"] == 1]
        .groupby(["dose", "yob_decade"], as_index=False)
        .agg(
            pass_rate=("pass", "mean"),
            theta_mean=("theta_hat", "mean"),
            theta_sd=("theta_hat", "std"),
            theta_count=("theta_hat", "count"),
        )
    )
    summary["theta_cv"] = summary["theta_sd"] / summary["theta_mean"]

    summary_path = out_dir / "quiet_window_scan_theta_summary.csv"
    summary.to_csv(summary_path, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = repo_root / "figures" / "si"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "fig_quiet_window_theta_scan_czech_2021_24.png"

        plot_df = results_df.copy()
        plot_df["window_midpoint_date"] = pd.to_datetime(plot_df["window_midpoint_date"])

        doses = sorted(plot_df["dose"].unique())
        markers = {1930: "o", 1940: "s", 1950: "^"}
        edge_color = "#222222"

        fig, axes = plt.subplots(len(doses), 1, figsize=(10, 3 * len(doses)), sharex=True)
        if len(doses) == 1:
            axes = [axes]

        for ax, dose in zip(axes, doses):
            dose_df = plot_df[plot_df["dose"] == dose]
            for decade in sorted(dose_df["yob_decade"].unique()):
                decade_df = dose_df[dose_df["yob_decade"] == decade]
                for pass_value in [0, 1]:
                    subset = decade_df[decade_df["pass"] == pass_value]
                    ax.scatter(
                        subset["window_midpoint_date"],
                        subset["theta_hat"],
                        label=f"{decade}s",
                        alpha=0.8,
                        marker=markers.get(decade, "o"),
                        facecolors=edge_color if pass_value == 1 else "none",
                        edgecolors=edge_color,
                        linewidths=1.0,
                    )
            ax.set_ylabel("theta_hat")
            ax.set_title(f"Dose {dose}")

        axes[-1].set_xlabel("Window midpoint date")
        handles, labels = axes[0].get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle
        if unique:
            fig.legend(
                list(unique.values()),
                list(unique.keys()),
                loc="center right",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
            )
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"Wrote figure to {fig_path}")
    except Exception as exc:
        print(f"Skipping figure generation: {exc}")

    print(f"Wrote detailed results to {results_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

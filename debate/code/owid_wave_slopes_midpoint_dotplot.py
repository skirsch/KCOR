"""
Dot plot: one point per detected wave from owid_all_locations_wave_slopes.csv.
x = calendar midpoint between week_start and week_end; y = OLS slope.
Only rows with r2_fit >= R2_MIN (default 0.99).

Adds an OLS trend line: with the default log y-axis, the line is linear in
log10(slope) vs time (matplotlib date numbers); legend shows that R² and the
fitted end/start ratio along the trend. Default log y-limits are 1–400 so the
full scatter (≈1–340 d/m/wk) is visible, not auto-cropped to the trend.

Writes a PNG under debate/figures/ by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "debate" / "data" / "owid_all_locations_wave_slopes.csv"
DEFAULT_OUTPUT = REPO_ROOT / "debate" / "figures" / "owid_wave_slopes_midpoint_r2ge99_dotplot.png"


def ols_r2_line(
    x_num: np.ndarray, y: np.ndarray, log_y: bool
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Return (poly_coeffs_high_to_low, r_squared, y_pred_on_points).
    If log_y: fit log10(y) = p[0]*x + p[1]; r² is for log10(y); y_pred = 10**polyval.
    Else: fit y = p[0]*x + p[1]; r² for y.
    """
    if log_y:
        ly = np.log10(y)
        p = np.polyfit(x_num, ly, 1)
        pred_l = np.polyval(p, x_num)
        ss_res = float(np.sum((ly - pred_l) ** 2))
        ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return p, r2, 10.0**pred_l
    p = np.polyfit(x_num, y, 1)
    pred = np.polyval(p, x_num)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return p, r2, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Wave slopes CSV")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    ap.add_argument(
        "--r2-min",
        type=float,
        default=0.99,
        help="Keep segments with r2_fit >= this (default: 0.99)",
    )
    ap.add_argument("--linear-y", action="store_true", help="Use linear y-axis (default: log)")
    ap.add_argument(
        "--ylim-log",
        type=float,
        nargs=2,
        default=(1.0, 400.0),
        metavar=("YMIN", "YMAX"),
        help="Y limits when using log scale (default: 1 400)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["week_start", "week_end"])
    df = df.loc[np.isfinite(df["r2_fit"]) & np.isfinite(df["slope_ols_deaths_per_million_per_week"])].copy()
    df = df.loc[df["r2_fit"] >= args.r2_min].copy()
    if df.empty:
        raise SystemExit(f"No rows with r2_fit >= {args.r2_min} in {args.input}")

    t0 = df["week_start"].values.astype("datetime64[ns]")
    t1 = df["week_end"].values.astype("datetime64[ns]")
    mid = pd.to_datetime((t0.astype(np.int64) + t1.astype(np.int64)) / 2.0)
    df["segment_midpoint"] = mid

    y = df["slope_ols_deaths_per_million_per_week"].astype(float).values
    if (y <= 0).any():
        raise SystemExit("Non-positive slopes cannot be shown on log y; fix data or use --linear-y")

    mid = df["segment_midpoint"]
    x_num = mdates.date2num(mid)

    use_log = not args.linear_y
    coef, r2_line, _pred = ols_r2_line(x_num, y, log_y=use_log)

    x_line_num = np.linspace(float(x_num.min()), float(x_num.max()), 200)
    if use_log:
        y_line = 10.0 ** np.polyval(coef, x_line_num)
    else:
        y_line = np.polyval(coef, x_line_num)

    y_fit_start = float(y_line[0])
    y_fit_end = float(y_line[-1])
    fit_ratio = y_fit_end / y_fit_start if y_fit_start > 0 else float("nan")

    if use_log:
        trend_legend = (
            f"log10(slope) trend R² = {r2_line:.3f}, fitted end/start = {fit_ratio:.2f}x"
        )
    else:
        trend_legend = f"slope vs time R² = {r2_line:.3f}, fitted end/start = {fit_ratio:.2f}x"

    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.scatter(
        mid,
        y,
        s=14,
        alpha=0.35,
        c="#1f77b4",
        edgecolors="none",
        zorder=3,
        label="Auto-identified wave segments",
    )
    ax.plot(
        mdates.num2date(x_line_num),
        y_line,
        color="#c0392b",
        linewidth=2.2,
        zorder=4,
        label=trend_legend,
    )
    ax.set_xlabel("Segment calendar midpoint (mean of week_start and week_end)")
    ax.set_ylabel("OLS slope (cumulative deaths per million per week)")
    ax.set_title(
        f"All OWID countries · All auto-identified wave segments · R² ≥ {args.r2_min:g}\n"
        f"Slope vs segment calendar midpoint (n = {len(df)})",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, alpha=0.28, zorder=0)
    if use_log:
        ax.set_yscale("log")
        lo, hi = float(args.ylim_log[0]), float(args.ylim_log[1])
        if lo <= 0 or hi <= 0:
            raise SystemExit("--ylim-log values must be positive for log scale")
        ax.set_ylim(lo, hi)

    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    summary = (
        f"trend R²={r2_line:.4f}, fitted slope {y_fit_start:.2f} → {y_fit_end:.2f} d/m/wk "
        f"(end/start={fit_ratio:.2f}x)"
    )
    print(f"Wrote {args.output} ({len(df)} points)  |  {summary}")


if __name__ == "__main__":
    main()

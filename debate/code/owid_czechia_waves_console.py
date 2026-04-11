"""
Czechia-only: detect COVID mortality waves from weekly new deaths (flat → active → flat),
trim middle 80% of each wave, regress cumulative vs week index, print slopes.

Data: debate/data/OWID_total_deaths_per_million.csv
Cutoff: 2022-04-01 (inclusive daily rows; weekly stamps may end that week).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "debate" / "data" / "OWID_total_deaths_per_million.csv"

# Hysteresis on smoothed weekly new deaths (per million): enter wave when above hi,
# exit after `settle` consecutive weeks below lo (near-flat cumulative).
HI_THR = 12.0
# "Near flat" on cumulative while weekly deaths are still >8 — use ~30 so wave 1 ends ~May 2021
# (closer to the cumulative plateau than lo=8, which waits until June).
LO_THR = 28.0
SETTLE_WEEKS = 4
SMOOTH_WEEKS = 4


def ols_slope_intercept_r2(y: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = np.sum((x - xm) * (y - ym)) / sxx
    intercept = ym - slope * xm
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - ym) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def trim_middle_80(start: int, end: int) -> tuple[int, int]:
    """Inclusive indices; remove 10% from each end (ceil)."""
    n = end - start + 1
    if n < 5:
        return start, end
    t = max(1, math.ceil(0.10 * n))
    a, b = start + t, end - t
    if b < a:
        return start, end
    return a, b


def hysteresis_runs(
    values: np.ndarray,
    hi: float,
    lo: float,
    settle: int,
) -> list[tuple[int, int]]:
    """Return inclusive (start, end) indices of each active run."""
    active = False
    out: list[tuple[int, int]] = []
    start = 0
    consec_low = 0
    for i, v in enumerate(values):
        if not active and v > hi:
            active = True
            start = i
            consec_low = 0
        elif active:
            if v < lo:
                consec_low += 1
            else:
                consec_low = 0
            if consec_low >= settle:
                end = i - settle
                if end >= start:
                    out.append((start, end))
                active = False
    if active:
        out.append((start, len(values) - 1))
    return out


def split_run_if_secondary_bump(
    sm: np.ndarray,
    run_start: int,
    run_end: int,
    min_rebound: float = 8.0,
    trough_frac: float = 0.38,
) -> list[tuple[int, int]]:
    """
    If within [run_start, run_end] there is a clear dip then rebound (third wave on cumulative),
    split into two runs. Czechia: winter 21-22 dip ~late Jan then Feb bump.
    """
    seg = sm[run_start : run_end + 1]
    if len(seg) < 12:
        return [(run_start, run_end)]
    inner = seg[4:-4]
    if inner.size == 0:
        return [(run_start, run_end)]
    rel_min = int(np.argmin(inner)) + 4
    abs_min = run_start + rel_min
    depth = float(seg[rel_min])
    peak_seg = float(np.max(seg))
    after_peak = float(np.max(seg[rel_min:])) if rel_min < len(seg) else 0.0
    # Trough must be well below the segment peak (near-flat cumulative), then mortality rises again.
    if peak_seg <= 0 or depth > trough_frac * peak_seg:
        return [(run_start, run_end)]
    rebound = after_peak - depth
    if rebound < min_rebound:
        return [(run_start, run_end)]
    if abs_min <= run_start or abs_min + 1 > run_end:
        return [(run_start, run_end)]
    # Trough week is last week of wave 2; rebound starts the following week.
    return [(run_start, abs_min), (abs_min + 1, run_end)]


def main() -> None:
    path = DEFAULT_CSV
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.loc[df["date"] <= "2022-04-01"].copy()

    daily = pd.to_numeric(df["Czechia"], errors="coerce").ffill().bfill()
    daily.index = pd.DatetimeIndex(df["date"])
    w_cum = daily.resample("W-SUN").last()
    w_new = w_cum.diff().fillna(0.0).clip(lower=0.0)
    sm = w_new.rolling(SMOOTH_WEEKS, min_periods=1).mean()
    arr = sm.values.astype(float)
    idx = sm.index

    raw_runs = hysteresis_runs(arr, HI_THR, LO_THR, SETTLE_WEEKS)
    runs: list[tuple[int, int]] = []
    for a, b in raw_runs:
        runs.extend(split_run_if_secondary_bump(arr, a, b))

    print("Czechia — OWID cumulative COVID deaths per million")
    print(f"Daily data through {df['date'].max().date()} (weekly series ends ~{idx[-1].date()})")
    print(f"Wave rule: smoothed({SMOOTH_WEEKS}w) weekly new > {HI_THR} → active; "
          f"end after {SETTLE_WEEKS} consecutive weeks < {LO_THR}.")
    print(f"Long winter 2021–22 run split if deep trough + rebound inside segment.\n")

    for k, (a, b) in enumerate(runs, start=1):
        ta, tb = trim_middle_80(a, b)
        y = w_cum.iloc[ta : tb + 1].to_numpy(dtype=float)
        slope, intercept, r2 = ols_slope_intercept_r2(y)
        n_all = b - a + 1
        n_mid = tb - ta + 1
        print(f"Wave {k}")
        print(f"  full segment (week ending): {idx[a].date()} .. {idx[b].date()}  ({n_all} weeks)")
        print(f"  middle 80% for regression: {idx[ta].date()} .. {idx[tb].date()}  ({n_mid} weeks)")
        print(f"  slope (deaths per million per week on cumulative): {slope:.4f}")
        print(f"  R^2 (linear fit): {r2:.4f}")
        print(f"  Δ cumulative over middle 80%: {y[-1] - y[0]:.2f}")
        print()


if __name__ == "__main__":
    main()

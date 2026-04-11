"""
OWID cumulative COVID deaths (per million): detect wave intervals from weekly
mortality dynamics and fit OLS slope of cumulative vs week index.

Default paths:
  --input  debate/data/owid_source/OWID_total_deaths_per_million.csv
  --output debate/data/owid_slope/owid_wave_cumulative_slopes.csv

Alpha-era winter surge: searched in 2020-10-01 .. 2021-05-31 (B.1.1.7 dominant
in many Northern countries in early 2021; same window captures that mortality
pulse even when labeling is imperfect).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = parent of debate/
REPO_ROOT = Path(__file__).resolve().parents[2]
OWID_SOURCE_DIR = REPO_ROOT / "debate" / "data" / "owid_source"
OWID_SLOPE_DATA_DIR = REPO_ROOT / "debate" / "data" / "owid_slope"
DEFAULT_CSV = OWID_SOURCE_DIR / "OWID_total_deaths_per_million.csv"
DEFAULT_OUT = OWID_SLOPE_DATA_DIR / "owid_wave_cumulative_slopes.csv"

AGGREGATE_COLUMNS = {
    "World",
    "Asia",
    "Africa",
    "Europe",
    "North America",
    "South America",
    "Oceania",
    "European Union (27)",
    "High-income countries",
    "Low-income countries",
    "Lower-middle-income countries",
    "Upper-middle-income countries",
}


def load_wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def daily_cumulative_series(df: pd.DataFrame, country: str) -> pd.Series:
    s = pd.to_numeric(df[country], errors="coerce")
    s = s.ffill().bfill()
    s.index = pd.DatetimeIndex(df["date"])
    return s.sort_index()


def weekly_cumulative_and_new(daily_cum: pd.Series) -> tuple[pd.Series, pd.Series]:
    """End-of-week (Sunday) cumulative; weekly new = diff (negative diffs clipped)."""
    w = daily_cum.resample("W-SUN").last()
    new = w.diff()
    new = new.fillna(0.0).clip(lower=0.0)
    return w, new


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    if len(a) == 0:
        return a
    x = pd.Series(a)
    return x.rolling(window, min_periods=1).mean().to_numpy()


def detect_wave_segment(
    week_dates: np.ndarray,
    weekly_new: np.ndarray,
    search_start: np.datetime64,
    search_end: np.datetime64,
    smooth_weeks: int = 3,
    rise_frac: float = 0.20,
    fall_frac: float = 0.22,
    min_weeks: int = 5,
    min_peak_smoothed: float = 0.08,
    settle_weeks: int = 3,
) -> tuple[int, int, int] | None:
    """
    Return inclusive (start_idx, end_idx, peak_idx) into week_* arrays, or None.

    peak_idx: argmax of smoothed weekly new inside the search window.
    start: first week at or after window start where smoothed >= rise_frac * peak.
    end: last week before smoothed stays below fall_frac * peak for settle_weeks.
    """
    n = len(week_dates)
    if n == 0:
        return None
    mask = (week_dates >= search_start) & (week_dates <= search_end)
    idxs = np.flatnonzero(mask)
    if idxs.size < min_weeks:
        return None

    sub_new = weekly_new[idxs].astype(float)
    sm = rolling_mean(sub_new, smooth_weeks)
    peak_rel = int(np.nanargmax(sm))
    peak_val = float(sm[peak_rel])
    if not np.isfinite(peak_val) or peak_val < min_peak_smoothed:
        return None

    rise_t = rise_frac * peak_val
    fall_t = fall_frac * peak_val

    start_rel = 0
    for i in range(0, peak_rel + 1):
        if sm[i] >= rise_t:
            start_rel = i
            break

    end_rel = len(sm) - 1
    for j in range(peak_rel, len(sm)):
        if j + settle_weeks <= len(sm):
            if np.all(sm[j : j + settle_weeks] < fall_t):
                end_rel = max(start_rel, j - 1)
                break
    if end_rel < start_rel:
        return None
    if end_rel - start_rel + 1 < min_weeks:
        return None

    peak_idx = int(idxs[peak_rel])
    start_idx = int(idxs[start_rel])
    end_idx = int(idxs[end_rel])
    return start_idx, end_idx, peak_idx


def ols_slope_intercept_r2(y: np.ndarray) -> tuple[float, float, float]:
    """y vs x = 0..n-1; returns slope, intercept, R^2."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def country_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "date"]
    return [c for c in cols if c not in AGGREGATE_COLUMNS]


def process_country(
    df: pd.DataFrame,
    country: str,
    wave_id: str,
    search_start: str,
    search_end: str,
) -> dict:
    daily_cum = daily_cumulative_series(df, country)
    w_cum, w_new = weekly_cumulative_and_new(daily_cum)
    week_dates = w_cum.index.to_numpy()
    weekly_new = w_new.to_numpy()

    seg = detect_wave_segment(
        week_dates,
        weekly_new,
        np.datetime64(search_start),
        np.datetime64(search_end),
    )
    if seg is None:
        return {
            "country": country,
            "wave": wave_id,
            "search_start": search_start,
            "search_end": search_end,
            "segment_start": pd.NA,
            "segment_end": pd.NA,
            "peak_week": pd.NA,
            "n_weeks": pd.NA,
            "slope_deaths_per_million_per_week": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "delta_cum_over_segment": np.nan,
            "status": "no_segment",
        }

    start_idx, end_idx, peak_idx = seg
    y = w_cum.iloc[start_idx : end_idx + 1].to_numpy(dtype=float)
    slope, intercept, r2 = ols_slope_intercept_r2(y)
    delta_cum = float(y[-1] - y[0]) if y.size else float("nan")

    return {
        "country": country,
        "wave": wave_id,
        "search_start": search_start,
        "search_end": search_end,
        "segment_start": str(w_cum.index[start_idx].date()),
        "segment_end": str(w_cum.index[end_idx].date()),
        "peak_week": str(w_cum.index[peak_idx].date()),
        "n_weeks": int(end_idx - start_idx + 1),
        "slope_deaths_per_million_per_week": slope,
        "intercept": intercept,
        "r2": r2,
        "delta_cum_over_segment": delta_cum,
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV,
        help="Wide OWID deaths CSV (default: debate/data/owid_source/OWID_total_deaths_per_million.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output slopes CSV (default: debate/data/owid_slope/owid_wave_cumulative_slopes.csv)",
    )
    args = parser.parse_args()

    # (wave_id, search_start, search_end) — wide windows; edges refined per country.
    waves = [
        ("first_wave_2020", "2020-02-15", "2020-10-15"),
        ("alpha_winter_2020_21", "2020-10-01", "2021-05-31"),
        ("delta_2021", "2021-05-01", "2021-11-30"),
        ("omicron_ba1_2021_22", "2021-11-15", "2022-04-30"),
    ]

    df = load_wide(args.input)
    countries = country_columns(df)
    rows: list[dict] = []
    for country in countries:
        for wave_id, ws, we in waves:
            rows.append(process_country(df, country, wave_id, ws, we))

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    ok = (out["status"] == "ok").sum()
    print(f"Wrote {args.output} ({ok} ok rows of {len(out)} total).")


if __name__ == "__main__":
    main()

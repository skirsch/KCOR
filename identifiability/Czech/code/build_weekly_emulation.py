"""
Weekly incident booster identifiability emulation (Czech 2021).

Implements the design in documentation/preprint/identifiability.md.

Key semantics:
- ISO-week strings are interpreted as the Monday of that ISO week.
- Cohort membership and risk sets are defined at week-start.
- Transition censoring is rule B: censor starting the week AFTER the transition week.
- Hazards use discrete-time transform: h(t) = -ln(1 - dead/alive).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Force a headless Matplotlib backend (WSL/servers often have no display).
# This must run BEFORE importing matplotlib.pyplot anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    # If matplotlib isn't installed, plotting will fail later with an import error.
    pass


def _find_repo_root(start: Path) -> Optional[Path]:
    """
    Find the repo root by walking upwards until we find code/mfg_codes.py.

    This keeps imports working regardless of where this script lives (or current cwd).
    """
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "code" / "mfg_codes.py").exists():
            return parent
    return None


# Ensure we can import repo-local modules (notably code/mfg_codes.py) regardless of cwd.
_REPO_ROOT = _find_repo_root(Path(__file__))
if _REPO_ROOT is not None:
    sys.path.insert(0, str(_REPO_ROOT / "code"))


DOSE_DATE_COLS = [
    "Date_FirstDose",
    "Date_SecondDose",
    "Date_ThirdDose",
    "Date_FourthDose",
    "Date_FifthDose",
    "Date_SixthDose",
]

INFECTION_DATE_COLS = [
    "DateOfPositiveTest",
]

VCODE_COLS = [
    "VaccineCode_FirstDose",
    "VaccineCode_SecondDose",
    "VaccineCode_ThirdDose",
    "VaccineCode_FourthDose",
]


def _read_csv_flex(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read CSV with robust delimiter/encoding handling. Returns dtype=str."""
    # Fast path
    try:
        # low_memory=False ensures pandas parses in a single pass (more RAM, fewer dtype surprises).
        return pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8", nrows=max_rows)
    except Exception:
        pass
    for enc in ("utf-8-sig", None, "latin1"):
        attempts = (
            {"sep": ","},
            {"sep": ";", "engine": "python"},
            {"sep": "\t", "engine": "python"},
            {"sep": None, "engine": "python"},  # sniff
        )
        for opts in attempts:
            try:
                # Always parse in one pass (we have the RAM and it reduces dtype corner cases).
                common_kwargs = {"dtype": str, "encoding": enc, "nrows": max_rows, "low_memory": False}
                df = pd.read_csv(path, **opts, **common_kwargs)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(path, dtype=str, engine="python", sep=None, nrows=max_rows, low_memory=False)


def iso_week_str_to_monday_ts(iso_week_str: pd.Series) -> pd.Series:
    """Convert series of ISO week strings YYYY-WW to pandas Timestamp for Monday of that week."""
    # Expect strings like "2021-24"; tolerate junk by errors='coerce'
    return pd.to_datetime(iso_week_str.astype(str) + "-1", format="%G-%V-%u", errors="coerce")


def parse_dates_inplace(df: pd.DataFrame) -> None:
    """Parse relevant ISO-week columns in-place into pandas Timestamps (Monday of week)."""
    for col in DOSE_DATE_COLS:
        if col in df.columns:
            df[col] = iso_week_str_to_monday_ts(df[col])
        else:
            df[col] = pd.NaT

    for col in INFECTION_DATE_COLS:
        if col in df.columns:
            df[col] = iso_week_str_to_monday_ts(df[col])
        else:
            df[col] = pd.NaT

    # DateOfDeath is ISO-week; strip any non [0-9-] first (mirrors KCOR_CMR.py)
    if "DateOfDeath" in df.columns:
        s = df["DateOfDeath"].astype(str).str.replace(r"[^0-9-]", "", regex=True)
        df["DateOfDeath"] = pd.to_datetime(s + "-1", format="%G-%V-%u", errors="coerce")
    else:
        df["DateOfDeath"] = pd.NaT


def parse_birth_year(df: pd.DataFrame) -> None:
    """Parse YearOfBirth field into integer birth_year (or -1 if missing)."""
    if "YearOfBirth" not in df.columns:
        df["birth_year"] = -1
        return
    by = df["YearOfBirth"].astype(str).str.extract(r"(\d{4})")[0]
    by_num = pd.to_numeric(by, errors="coerce")
    df["birth_year"] = by_num.fillna(-1).astype(int)


def filter_infection_single(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Infection <= 1; treat missing as 0."""
    if "Infection" not in df.columns:
        return df
    inf = pd.to_numeric(df["Infection"].fillna("0"), errors="coerce").fillna(0).astype(int)
    return df[inf <= 1].copy()


def mask_not_recent_infection(df: pd.DataFrame, *, E: pd.Timestamp, exclude_recent_weeks: int) -> pd.Series:
    """True if no positive test within last exclude_recent_weeks before enrollment E."""
    if exclude_recent_weeks <= 0 or "DateOfPositiveTest" not in df.columns:
        return pd.Series(True, index=df.index)
    pos = df["DateOfPositiveTest"]
    cutoff = E - pd.Timedelta(days=7 * int(exclude_recent_weeks))
    # Exclude if cutoff <= pos < E (within lookback window)
    return pos.isna() | (pos < cutoff) | (pos >= E)


def mask_tsd2_range(df: pd.DataFrame, *, E: pd.Timestamp, min_days: Optional[int], max_days: Optional[int]) -> pd.Series:
    """Restrict to dose2-known individuals with time-since-dose2 in [min_days, max_days)."""
    if min_days is None and max_days is None:
        return pd.Series(True, index=df.index)
    d2 = df["Date_SecondDose"]
    ok = d2.notna()
    tsd2 = (E - d2).dt.days
    if min_days is not None:
        ok = ok & (tsd2 >= int(min_days))
    if max_days is not None:
        ok = ok & (tsd2 < int(max_days))
    return ok.fillna(False)


def mask_eligible_dose2(
    df: pd.DataFrame,
    *,
    E: pd.Timestamp,
    tsd2_min_days: int,
    tsd2_max_days: int,
) -> pd.Series:
    """
    Eligible dose2 risk set at enrollment E:
    - dose2 at E (Date_SecondDose <= E)
    - not dose3 before E (Date_ThirdDose is NaT or >= E)
    - tsd2 in [tsd2_min_days, tsd2_max_days)
    """
    d2 = df["Date_SecondDose"]
    d3 = df["Date_ThirdDose"]
    dose2_at_E = d2.notna() & (d2 <= E) & (d3.isna() | (d3 >= E))
    keep_tsd2 = mask_tsd2_range(df, E=E, min_days=int(tsd2_min_days), max_days=int(tsd2_max_days))
    return (dose2_at_E & keep_tsd2).fillna(False)


def mask_tt_treated_recent_dose3(
    df: pd.DataFrame,
    *,
    E: pd.Timestamp,
    treated_window_weeks: int,
    tsd2_min_days: int,
    tsd2_max_days: int,
) -> pd.Series:
    """
    Target-trial treated group at enrollment E:
    - incident dose3 in [E-7*treated_window_weeks, E)
    - has dose2 date and tsd2 in [tsd2_min_days, tsd2_max_days) at E
    - excludes dose4 before enrollment (Date_FourthDose < E)
    """
    w = int(treated_window_weeks)
    if w <= 0:
        return pd.Series(False, index=df.index)
    start = E - pd.Timedelta(days=7 * w)
    d2 = df["Date_SecondDose"]
    d3 = df["Date_ThirdDose"]
    d4 = df["Date_FourthDose"] if "Date_FourthDose" in df.columns else pd.Series(pd.NaT, index=df.index)
    incident3 = d3.notna() & (d3 >= start) & (d3 < E) & (d4.isna() | (d4 >= E))
    dose2_at_E = d2.notna() & (d2 <= E)
    keep_tsd2 = mask_tsd2_range(df, E=E, min_days=int(tsd2_min_days), max_days=int(tsd2_max_days))
    return (incident3 & dose2_at_E & keep_tsd2).fillna(False)


def mask_future_booster_k(
    df: pd.DataFrame,
    *,
    E: pd.Timestamp,
    k: int,
    tsd2_min_days: int,
    tsd2_max_days: int,
) -> pd.Series:
    """
    Lead/lag group: will receive dose3 in [E+7k, E+7(k+1)), and is dose2 at E with tsd2 in window.
    Treated as dose2 at baseline and censored at dose3 using Rule B (via CohortSpec(dose2)).
    """
    kk = int(k)
    if kk < 0:
        return pd.Series(False, index=df.index)
    d3 = df["Date_ThirdDose"]
    start = E + pd.Timedelta(days=7 * kk)
    end = E + pd.Timedelta(days=7 * (kk + 1))
    in_window = d3.notna() & (d3 >= start) & (d3 < end)
    eligible2 = mask_eligible_dose2(df, E=E, tsd2_min_days=int(tsd2_min_days), tsd2_max_days=int(tsd2_max_days))
    return (eligible2 & in_window).fillna(False)


def _int_week_since(E: pd.Timestamp, when: pd.Series) -> np.ndarray:
    """Integer weeks since E for Monday-aligned timestamps; NaT -> -1."""
    out = np.full(len(when), -1, dtype=int)
    ok = when.notna()
    if ok.any():
        out[ok.to_numpy()] = ((when[ok] - E).dt.days // 7).astype(int).to_numpy()
    return out


def _perm_band_for_enrollment(
    df: pd.DataFrame,
    *,
    E: pd.Timestamp,
    eligible_mask: pd.Series,
    n_treated: int,
    followup_weeks: int,
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Permutation placebo within the eligible dose2 risk set at enrollment E.

    Returns:
    - perm_df: rows for (enrollment_date, t) with quantiles q05/q50/q95 for HR_perm(t)
    - q05/q50/q95 arrays (len=followup_weeks)
    """
    reps = int(reps)
    followup_weeks = int(followup_weeks)
    n_treated = int(n_treated)

    pool_idx = np.flatnonzero(eligible_mask.to_numpy())
    pool_n = int(pool_idx.size)
    if reps <= 0 or n_treated <= 0 or pool_n <= 1 or n_treated >= pool_n:
        q = np.full(followup_weeks, np.nan, dtype=float)
        perm_df = pd.DataFrame(
            {
                "enrollment_date": [E.date().isoformat()] * followup_weeks,
                "t": list(range(followup_weeks)),
                "perm_q05": q,
                "perm_q50": q,
                "perm_q95": q,
                "perm_reps": reps,
                "perm_N": n_treated,
                "eligible_size": pool_n,
            }
        )
        return perm_df, q, q, q

    sub = df.loc[eligible_mask, ["DateOfDeath", "Date_ThirdDose"]].copy()
    death_week = _int_week_since(E, sub["DateOfDeath"])
    trans_week = _int_week_since(E, sub["Date_ThirdDose"])

    # Censor week-start: week after death or transition; -1 => "never" (cap at followup+1)
    big = followup_weeks + 1
    death_censor = np.where(death_week >= 0, death_week + 1, big)
    trans_censor = np.where(trans_week >= 0, trans_week + 1, big)
    censor_week = np.minimum(death_censor, trans_censor).astype(int)

    # Total histograms for the entire eligible pool (for fast "rest = total - pseudo")
    total_censor_hist = np.bincount(np.clip(censor_week, 0, big), minlength=big + 1)
    # Deaths are only counted if the individual is still at-risk at that week, i.e.
    # censor_week > t. For a death at week t, this mainly matters if the person
    # transitions to dose3 earlier (trans_week < t), in which case they are censored
    # before t and should NOT contribute a death at t.
    total_death_hist = np.zeros(followup_weeks, dtype=int)
    total_edge_hist = np.zeros(followup_weeks, dtype=int)
    for t in range(followup_weeks):
        died_t = death_week == t
        at_risk_t = censor_week > t
        if np.any(died_t & at_risk_t):
            total_death_hist[t] = int(np.sum(died_t & at_risk_t))
        # Edge-case: death and transition same week AND still at risk (Rule B keeps them at-risk this week)
        if np.any(died_t & at_risk_t & (trans_week == t)):
            total_edge_hist[t] = int(np.sum(died_t & at_risk_t & (trans_week == t)))

    rng = np.random.default_rng(int(seed))
    HR = np.full((reps, followup_weeks), np.nan, dtype=float)
    for r in range(reps):
        pick = rng.choice(pool_n, size=n_treated, replace=False)
        # pick indexes into the "sub" arrays (not global df)
        cw_p = censor_week[pick]
        dw_p = death_week[pick]
        tw_p = trans_week[pick]

        censor_hist_p = np.bincount(np.clip(cw_p, 0, big), minlength=big + 1)
        death_hist_p = np.zeros(followup_weeks, dtype=int)
        edge_hist_p = np.zeros(followup_weeks, dtype=int)
        for t in range(followup_weeks):
            died_t = dw_p == t
            at_risk_t = cw_p > t
            if np.any(died_t & at_risk_t):
                death_hist_p[t] = int(np.sum(died_t & at_risk_t))
            if np.any(died_t & at_risk_t & (tw_p == t)):
                edge_hist_p[t] = int(np.sum(died_t & at_risk_t & (tw_p == t)))

        censor_hist_r = total_censor_hist - censor_hist_p
        death_hist_r = total_death_hist - death_hist_p

        # Alive(t) = n - count(censor_week <= t)
        cum_p = np.cumsum(censor_hist_p)
        cum_r = np.cumsum(censor_hist_r)
        alive_p = (n_treated - cum_p[:followup_weeks]).astype(int)
        alive_r = ((pool_n - n_treated) - cum_r[:followup_weeks]).astype(int)

        # Hazards and HR
        for t in range(followup_weeks):
            h_p = discrete_time_hazard(int(death_hist_p[t]), int(alive_p[t]))
            h_r = discrete_time_hazard(int(death_hist_r[t]), int(alive_r[t]))
            HR[r, t] = safe_div(h_p, h_r)

    q05 = np.nanquantile(HR, 0.05, axis=0)
    q50 = np.nanquantile(HR, 0.50, axis=0)
    q95 = np.nanquantile(HR, 0.95, axis=0)

    perm_df = pd.DataFrame(
        {
            "enrollment_date": [E.date().isoformat()] * followup_weeks,
            "t": list(range(followup_weeks)),
            "perm_q05": q05,
            "perm_q50": q50,
            "perm_q95": q95,
            "perm_reps": reps,
            "perm_N": n_treated,
            "eligible_size": pool_n,
        }
    )
    return perm_df, q05, q50, q95


def build_dose3_future_placebo_mask(
    df: pd.DataFrame,
    *,
    E: pd.Timestamp,
    start_weeks: int,
    end_weeks: int,
) -> pd.Series:
    """
    Placebo cohort: individuals who will receive dose 3 in [E+7*start_weeks, E+7*end_weeks),
    and are dose2 as-of E (i.e., dose2 date exists and <= E).

    Cohort semantics: these are treated as dose2 at week-start, and will be censored on dose3 transition
    using Rule B (starting the week after the transition week), via CohortSpec(dose2).
    """
    if start_weeks < 0 or end_weeks <= start_weeks:
        return pd.Series(False, index=df.index)
    alive_at_E = df["DateOfDeath"].isna() | (df["DateOfDeath"] >= E)
    d2 = df["Date_SecondDose"]
    d3 = df["Date_ThirdDose"]
    start = E + pd.Timedelta(days=7 * int(start_weeks))
    end = E + pd.Timedelta(days=7 * int(end_weeks))
    dose2_at_E = d2.notna() & (d2 <= E) & (d3.isna() | (d3 >= E))
    future3 = d3.notna() & (d3 >= start) & (d3 < end)
    return alive_at_E & dose2_at_E & future3


def restrict_dose2_to_eventual_dose3(
    df: pd.DataFrame,
    *,
    E: pd.Timestamp,
    horizon_weeks: int,
) -> pd.Series:
    """Mask for people who (as of E) will receive dose3 within [E, E+7*horizon_weeks)."""
    if horizon_weeks <= 0:
        return pd.Series(True, index=df.index)
    d3 = df["Date_ThirdDose"]
    start = E
    end = E + pd.Timedelta(days=7 * int(horizon_weeks))
    return d3.notna() & (d3 >= start) & (d3 < end)


def filter_non_mrna(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out any record with non-mRNA vaccines in doses 1-4 (mirrors KCOR_CMR/KCOR_ts behavior)."""
    try:
        from mfg_codes import parse_mfg, PFIZER, MODERNA  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(f"Could not import mfg_codes.parse_mfg: {e}") from e

    if not any(c in df.columns for c in VCODE_COLS):
        return df

    has_non_mrna = pd.Series(False, index=df.index)
    for col in VCODE_COLS:
        if col not in df.columns:
            continue
        codes = df[col]
        mfg_values = codes.apply(lambda x: parse_mfg(x) if pd.notna(x) and x != "" else None)
        non_mrna = (mfg_values.notna()) & (mfg_values != PFIZER) & (mfg_values != MODERNA)
        has_non_mrna = has_non_mrna | non_mrna
    return df[~has_non_mrna].copy()


def safe_div(num: float, den: float) -> float:
    if den == 0 or not math.isfinite(den):
        return float("nan")
    return num / den


def discrete_time_hazard(dead: int, alive: int) -> float:
    """h = -ln(1 - dead/alive); returns NaN if alive==0 or dead>alive or invalid."""
    if alive <= 0:
        return float("nan")
    if dead < 0 or dead > alive:
        return float("nan")
    mr = dead / alive
    # Clamp to avoid -ln(0) for mr=1 due to data issues.
    mr = min(max(mr, 0.0), 1.0 - 1e-12)
    return -math.log(1.0 - mr)


@dataclass(frozen=True)
class CohortSpec:
    name: str
    baseline_dose: int
    transition_col: str  # next dose date column name


COHORTS = [
    CohortSpec(name="dose0", baseline_dose=0, transition_col="Date_FirstDose"),
    CohortSpec(name="dose2", baseline_dose=2, transition_col="Date_ThirdDose"),
    CohortSpec(name="dose3", baseline_dose=3, transition_col="Date_FourthDose"),
]


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def enrollment_label(E: pd.Timestamp) -> str:
    return E.strftime("%Y%m%d")


def remove_legacy_plot(new_path: Path, legacy_path: Path) -> None:
    """Remove legacy plot filename if it exists and differs from new."""
    try:
        if legacy_path.exists() and legacy_path.resolve() != new_path.resolve():
            legacy_path.unlink()
    except Exception:
        # Best-effort cleanup; avoid breaking runs due to filesystem issues.
        pass


def birth_year_label(by_min: Optional[int], by_max: Optional[int]) -> str:
    if by_min is None or by_max is None:
        return "birth years: all (unknown included)"
    return f"birth years: {int(by_min)}-{int(by_max)}"


def birth_year_tag(by_min: Optional[int], by_max: Optional[int]) -> str:
    if by_min is None or by_max is None:
        return "by_all"
    return f"by_{int(by_min)}_{int(by_max)}"


def build_baseline_masks(df: pd.DataFrame, S: pd.Timestamp, E: pd.Timestamp) -> dict[str, pd.Series]:
    """Return boolean masks for baseline cohorts using as-of S and alive at E."""
    # Alive at E (week-start): death missing or death >= E
    alive_at_E = df["DateOfDeath"].isna() | (df["DateOfDeath"] >= E)

    d1 = df["Date_FirstDose"]
    d2 = df["Date_SecondDose"]
    d3 = df["Date_ThirdDose"]

    dose3_incident = (d3.notna()) & (d3 >= S) & (d3 < E)  # default; overridden by dose3_window_start in caller
    dose2_prevalent = (d2.notna()) & (d2 <= S) & (d3.isna() | (d3 > S))
    dose0_prevalent = d1.isna() | (d1 > S)

    # Apply alive-at-E gate to all cohorts
    return {
        "dose3": alive_at_E & dose3_incident,
        "dose2": alive_at_E & dose2_prevalent,
        "dose0": alive_at_E & dose0_prevalent,
    }


def build_baseline_masks_with_dose3_window(
    df: pd.DataFrame,
    *,
    S: pd.Timestamp,
    E: pd.Timestamp,
    dose3_window_start: pd.Timestamp,
) -> dict[str, pd.Series]:
    """Like build_baseline_masks, but dose3_incident uses [dose3_window_start, E)."""
    masks = build_baseline_masks(df, S=S, E=E)
    d3 = df["Date_ThirdDose"]
    d4 = df["Date_FourthDose"] if "Date_FourthDose" in df.columns else pd.Series(pd.NaT, index=df.index)
    alive_at_E = df["DateOfDeath"].isna() | (df["DateOfDeath"] >= E)
    # Dose 3 incident cohort: got dose 3 in the last N weeks before enrollment,
    # and did NOT already have dose 4 before enrollment.
    dose3_incident = (d3.notna()) & (d3 >= dose3_window_start) & (d3 < E) & (d4.isna() | (d4 >= E))
    masks["dose3"] = alive_at_E & dose3_incident
    return masks


def build_dose3_bin_masks(
    df: pd.DataFrame,
    *,
    E: pd.Timestamp,
    lookback_weeks: int,
) -> dict[int, pd.Series]:
    """
    Build disjoint dose-3 incident masks for week bins:
      bin k (k=1..lookback_weeks) corresponds to Date_ThirdDose in [E-7k, E-7(k-1)).

    Notes:
    - Uses week-start semantics with ISO-week Mondays.
    - Excludes those with dose 4 before enrollment (Date_FourthDose < E).
    - Does NOT apply alive-at-E gating; caller can combine with alive-at-E.
    """
    d3 = df["Date_ThirdDose"]
    d4 = df["Date_FourthDose"] if "Date_FourthDose" in df.columns else pd.Series(pd.NaT, index=df.index)
    no_d4_pre_enroll = d4.isna() | (d4 >= E)
    out: dict[int, pd.Series] = {}
    for k in range(1, int(lookback_weeks) + 1):
        start = E - pd.Timedelta(days=7 * k)
        end = E - pd.Timedelta(days=7 * (k - 1))
        out[k] = d3.notna() & (d3 >= start) & (d3 < end) & no_d4_pre_enroll
    return out


def compute_counts_for_cohort(
    df: pd.DataFrame,
    mask: pd.Series,
    cohort: CohortSpec,
    E: pd.Timestamp,
    followup_weeks: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Return (alive_series, dead_series, edge_sameweek_series) for t=0..followup_weeks-1.

    Edge case counted: DateOfDeath == transition_date (same ISO-week) AND within follow-up.
    """
    sub = df.loc[mask, ["DateOfDeath", cohort.transition_col]].copy()

    death = sub["DateOfDeath"]
    trans = sub[cohort.transition_col]

    # Rule B: censor starting the week AFTER death/transition week.
    death_censor_start = death + pd.Timedelta(days=7)
    trans_censor_start = trans + pd.Timedelta(days=7)

    # NaT -> +inf for min()
    death_censor_start = death_censor_start.fillna(pd.Timestamp.max)
    trans_censor_start = trans_censor_start.fillna(pd.Timestamp.max)
    censor_start = pd.concat([death_censor_start, trans_censor_start], axis=1).min(axis=1)

    alive_series: list[int] = []
    dead_series: list[int] = []
    edge_series: list[int] = []

    for t in range(followup_weeks):
        week_start = E + pd.Timedelta(days=7 * t)
        at_risk = censor_start > week_start
        alive = int(at_risk.sum())
        dead_this_week = int(((death == week_start) & at_risk).sum())

        # Edge-case count: death and transition same week (within follow-up)
        edge = int(((death == week_start) & (trans == week_start) & at_risk).sum())

        alive_series.append(alive)
        dead_series.append(dead_this_week)
        edge_series.append(edge)

    return alive_series, dead_series, edge_series


def plot_per_enrollment(
    outdir: Path,
    enroll_label: str,
    age_label: str,
    age_tag: str,
    t: np.ndarray,
    h0: np.ndarray,
    h2: np.ndarray,
    h3: np.ndarray,
    hr20: np.ndarray,
    hr30: np.ndarray,
    hr30_bins: Optional[dict[int, np.ndarray]] = None,
    hr32: Optional[np.ndarray] = None,
    hr32_bins: Optional[dict[int, np.ndarray]] = None,
    dead0: Optional[np.ndarray] = None,
    alive0: Optional[np.ndarray] = None,
    dead2: Optional[np.ndarray] = None,
    alive2: Optional[np.ndarray] = None,
    dead3: Optional[np.ndarray] = None,
    alive3: Optional[np.ndarray] = None,
    dead3_bins: Optional[dict[int, np.ndarray]] = None,
    alive3_bins: Optional[dict[int, np.ndarray]] = None,
) -> None:
    import matplotlib.pyplot as plt

    def _counts_panel(ax, *, title: str, death_series: list[tuple[str, np.ndarray]], alive_series: list[tuple[str, np.ndarray]]) -> None:
        ax2 = ax.twinx()
        n = max(1, len(death_series))
        width = 0.8 / n
        for i, (lab, d) in enumerate(death_series):
            ax.bar(t + (i - (n - 1) / 2) * width, d, width=width, alpha=0.35, label=f"dead {lab}")
        for lab, a in alive_series:
            ax2.plot(t, a, linewidth=1.6, alpha=0.9, label=f"alive {lab}")
        ax.set_ylabel("deaths")
        ax2.set_ylabel("alive (at risk)")
        ax.set_title(title)
        # merge legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    # Hazards
    plt.figure(figsize=(9, 5))
    plt.plot(t, h0, label="h0 (dose0)")
    plt.plot(t, h2, label="h2 (dose2)")
    plt.plot(t, h3, label="h3 (dose3 incident)")
    plt.xlabel("t (weeks since enrollment)")
    plt.ylabel("h(t) = -ln(1 - dead/alive)")
    plt.title(f"Hazards by cohort (E={enroll_label}; {age_label})")
    plt.legend()
    plt.tight_layout()
    new_path = outdir / f"h_curves_E{enroll_label}_{age_tag}.png"
    plt.savefig(new_path, dpi=160)
    remove_legacy_plot(new_path, outdir / f"h_curves_E{enroll_label}.png")
    plt.close()

    # HR (with counts panel)
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9, 7),
        gridspec_kw={"height_ratios": [3, 1.4]},
        sharex=True,
    )
    ax_top.plot(t, hr20, label="HR20 = h2/h0")
    ax_top.plot(t, hr30, label="HR30 = h3/h0")
    if hr30_bins:
        for k in sorted(hr30_bins.keys()):
            ax_top.plot(t, hr30_bins[k], linewidth=1.2, alpha=0.75, label=f"HR30_w{k} = h3_w{k}/h0")
    ax_top.set_ylabel("Hazard ratio")
    ax_top.set_title(f"HR curves (E={enroll_label}; {age_label})")
    ax_top.legend(fontsize=8)

    d0 = dead0 if dead0 is not None else np.full_like(t, np.nan, dtype=float)
    a0 = alive0 if alive0 is not None else np.full_like(t, np.nan, dtype=float)
    d2 = dead2 if dead2 is not None else np.full_like(t, np.nan, dtype=float)
    a2 = alive2 if alive2 is not None else np.full_like(t, np.nan, dtype=float)
    d3 = dead3 if dead3 is not None else np.full_like(t, np.nan, dtype=float)
    a3 = alive3 if alive3 is not None else np.full_like(t, np.nan, dtype=float)
    death_series = [("dose0", d0), ("dose2", d2), ("dose3", d3)]
    alive_series = [("dose0", a0), ("dose2", a2), ("dose3", a3)]
    if dead3_bins and alive3_bins:
        # Optional: include bin counts as faint additional cohorts (can be noisy).
        for k in sorted(dead3_bins.keys()):
            death_series.append((f"3_w{k}", dead3_bins[k]))
            alive_series.append((f"3_w{k}", alive3_bins[k]))
    _counts_panel(ax_bot, title="Counts by cohort", death_series=death_series, alive_series=alive_series)
    ax_bot.set_xlabel("t (weeks since enrollment)")
    fig.tight_layout()
    new_path = outdir / f"HR_curves_E{enroll_label}_{age_tag}.png"
    fig.savefig(new_path, dpi=160)
    remove_legacy_plot(new_path, outdir / f"HR_curves_E{enroll_label}.png")
    plt.close(fig)

    # HR32 (dose 3 vs dose 2)
    if hr32 is not None:
        fig, (ax_top, ax_bot) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(9, 7),
            gridspec_kw={"height_ratios": [3, 1.4]},
            sharex=True,
        )
        ax_top.plot(t, hr32, label="HR32 = h3/h2")
        if hr32_bins:
            for k in sorted(hr32_bins.keys()):
                ax_top.plot(t, hr32_bins[k], linewidth=1.2, alpha=0.75, label=f"HR32_w{k} = h3_w{k}/h2")
        ax_top.set_ylabel("Hazard ratio")
        ax_top.set_title(f"HR32 curves (E={enroll_label}; {age_label})")
        ax_top.legend(fontsize=8)

        d2 = dead2 if dead2 is not None else np.full_like(t, np.nan, dtype=float)
        a2 = alive2 if alive2 is not None else np.full_like(t, np.nan, dtype=float)
        d3 = dead3 if dead3 is not None else np.full_like(t, np.nan, dtype=float)
        a3 = alive3 if alive3 is not None else np.full_like(t, np.nan, dtype=float)
        death_series = [("dose2", d2), ("dose3", d3)]
        alive_series = [("dose2", a2), ("dose3", a3)]
        if dead3_bins and alive3_bins:
            for k in sorted(dead3_bins.keys()):
                death_series.append((f"3_w{k}", dead3_bins[k]))
                alive_series.append((f"3_w{k}", alive3_bins[k]))
        _counts_panel(ax_bot, title="Counts by cohort", death_series=death_series, alive_series=alive_series)
        ax_bot.set_xlabel("t (weeks since enrollment)")
        fig.tight_layout()
        new_path = outdir / f"HR32_curves_E{enroll_label}_{age_tag}.png"
        fig.savefig(new_path, dpi=160)
        remove_legacy_plot(new_path, outdir / f"HR32_curves_E{enroll_label}.png")
        plt.close(fig)


def plot_spaghetti(
    outdir: Path, series_df: pd.DataFrame, which: str, *, age_label: str = "", age_tag: str = "by_all"
) -> None:
    """Deprecated: kept for backwards compatibility; use plot_spaghetti_summary()."""
    import matplotlib.pyplot as plt

    if which not in series_df.columns:
        return

    plt.figure(figsize=(9, 5))
    for enroll_date, g in series_df.groupby("enrollment_date"):
        g2 = g.sort_values("t")
        plt.plot(g2["t"].values, g2[which].values, alpha=0.7, linewidth=1.5, label=str(enroll_date))
    plt.xlabel("t (weeks since enrollment)")
    plt.ylabel(which)
    suffix = f" ({age_label})" if age_label else ""
    plt.title(f"{which} spaghetti across enrollments{suffix}")
    # Too many legend entries; omit by default.
    plt.tight_layout()
    new_path = outdir / f"{which}_spaghetti_{age_tag}.png"
    plt.savefig(new_path, dpi=160)
    remove_legacy_plot(new_path, outdir / f"{which}_spaghetti.png")
    plt.close()


def _curve_count_columns(which: str) -> list[tuple[str, str, str]]:
    """
    Returns list of (label, dead_col, alive_col) to show for this curve.
    """
    # Core curves
    if which == "HR20":
        return [("dose0", "dead0", "alive0"), ("dose2", "dead2", "alive2")]
    if which == "HR30":
        return [("dose0", "dead0", "alive0"), ("dose3", "dead3", "alive3")]
    if which == "HR32":
        return [("dose2", "dead2", "alive2"), ("dose3", "dead3", "alive3")]

    # Dose-3 bin curves
    if which.startswith("HR30_w"):
        k = which.split("_w")[1]
        return [("dose0", "dead0", "alive0"), (f"3_w{k}", f"dead3_w{k}", f"alive3_w{k}")]
    if which.startswith("HR32_w"):
        k = which.split("_w")[1]
        return [("dose2", "dead2", "alive2"), (f"3_w{k}", f"dead3_w{k}", f"alive3_w{k}")]

    # Existing placebo curves
    if which == "HR_future30":
        return [("dose0", "dead0", "alive0"), ("future3", "dead_future3", "alive_future3")]
    if which == "HR_future32":
        return [("dose2", "dead2", "alive2"), ("future3", "dead_future3", "alive_future3")]

    # Selection suite curves (if present)
    if which == "HR_tt32":
        return [("2_elig", "dead2_elig", "alive2_elig"), ("3_tt", "dead3_tt", "alive3_tt")]
    if which.startswith("HR_lead"):
        k = which.replace("HR_lead", "")
        return [("2_elig", "dead2_elig", "alive2_elig"), (f"lead{k}", f"dead_lead{k}", f"alive_lead{k}")]

    # Fallback: no count panel
    return []


def plot_spaghetti_summary(
    outdir: Path,
    series_df: pd.DataFrame,
    which: str,
    *,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    age_label: str = "",
    age_tag: str = "by_all",
) -> None:
    """
    Summary spaghetti plot:
    - Top: median across enrollments with quantile envelope.
    - Bottom: deaths (bars) + alive (line) for relevant cohorts (median across enrollments).
    """
    import matplotlib.pyplot as plt

    if which not in series_df.columns:
        return

    df = series_df.loc[:, ["enrollment_date", "t", which]].copy()
    df[which] = pd.to_numeric(df[which], errors="coerce")

    # Curve summaries by t
    g = df.groupby("t")[which]
    med = g.median()
    lo = g.quantile(q_lo)
    hi = g.quantile(q_hi)
    t_vals = med.index.to_numpy(dtype=int)

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9, 7),
        gridspec_kw={"height_ratios": [3, 1.4]},
        sharex=True,
    )
    ax_top.fill_between(t_vals, lo.to_numpy(dtype=float), hi.to_numpy(dtype=float), alpha=0.25, label=f"q{int(q_lo*100)}–q{int(q_hi*100)} across enrollments")
    ax_top.plot(t_vals, med.to_numpy(dtype=float), linewidth=2.2, label="median")
    ax_top.set_ylabel(which)
    suffix = f" ({age_label})" if age_label else ""
    ax_top.set_title(f"{which} summary across enrollments{suffix}")
    ax_top.legend(fontsize=9)

    # Counts panel
    counts = _curve_count_columns(which)
    if counts:
        need_cols = ["enrollment_date", "t"]
        for _, dcol, acol in counts:
            if dcol in series_df.columns:
                need_cols.append(dcol)
            if acol in series_df.columns:
                need_cols.append(acol)
        cdf = series_df.loc[:, list(dict.fromkeys(need_cols))].copy()
        for _, dcol, acol in counts:
            if dcol in cdf.columns:
                cdf[dcol] = pd.to_numeric(cdf[dcol], errors="coerce")
            if acol in cdf.columns:
                cdf[acol] = pd.to_numeric(cdf[acol], errors="coerce")

        # Median across enrollments by t
        by_t = cdf.groupby("t").median(numeric_only=True)

        ax2 = ax_bot.twinx()
        n = len(counts)
        width = 0.8 / max(1, n)
        for i, (lab, dcol, acol) in enumerate(counts):
            if dcol in by_t.columns:
                ax_bot.bar(
                    t_vals + (i - (n - 1) / 2) * width,
                    by_t.loc[t_vals, dcol].to_numpy(dtype=float),
                    width=width,
                    alpha=0.35,
                    label=f"dead {lab}",
                )
            if acol in by_t.columns:
                ax2.plot(
                    t_vals,
                    by_t.loc[t_vals, acol].to_numpy(dtype=float),
                    linewidth=1.6,
                    alpha=0.9,
                    label=f"alive {lab}",
                )
        ax_bot.set_ylabel("deaths (median across enrollments)")
        ax2.set_ylabel("alive (median across enrollments)")
        ax_bot.set_title("Counts (median across enrollments)")
        h1, l1 = ax_bot.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax_bot.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    else:
        ax_bot.axis("off")

    ax_bot.set_xlabel("t (weeks since enrollment)")
    fig.tight_layout()
    new_path = outdir / f"{which}_summary_{age_tag}.png"
    fig.savefig(new_path, dpi=160)
    plt.close(fig)


def plot_placebo_per_enrollment(
    outdir: Path,
    enroll_label: str,
    age_label: str,
    age_tag: str,
    t: np.ndarray,
    hr_future32: np.ndarray,
    hr_future30: Optional[np.ndarray] = None,
    dead_future3: Optional[np.ndarray] = None,
    alive_future3: Optional[np.ndarray] = None,
    dead2: Optional[np.ndarray] = None,
    alive2: Optional[np.ndarray] = None,
    dead0: Optional[np.ndarray] = None,
    alive0: Optional[np.ndarray] = None,
) -> None:
    """Plot placebo future-booster HR curves if present."""
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9, 7),
        gridspec_kw={"height_ratios": [3, 1.4]},
        sharex=True,
    )
    ax_top.plot(t, hr_future32, label="HR_future32 = h_future3/h2 (placebo)")
    if hr_future30 is not None:
        ax_top.plot(t, hr_future30, label="HR_future30 = h_future3/h0 (placebo)", alpha=0.85)
    ax_top.set_ylabel("Hazard ratio")
    ax_top.set_title(f"Future-booster placebo HR (E={enroll_label}; {age_label})")
    ax_top.legend(fontsize=9)

    # counts panel (future3 plus comparators)
    death_series: list[tuple[str, np.ndarray]] = []
    alive_series: list[tuple[str, np.ndarray]] = []
    if dead_future3 is not None and alive_future3 is not None:
        death_series.append(("future3", dead_future3))
        alive_series.append(("future3", alive_future3))
    if dead2 is not None and alive2 is not None:
        death_series.append(("dose2", dead2))
        alive_series.append(("dose2", alive2))
    if hr_future30 is not None and dead0 is not None and alive0 is not None:
        death_series.append(("dose0", dead0))
        alive_series.append(("dose0", alive0))

    ax2 = ax_bot.twinx()
    n = max(1, len(death_series))
    width = 0.8 / n
    for i, (lab, d) in enumerate(death_series):
        ax_bot.bar(t + (i - (n - 1) / 2) * width, d, width=width, alpha=0.35, label=f"dead {lab}")
    for lab, a in alive_series:
        ax2.plot(t, a, linewidth=1.6, alpha=0.9, label=f"alive {lab}")
    ax_bot.set_ylabel("deaths")
    ax2.set_ylabel("alive (at risk)")
    ax_bot.set_title("Counts by cohort")
    h1, l1 = ax_bot.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax_bot.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax_bot.set_xlabel("t (weeks since enrollment)")
    fig.tight_layout()
    new_path = outdir / f"HR_future_placebo_curves_E{enroll_label}_{age_tag}.png"
    fig.savefig(new_path, dpi=160)
    remove_legacy_plot(new_path, outdir / f"HR_future_placebo_curves_E{enroll_label}.png")
    plt.close(fig)


def summarize_enrollment(g: pd.DataFrame) -> dict[str, object]:
    """Compute summary metrics for one enrollment group g (rows for t=0..)."""
    g2 = g.sort_values("t")
    hr30 = g2["HR30"].to_numpy(dtype=float)
    hr20 = g2["HR20"].to_numpy(dtype=float)
    hr32 = g2["HR32"].to_numpy(dtype=float) if "HR32" in g2.columns else np.array([], dtype=float)
    hr_future32 = g2["HR_future32"].to_numpy(dtype=float) if "HR_future32" in g2.columns else np.array([], dtype=float)

    def peak_info(arr: np.ndarray) -> tuple[Optional[int], float]:
        if arr.size == 0:
            return None, float("nan")
        if np.all(~np.isfinite(arr)):
            return None, float("nan")
        idx = int(np.nanargmax(arr))
        return idx, float(arr[idx])

    peak_t_hr30, peak_val_hr30 = peak_info(hr30)
    peak_t_hr20, peak_val_hr20 = peak_info(hr20)
    peak_t_hr32, peak_val_hr32 = peak_info(hr32)
    peak_t_hrf32, peak_val_hrf32 = peak_info(hr_future32)

    def at_t(arr: np.ndarray, t: int) -> float:
        if t < 0 or t >= arr.size:
            return float("nan")
        return float(arr[t])

    return {
        "enrollment_date": g2["enrollment_date"].iloc[0],
        "peak_week_HR30": peak_t_hr30,
        "peak_value_HR30": peak_val_hr30,
        "HR30_at_t0": at_t(hr30, 0),
        "HR30_at_t2": at_t(hr30, 2),
        "peak_week_HR20": peak_t_hr20,
        "peak_value_HR20": peak_val_hr20,
        "HR20_at_t0": at_t(hr20, 0),
        "HR20_at_t2": at_t(hr20, 2),
        "peak_week_HR32": peak_t_hr32,
        "peak_value_HR32": peak_val_hr32,
        "HR32_at_t0": at_t(hr32, 0),
        "HR32_at_t2": at_t(hr32, 2),
        "peak_week_HR_future32": peak_t_hrf32,
        "peak_value_HR_future32": peak_val_hrf32,
        "HR_future32_at_t0": at_t(hr_future32, 0),
        "HR_future32_at_t2": at_t(hr_future32, 2),
        "edge_death_transition_sameweek_dose0": int(g2["edge_sameweek0"].sum()),
        "edge_death_transition_sameweek_dose2": int(g2["edge_sameweek2"].sum()),
        "edge_death_transition_sameweek_dose3": int(g2["edge_sameweek3"].sum()),
    }


def main() -> int:
    t0 = time.perf_counter()
    start_local = datetime.now().astimezone()
    print(f"[build_weekly_emulation] start_local={start_local.isoformat()}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Czech/records.csv")
    ap.add_argument("--outdir", default="identifiability/Czech/booster")
    ap.add_argument("--enrollment-start", default="2021-10-18", help="YYYY-MM-DD (Monday)")
    ap.add_argument("--n-enrollments", type=int, default=10)
    ap.add_argument("--followup-weeks", type=int, default=26)
    # Default: no birth-year restriction (all ages). Provide both to enable filtering.
    ap.add_argument("--birth-year-min", type=int, default=None)
    ap.add_argument("--birth-year-max", type=int, default=None)
    ap.add_argument(
        "--strata",
        action="append",
        default=None,
        help=(
            "Optional: run multiple birth-year strata in one process, writing each to <outdir>/<name>/ . "
            "Format: name[:min[:max]] where min/max are birth years (inclusive). "
            "Example: --strata all_ages --strata born_193x:1930:1939"
        ),
    )
    ap.add_argument("--lookback-days", type=int, default=7)
    ap.add_argument(
        "--dose3-incident-lookback-weeks",
        type=int,
        default=4,
        help="Dose 3 incident cohort window length (weeks before enrollment).",
    )
    ap.add_argument(
        "--dose3-bin-weeks",
        type=int,
        default=4,
        help="Also compute dose-3 HR30 curves for week bins 1..K (K weeks before enrollment).",
    )
    ap.add_argument(
        "--dose3-future-start-weeks",
        type=int,
        default=None,
        help="Future-booster placebo: start of future window in weeks after enrollment (e.g., 4).",
    )
    ap.add_argument(
        "--dose3-future-end-weeks",
        type=int,
        default=None,
        help="Future-booster placebo: end of future window in weeks after enrollment (e.g., 8).",
    )
    ap.add_argument(
        "--restrict-dose2-eventual-dose3-weeks",
        type=int,
        default=None,
        help="Restrict dose2 cohort to those who will receive dose3 within this many weeks after enrollment (eventual-booster restriction).",
    )
    ap.add_argument(
        "--restrict-tsd2-min-days",
        type=int,
        default=None,
        help="Restrict to individuals with time-since-dose2 at enrollment >= this many days (tsd2 stratification).",
    )
    ap.add_argument(
        "--restrict-tsd2-max-days",
        type=int,
        default=None,
        help="Restrict to individuals with time-since-dose2 at enrollment < this many days (tsd2 stratification).",
    )
    ap.add_argument(
        "--exclude-recent-infection-weeks",
        type=int,
        default=None,
        help="Exclude anyone with DateOfPositiveTest within the last N weeks before enrollment.",
    )
    # --- Selection/eligibility falsification suite (dose2-eligible risk set) ---
    ap.add_argument(
        "--selection-suite",
        action="store_true",
        default=False,
        help="Compute selection/eligibility falsification outputs: HR_tt32, HR_lead_k, and permutation placebo bands.",
    )
    ap.add_argument("--tt-tsd2-min-days", type=int, default=180)
    ap.add_argument("--tt-tsd2-max-days", type=int, default=360)
    ap.add_argument("--tt-treated-window-weeks", type=int, default=1)
    ap.add_argument("--lead-max-weeks", type=int, default=8)
    ap.add_argument("--perm-reps", type=int, default=200)
    ap.add_argument("--perm-seed", type=int, default=1)
    ap.add_argument("--filter-non-mrna", dest="filter_non_mrna", action="store_true", default=True)
    ap.add_argument("--no-filter-non-mrna", dest="filter_non_mrna", action="store_false")
    ap.add_argument("--max-rows", type=int, default=None, help="Debug: read only first N rows")
    args = ap.parse_args()

    input_path = args.input
    print(f"Reading {input_path} ...", flush=True)
    df_raw = _read_csv_flex(input_path, max_rows=args.max_rows)

    expected_col_count = 53
    if df_raw.shape[1] != expected_col_count:
        raise SystemExit(
            f"ERROR: Input parsed into {df_raw.shape[1]} columns, expected {expected_col_count}. "
            "Delimiter/encoding mismatch likely."
        )

    # Rename to standard schema (same as KCOR_CMR.py / KCOR_ts.py)
    df_raw.columns = [
        "ID",
        "Infection",
        "Sex",
        "YearOfBirth",
        "DateOfPositiveTest",
        "DateOfResult",
        "Recovered",
        "Date_COVID_death",
        "Symptom",
        "TestType",
        "Date_FirstDose",
        "Date_SecondDose",
        "Date_ThirdDose",
        "Date_FourthDose",
        "Date_FifthDose",
        "Date_SixthDose",
        "Date_SeventhDose",
        "VaccineCode_FirstDose",
        "VaccineCode_SecondDose",
        "VaccineCode_ThirdDose",
        "VaccineCode_FourthDose",
        "VaccineCode_FifthDose",
        "VaccineCode_SixthDose",
        "VaccineCode_SeventhDose",
        "PrimaryCauseHospCOVID",
        "bin_Hospitalization",
        "min_Hospitalization",
        "days_Hospitalization",
        "max_Hospitalization",
        "bin_ICU",
        "min_ICU",
        "days_ICU",
        "max_ICU",
        "bin_StandardWard",
        "min_StandardWard",
        "days_StandardWard",
        "max_StandardWard",
        "bin_Oxygen",
        "min_Oxygen",
        "days_Oxygen",
        "max_Oxygen",
        "bin_HFNO",
        "min_HFNO",
        "days_HFNO",
        "max_HFNO",
        "bin_MechanicalVentilation_ECMO",
        "min_MechanicalVentilation_ECMO",
        "days_MechanicalVentilation_ECMO",
        "max_MechanicalVentilation_ECMO",
        "Mutation",
        "DateOfDeath",
        "Long_COVID",
        "DCCI",
    ]

    # Keep only columns needed for this analysis to reduce memory pressure.
    keep_cols = ["Infection", "YearOfBirth", "DateOfDeath"] + INFECTION_DATE_COLS + DOSE_DATE_COLS + VCODE_COLS
    keep_cols = [c for c in keep_cols if c in df_raw.columns]
    df = df_raw.loc[:, keep_cols].copy()
    del df_raw

    parse_birth_year(df)

    df = filter_infection_single(df)
    print(f"After Infection<=1 filter: {len(df):,} rows", flush=True)

    if args.filter_non_mrna:
        before = len(df)
        df = filter_non_mrna(df)
        print(f"After non-mRNA filter: kept {len(df):,}/{before:,}", flush=True)

    parse_dates_inplace(df)

    def run_one_stratum(
        df_s: pd.DataFrame, outdir_s: Path, stratum_label: str, *, birth_year_min: Optional[int], birth_year_max: Optional[int]
    ) -> None:
        ensure_outdir(outdir_s)
        age_label = birth_year_label(birth_year_min, birth_year_max)
        age_tag = birth_year_tag(birth_year_min, birth_year_max)
        enrollment_start = pd.to_datetime(args.enrollment_start)
        if enrollment_start.weekday() != 0:
            print("WARNING: enrollment_start is not a Monday; week-start semantics assume Monday.", flush=True)

        followup_weeks = int(args.followup_weeks)
        rows: list[dict[str, object]] = []
        perm_all: list[pd.DataFrame] = []

        print(f"\n=== Stratum: {stratum_label}  rows={len(df_s):,}  outdir={outdir_s} ===", flush=True)
        print(f"    {age_label}", flush=True)

        for i in range(int(args.n_enrollments)):
            E = enrollment_start + pd.Timedelta(days=7 * i)
            S = E - pd.Timedelta(days=int(args.lookback_days))

            dose3_window_start = E - pd.Timedelta(days=7 * int(args.dose3_incident_lookback_weeks))
            masks = build_baseline_masks_with_dose3_window(df_s, S=S, E=E, dose3_window_start=dose3_window_start)
            alive_at_E = df_s["DateOfDeath"].isna() | (df_s["DateOfDeath"] >= E)
            dose3_bins = build_dose3_bin_masks(df_s, E=E, lookback_weeks=int(args.dose3_bin_weeks))

            # Optional per-enrollment restrictions (recent infection / tsd2)
            if args.exclude_recent_infection_weeks is not None:
                keep_inf = mask_not_recent_infection(df_s, E=E, exclude_recent_weeks=int(args.exclude_recent_infection_weeks))
                for k in list(masks.keys()):
                    masks[k] = masks[k] & keep_inf
                for k in list(dose3_bins.keys()):
                    dose3_bins[k] = dose3_bins[k] & keep_inf
                alive_at_E = alive_at_E & keep_inf

            keep_tsd2: Optional[pd.Series] = None
            if args.restrict_tsd2_min_days is not None or args.restrict_tsd2_max_days is not None:
                keep_tsd2 = mask_tsd2_range(
                    df_s, E=E, min_days=args.restrict_tsd2_min_days, max_days=args.restrict_tsd2_max_days
                )
                # IMPORTANT: apply tsd2 gating only to cohorts that *require* dose2 to exist.
                for k in ("dose2", "dose3"):
                    if k in masks:
                        masks[k] = masks[k] & keep_tsd2
                for k in list(dose3_bins.keys()):
                    dose3_bins[k] = dose3_bins[k] & keep_tsd2

            # Optional: restrict dose2 cohort to eventual boosters (dose3 within horizon after E)
            if args.restrict_dose2_eventual_dose3_weeks is not None:
                will_boost = restrict_dose2_to_eventual_dose3(
                    df_s,
                    E=E,
                    horizon_weeks=int(args.restrict_dose2_eventual_dose3_weeks),
                )
                masks["dose2"] = masks["dose2"] & will_boost

            enroll_lbl = enrollment_label(E)
            print(
                f"\nEnrollment {i+1}/{args.n_enrollments}: E={E.date().isoformat()} (label={enroll_lbl}), S={S.date().isoformat()}",
                flush=True,
            )
            print(
                f"  Baseline sizes: dose0={int(masks['dose0'].sum()):,}, dose2={int(masks['dose2'].sum()):,}, dose3_incident={int(masks['dose3'].sum()):,}",
                flush=True,
            )
            if int(args.dose3_bin_weeks) > 1:
                _parts = ", ".join(
                    [f"w{k}={int((alive_at_E & dose3_bins[k]).sum()):,}" for k in sorted(dose3_bins.keys())]
                )
                print(f"  Dose3 bins (alive@E): {_parts}", flush=True)

            alive0, dead0, edge0 = compute_counts_for_cohort(
                df_s, masks["dose0"], COHORTS[0], E=E, followup_weeks=followup_weeks
            )
            alive2, dead2, edge2 = compute_counts_for_cohort(
                df_s, masks["dose2"], COHORTS[1], E=E, followup_weeks=followup_weeks
            )
            alive3, dead3, edge3 = compute_counts_for_cohort(
                df_s, masks["dose3"], COHORTS[2], E=E, followup_weeks=followup_weeks
            )

            # --- Selection suite (optional) ---
            alive2_elig: Optional[list[int]] = None
            dead2_elig: Optional[list[int]] = None
            edge2_elig: Optional[list[int]] = None
            alive3_tt: Optional[list[int]] = None
            dead3_tt: Optional[list[int]] = None
            edge3_tt: Optional[list[int]] = None
            lead_counts: dict[int, tuple[list[int], list[int], list[int]]] = {}
            perm_q: dict[str, np.ndarray] = {}

            if args.selection_suite:
                tsd2_min = int(args.tt_tsd2_min_days)
                tsd2_max = int(args.tt_tsd2_max_days)
                treated_w = int(args.tt_treated_window_weeks)
                lead_max = int(args.lead_max_weeks)

                elig2 = mask_eligible_dose2(df_s, E=E, tsd2_min_days=tsd2_min, tsd2_max_days=tsd2_max) & alive_at_E
                tt3 = (
                    mask_tt_treated_recent_dose3(
                        df_s,
                        E=E,
                        treated_window_weeks=treated_w,
                        tsd2_min_days=tsd2_min,
                        tsd2_max_days=tsd2_max,
                    )
                    & alive_at_E
                )

                print(
                    f"  Selection suite (tsd2 {tsd2_min}-{tsd2_max}d): eligible_dose2={int(elig2.sum()):,}, tt_treated3_recent={int(tt3.sum()):,}",
                    flush=True,
                )

                alive2_elig, dead2_elig, edge2_elig = compute_counts_for_cohort(
                    df_s, elig2, COHORTS[1], E=E, followup_weeks=followup_weeks
                )
                alive3_tt, dead3_tt, edge3_tt = compute_counts_for_cohort(
                    df_s, tt3, COHORTS[2], E=E, followup_weeks=followup_weeks
                )

                for k in range(1, lead_max + 1):
                    m_lead = mask_future_booster_k(df_s, E=E, k=k, tsd2_min_days=tsd2_min, tsd2_max_days=tsd2_max) & alive_at_E
                    lead_counts[k] = compute_counts_for_cohort(
                        df_s, m_lead, COHORTS[1], E=E, followup_weeks=followup_weeks
                    )

                # Permutation placebo envelope for HR_perm(t) among eligible dose2
                perm_df, q05, q50, q95 = _perm_band_for_enrollment(
                    df_s,
                    E=E,
                    eligible_mask=elig2,
                    n_treated=int(tt3.sum()),
                    followup_weeks=followup_weeks,
                    reps=int(args.perm_reps),
                    seed=int(args.perm_seed) + 1000 * i,
                )
                perm_all.append(perm_df)
                perm_q = {"q05": q05, "q50": q50, "q95": q95}

            # Future-booster placebo (optional): those who will get dose3 in a future window.
            alive_fut: Optional[list[int]] = None
            dead_fut: Optional[list[int]] = None
            edge_fut: Optional[list[int]] = None
            if args.dose3_future_start_weeks is not None and args.dose3_future_end_weeks is not None:
                m_fut = build_dose3_future_placebo_mask(
                    df_s,
                    E=E,
                    start_weeks=int(args.dose3_future_start_weeks),
                    end_weeks=int(args.dose3_future_end_weeks),
                )
                m_fut = m_fut & alive_at_E
                if keep_tsd2 is not None:
                    m_fut = m_fut & keep_tsd2
                alive_fut, dead_fut, edge_fut = compute_counts_for_cohort(
                    df_s, m_fut, COHORTS[1], E=E, followup_weeks=followup_weeks
                )

            # Dose-3 bins (week -1..-K); same censoring semantics as dose3
            alive3b: dict[int, list[int]] = {}
            dead3b: dict[int, list[int]] = {}
            edge3b: dict[int, list[int]] = {}
            for k, m in dose3_bins.items():
                m2 = alive_at_E & m
                a_k, d_k, e_k = compute_counts_for_cohort(df_s, m2, COHORTS[2], E=E, followup_weeks=followup_weeks)
                alive3b[k] = a_k
                dead3b[k] = d_k
                edge3b[k] = e_k

            for t in range(followup_weeks):
                week_start = (E + pd.Timedelta(days=7 * t)).date().isoformat()
                h0 = discrete_time_hazard(dead0[t], alive0[t])
                h2 = discrete_time_hazard(dead2[t], alive2[t])
                h3 = discrete_time_hazard(dead3[t], alive3[t])
                row: dict[str, object] = {
                    "enrollment_date": E.date().isoformat(),
                    "t": t,
                    "calendar_week": week_start,
                    "dead0": dead0[t],
                    "alive0": alive0[t],
                    "h0": h0,
                    "dead2": dead2[t],
                    "alive2": alive2[t],
                    "h2": h2,
                    "dead3": dead3[t],
                    "alive3": alive3[t],
                    "h3": h3,
                    "HR20": safe_div(h2, h0),
                    "HR30": safe_div(h3, h0),
                    "HR32": safe_div(h3, h2),
                    "edge_sameweek0": edge0[t],
                    "edge_sameweek2": edge2[t],
                    "edge_sameweek3": edge3[t],
                }
                if alive2_elig is not None and dead2_elig is not None and edge2_elig is not None:
                    h2e = discrete_time_hazard(dead2_elig[t], alive2_elig[t])
                    row["dead2_elig"] = dead2_elig[t]
                    row["alive2_elig"] = alive2_elig[t]
                    row["h2_elig"] = h2e
                    row["edge_sameweek2_elig"] = edge2_elig[t]
                if (
                    alive3_tt is not None
                    and dead3_tt is not None
                    and edge3_tt is not None
                    and alive2_elig is not None
                    and dead2_elig is not None
                ):
                    h3tt = discrete_time_hazard(dead3_tt[t], alive3_tt[t])
                    h2e = discrete_time_hazard(dead2_elig[t], alive2_elig[t])
                    row["dead3_tt"] = dead3_tt[t]
                    row["alive3_tt"] = alive3_tt[t]
                    row["h3_tt"] = h3tt
                    row["HR_tt32"] = safe_div(h3tt, h2e)
                    row["edge_sameweek3_tt"] = edge3_tt[t]
                if lead_counts:
                    if alive2_elig is not None and dead2_elig is not None:
                        h2e = discrete_time_hazard(dead2_elig[t], alive2_elig[t])
                    else:
                        h2e = float("nan")
                    for k in sorted(lead_counts.keys()):
                        a_k, d_k, e_k = lead_counts[k]
                        h_k = discrete_time_hazard(d_k[t], a_k[t])
                        row[f"dead_lead{k}"] = d_k[t]
                        row[f"alive_lead{k}"] = a_k[t]
                        row[f"h_lead{k}"] = h_k
                        row[f"HR_lead{k}"] = safe_div(h_k, h2e)
                        row[f"edge_sameweek_lead{k}"] = e_k[t]
                if alive_fut is not None and dead_fut is not None and edge_fut is not None:
                    h_fut = discrete_time_hazard(dead_fut[t], alive_fut[t])
                    row["dead_future3"] = dead_fut[t]
                    row["alive_future3"] = alive_fut[t]
                    row["h_future3"] = h_fut
                    row["HR_future30"] = safe_div(h_fut, h0)
                    row["HR_future32"] = safe_div(h_fut, h2)
                    row["edge_sameweek_future3"] = edge_fut[t]
                for k in sorted(alive3b.keys()):
                    h3k = discrete_time_hazard(dead3b[k][t], alive3b[k][t])
                    row[f"dead3_w{k}"] = dead3b[k][t]
                    row[f"alive3_w{k}"] = alive3b[k][t]
                    row[f"h3_w{k}"] = h3k
                    row[f"HR30_w{k}"] = safe_div(h3k, h0)
                    row[f"HR32_w{k}"] = safe_div(h3k, h2)
                    row[f"edge_sameweek3_w{k}"] = edge3b[k][t]
                rows.append(row)

            # Plots per enrollment
            g = pd.DataFrame([r for r in rows if r["enrollment_date"] == E.date().isoformat()])
            plot_per_enrollment(
                outdir_s,
                enroll_lbl,
                age_label,
                age_tag,
                t=g["t"].to_numpy(dtype=int),
                h0=g["h0"].to_numpy(dtype=float),
                h2=g["h2"].to_numpy(dtype=float),
                h3=g["h3"].to_numpy(dtype=float),
                hr20=g["HR20"].to_numpy(dtype=float),
                hr30=g["HR30"].to_numpy(dtype=float),
                hr30_bins={k: g[f"HR30_w{k}"].to_numpy(dtype=float) for k in sorted(dose3_bins.keys())}
                if dose3_bins
                else None,
                hr32=g["HR32"].to_numpy(dtype=float),
                hr32_bins={k: g[f"HR32_w{k}"].to_numpy(dtype=float) for k in sorted(dose3_bins.keys())}
                if dose3_bins
                else None,
                dead0=g["dead0"].to_numpy(dtype=float),
                alive0=g["alive0"].to_numpy(dtype=float),
                dead2=g["dead2"].to_numpy(dtype=float),
                alive2=g["alive2"].to_numpy(dtype=float),
                dead3=g["dead3"].to_numpy(dtype=float),
                alive3=g["alive3"].to_numpy(dtype=float),
                dead3_bins={k: g[f"dead3_w{k}"].to_numpy(dtype=float) for k in sorted(dose3_bins.keys())}
                if dose3_bins
                else None,
                alive3_bins={k: g[f"alive3_w{k}"].to_numpy(dtype=float) for k in sorted(dose3_bins.keys())}
                if dose3_bins
                else None,
            )
            if "HR_future32" in g.columns:
                plot_placebo_per_enrollment(
                    outdir_s,
                    enroll_lbl,
                    age_label,
                    age_tag,
                    t=g["t"].to_numpy(dtype=int),
                    hr_future32=g["HR_future32"].to_numpy(dtype=float),
                    hr_future30=g["HR_future30"].to_numpy(dtype=float) if "HR_future30" in g.columns else None,
                    dead_future3=g["dead_future3"].to_numpy(dtype=float) if "dead_future3" in g.columns else None,
                    alive_future3=g["alive_future3"].to_numpy(dtype=float) if "alive_future3" in g.columns else None,
                    dead2=g["dead2"].to_numpy(dtype=float),
                    alive2=g["alive2"].to_numpy(dtype=float),
                    dead0=g["dead0"].to_numpy(dtype=float),
                    alive0=g["alive0"].to_numpy(dtype=float),
                )

            # Permutation band plot (optional)
            if args.selection_suite and perm_q:
                try:
                    import matplotlib.pyplot as plt

                    t_arr = g["t"].to_numpy(dtype=int)
                    plt.figure(figsize=(9, 5))
                    plt.fill_between(
                        t_arr, perm_q["q05"], perm_q["q95"], alpha=0.25, label="perm 5–95% band (eligible dose2)"
                    )
                    plt.plot(t_arr, perm_q["q50"], linewidth=1.8, label="perm median", alpha=0.9)
                    lead_cols = sorted(
                        [c for c in g.columns if c.startswith("HR_lead")], key=lambda x: int(x.replace("HR_lead", ""))
                    )
                    for c in lead_cols:
                        plt.plot(
                            t_arr,
                            pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float),
                            linewidth=1.0,
                            alpha=0.6,
                            label=c,
                        )
                    if "HR_tt32" in g.columns:
                        plt.plot(
                            t_arr,
                            pd.to_numeric(g["HR_tt32"], errors="coerce").to_numpy(dtype=float),
                            color="black",
                            linewidth=2.2,
                            label="HR_tt32",
                        )
                    plt.xlabel("t (weeks since enrollment)")
                    plt.ylabel("Hazard ratio")
                    plt.title(f"Permutation placebo band vs observed (E={enroll_lbl}; {age_label})")
                    if len(lead_cols) <= 6:
                        plt.legend()
                    plt.tight_layout()
                    new_path = outdir_s / f"perm_band_E{enroll_lbl}_{age_tag}.png"
                    plt.savefig(new_path, dpi=160)
                    remove_legacy_plot(new_path, outdir_s / f"perm_band_E{enroll_lbl}.png")
                    plt.close()
                except Exception as e:
                    print(f"WARNING: permutation band plot failed: {e}", flush=True)

            if (g["dead0"] > g["alive0"]).any() or (g["dead2"] > g["alive2"]).any() or (g["dead3"] > g["alive3"]).any():
                print("WARNING: Found dead > alive in some week; check parsing/semantics.", flush=True)

        series_df = pd.DataFrame(rows)
        series_path = outdir_s / "series.csv"
        series_df.to_csv(series_path, index=False)
        print(f"\nWrote {series_path} ({len(series_df):,} rows)", flush=True)

        if args.selection_suite and perm_all:
            perm_path = outdir_s / "perm_summary.csv"
            pd.concat(perm_all, ignore_index=True).to_csv(perm_path, index=False)
            print(f"Wrote {perm_path} ({sum(len(d) for d in perm_all):,} rows)", flush=True)

        summaries = [summarize_enrollment(g) for _, g in series_df.groupby("enrollment_date")]
        summary_df = pd.DataFrame(summaries).sort_values("enrollment_date")
        summary_path = outdir_s / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path} ({len(summary_df):,} rows)", flush=True)

        # Summary spaghetti plots: median + quantile envelope + counts panel
        plot_spaghetti_summary(outdir_s, series_df, "HR30", age_label=age_label, age_tag=age_tag)
        plot_spaghetti_summary(outdir_s, series_df, "HR20", age_label=age_label, age_tag=age_tag)
        plot_spaghetti_summary(outdir_s, series_df, "HR32", age_label=age_label, age_tag=age_tag)
        plot_spaghetti_summary(outdir_s, series_df, "HR_future32", age_label=age_label, age_tag=age_tag)
        plot_spaghetti_summary(outdir_s, series_df, "HR_future30", age_label=age_label, age_tag=age_tag)

        # Raw spaghetti plots (one line per enrollment)
        plot_spaghetti(outdir_s, series_df, "HR30", age_label=age_label, age_tag=age_tag)
        plot_spaghetti(outdir_s, series_df, "HR20", age_label=age_label, age_tag=age_tag)
        plot_spaghetti(outdir_s, series_df, "HR32", age_label=age_label, age_tag=age_tag)

        # Selection suite curves if present
        plot_spaghetti_summary(outdir_s, series_df, "HR_tt32", age_label=age_label, age_tag=age_tag)
        # Plot lead curves that exist (HR_lead1..)
        for c in sorted([c for c in series_df.columns if c.startswith("HR_lead")], key=lambda x: int(x.replace("HR_lead", ""))):
            plot_spaghetti_summary(outdir_s, series_df, c, age_label=age_label, age_tag=age_tag)

    # If --strata is provided, interpret --outdir as a base directory and run each stratum to <outdir>/<name>/.
    if args.strata:
        if (args.birth_year_min is not None) or (args.birth_year_max is not None):
            raise SystemExit("ERROR: Do not use --birth-year-min/--birth-year-max together with --strata.")

        base_outdir = Path(args.outdir)
        for spec in args.strata:
            parts = str(spec).split(":")
            name = parts[0].strip()
            if not name:
                raise SystemExit(f"ERROR: invalid --strata '{spec}' (missing name)")
            if len(parts) == 1 or (len(parts) >= 2 and parts[1].strip() == ""):
                # all ages (no birth-year filter); keep unknown birth years
                df_s = df
                label = name
                by_min = None
                by_max = None
            else:
                if len(parts) < 3 or parts[2].strip() == "":
                    raise SystemExit(f"ERROR: invalid --strata '{spec}' (expected name:min:max)")
                by_min = int(parts[1])
                by_max = int(parts[2])
                df_s = df[(df["birth_year"] >= by_min) & (df["birth_year"] <= by_max)].copy()
                label = f"{name} [{by_min},{by_max}]"
            run_one_stratum(df_s, base_outdir / name, label, birth_year_min=by_min, birth_year_max=by_max)
        print("Done.", flush=True)
        end_local = datetime.now().astimezone()
        print(f"[build_weekly_emulation] end_local={end_local.isoformat()}", flush=True)
        print(f"[build_weekly_emulation] elapsed_s={time.perf_counter() - t0:.3f}", flush=True)
        return 0

    # Legacy single-stratum behavior (optional birth-year filter via args)
    if (args.birth_year_min is None) ^ (args.birth_year_max is None):
        raise SystemExit("ERROR: Provide both --birth-year-min and --birth-year-max, or neither (for all ages).")
    if args.birth_year_min is not None and args.birth_year_max is not None:
        df = df[(df["birth_year"] >= args.birth_year_min) & (df["birth_year"] <= args.birth_year_max)].copy()
        print(f"After birth-year filter [{args.birth_year_min},{args.birth_year_max}]: {len(df):,} rows", flush=True)
        stratum_label = f"single [{args.birth_year_min},{args.birth_year_max}]"
    else:
        print(f"Birth-year filter: disabled (all ages; includes unknown birth years)", flush=True)
        stratum_label = "single (all ages)"

    outdir = Path(args.outdir)
    run_one_stratum(
        df,
        outdir,
        stratum_label,
        birth_year_min=args.birth_year_min,
        birth_year_max=args.birth_year_max,
    )
    print("Done.", flush=True)
    end_local = datetime.now().astimezone()
    print(f"[build_weekly_emulation] end_local={end_local.isoformat()}", flush=True)
    print(f"[build_weekly_emulation] elapsed_s={time.perf_counter() - t0:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


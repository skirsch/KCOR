"""
Weekly vaccine coverage by enrollment age bin (descriptive, full population).

``coverage_ge1`` is the share of people (one row per ``ID``) in each **enrollment** ``age_bin``
who had at least one dose on or before that week's Monday, among those **alive at the start of
that week** (same convention as ``metrics._pop_counts_vectorized`` / ``death_monday_allcause``).

This is **not** a cohort VE metric and does not use dose0/dose1/dose2 enrollment strata—it only
describes how vaccination rolled out by age to contextualize wave-period mortality/CFR patterns.

Age bins are **fixed at enrollment** (from ``cohort_builder.build_enrollment_table``); people are
not re-aged by calendar week.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cohort_builder import iter_followup_mondays, monday_to_iso_week
from metrics import (
    _last_alive_week_index,
    _pop_counts_vectorized,
    _sub_one_row_per_person,
    _week_index_map,
    iso_weeks_in_period,
)


def _first_week_vaccinated_index(
    dose_monday: date | float | Any,
    weeks: list[date],
    wmap: dict[date, int],
) -> int:
    """
    Smallest week index ``j`` such that ``dose_monday <= weeks[j]`` (dose on or before that Monday).

    Returns ``len(weeks)`` if never vaccinated in the window (missing dose or dose after horizon).
    """
    n = len(weeks)
    if n == 0:
        return 0
    if dose_monday is None:
        return n
    try:
        if pd.isna(dose_monday):
            return n
    except (TypeError, ValueError):
        return n
    if not isinstance(dose_monday, date):
        return n
    if dose_monday < weeks[0]:
        return 0
    if dose_monday > weeks[-1]:
        return n
    j = wmap.get(dose_monday)
    if j is not None:
        return j
    for j in range(n):
        if weeks[j] >= dose_monday:
            return j
    return n


def _vaccinated_counts_vectorized(
    sub: pd.DataFrame,
    weeks: list[date],
    wmap: dict[date, int],
    dose_col: str,
) -> np.ndarray:
    """Count people alive at week start with ``dose_col`` on or before that week's Monday (diff + cumsum)."""
    n_weeks = len(weeks)
    if len(sub) == 0 or n_weeks == 0:
        return np.zeros(n_weeks, dtype=np.int64)
    if dose_col not in sub.columns:
        return np.zeros(n_weeks, dtype=np.int64)

    dm = sub["death_monday_allcause"]
    doses = sub[dose_col]
    arr = np.zeros(n_weeks + 1, dtype=np.int64)

    for i in range(len(sub)):
        d_death = dm.iloc[i]
        if pd.isna(d_death):
            last_idx = n_weeks - 1
        else:
            last_idx = _last_alive_week_index(d_death, weeks, wmap)
        if last_idx < 0:
            continue
        d_idx = _first_week_vaccinated_index(doses.iloc[i], weeks, wmap)
        if d_idx > last_idx or d_idx >= n_weeks:
            continue
        arr[d_idx] += 1
        if last_idx + 1 < len(arr):
            arr[last_idx + 1] -= 1

    return np.cumsum(arr[:-1])


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(len(num), np.nan, dtype=np.float64)
    m = den > 0
    out[m] = num[m].astype(np.float64) / den[m].astype(np.float64)
    return out


def build_weekly_vaccine_coverage(
    df: pd.DataFrame,
    *,
    followup_start: str,
    followup_end: str,
    age_bins_config: list[list[int]],
) -> pd.DataFrame:
    """
    One row per ``iso_week`` × ``age_bin`` (fine bins plus ``age_bin='all'``).

    Requires columns: ``ID``, ``age_bin``, ``death_monday_allcause``, ``first_dose_monday``;
    optionally ``second_dose_monday``, ``third_dose_monday`` for ge2/ge3.
    """
    weeks = list(iter_followup_mondays(followup_start, followup_end))
    if not weeks:
        return pd.DataFrame()
    wmap = _week_index_map(weeks)

    base = _sub_one_row_per_person(df.loc[df["age_bin"].notna()].copy())
    age_labels = [f"{lo}-{hi}" for lo, hi in age_bins_config]

    fine_rows: list[dict] = []
    pops_stack: list[np.ndarray] = []
    v1_stack: list[np.ndarray] = []
    v2_stack: list[np.ndarray] = []
    v3_stack: list[np.ndarray] = []

    for ab in age_labels:
        sub = base.loc[base["age_bin"] == ab]
        pop = _pop_counts_vectorized(sub, weeks, wmap)
        v1 = _vaccinated_counts_vectorized(sub, weeks, wmap, "first_dose_monday")
        v2 = (
            _vaccinated_counts_vectorized(sub, weeks, wmap, "second_dose_monday")
            if "second_dose_monday" in sub.columns
            else np.zeros(len(weeks), dtype=np.int64)
        )
        v3 = (
            _vaccinated_counts_vectorized(sub, weeks, wmap, "third_dose_monday")
            if "third_dose_monday" in sub.columns
            else np.zeros(len(weeks), dtype=np.int64)
        )
        pops_stack.append(pop)
        v1_stack.append(v1)
        v2_stack.append(v2)
        v3_stack.append(v3)

        c1 = _safe_div(v1, pop)
        c2 = _safe_div(v2, pop)
        c3 = _safe_div(v3, pop)

        for t, w in enumerate(weeks):
            iso = monday_to_iso_week(w)
            rec: dict = {
                "iso_week": iso,
                "week_monday": w.isoformat(),
                "age_bin": ab,
                "population_at_risk": int(pop[t]),
                "vaccinated_ge1": int(v1[t]),
                "coverage_ge1": float(c1[t]) if np.isfinite(c1[t]) else np.nan,
            }
            if "second_dose_monday" in base.columns:
                rec["vaccinated_ge2"] = int(v2[t])
                rec["coverage_ge2"] = float(c2[t]) if np.isfinite(c2[t]) else np.nan
            if "third_dose_monday" in base.columns:
                rec["vaccinated_ge3"] = int(v3[t])
                rec["coverage_ge3"] = float(c3[t]) if np.isfinite(c3[t]) else np.nan
            fine_rows.append(rec)

    pop_all = np.sum(np.stack(pops_stack, axis=0), axis=0) if pops_stack else np.zeros(len(weeks), dtype=np.int64)
    v1_all = np.sum(np.stack(v1_stack, axis=0), axis=0) if v1_stack else np.zeros(len(weeks), dtype=np.int64)
    v2_all = (
        np.sum(np.stack(v2_stack, axis=0), axis=0)
        if v2_stack and "second_dose_monday" in base.columns
        else np.zeros(len(weeks), dtype=np.int64)
    )
    v3_all = (
        np.sum(np.stack(v3_stack, axis=0), axis=0)
        if v3_stack and "third_dose_monday" in base.columns
        else np.zeros(len(weeks), dtype=np.int64)
    )

    c1_all = _safe_div(v1_all, pop_all)
    c2_all = _safe_div(v2_all, pop_all)
    c3_all = _safe_div(v3_all, pop_all)

    for t, w in enumerate(weeks):
        iso = monday_to_iso_week(w)
        rec = {
            "iso_week": iso,
            "week_monday": w.isoformat(),
            "age_bin": "all",
            "population_at_risk": int(pop_all[t]),
            "vaccinated_ge1": int(v1_all[t]),
            "coverage_ge1": float(c1_all[t]) if np.isfinite(c1_all[t]) else np.nan,
        }
        if "second_dose_monday" in base.columns:
            rec["vaccinated_ge2"] = int(v2_all[t])
            rec["coverage_ge2"] = float(c2_all[t]) if np.isfinite(c2_all[t]) else np.nan
        if "third_dose_monday" in base.columns:
            rec["vaccinated_ge3"] = int(v3_all[t])
            rec["coverage_ge3"] = float(c3_all[t]) if np.isfinite(c3_all[t]) else np.nan
        fine_rows.append(rec)

    return pd.DataFrame(fine_rows)


def _first_iso_above(
    sub: pd.DataFrame,
    col: str,
    threshold: float,
) -> str | None:
    s = sub.sort_values("iso_week", kind="mergesort")
    hit = s[s[col] > threshold]
    if hit.empty:
        return None
    return str(hit.iloc[0]["iso_week"])


def build_vaccine_coverage_summary(
    coverage_weekly: pd.DataFrame,
    *,
    wave: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> pd.DataFrame:
    """Per ``age_bin``: first ISO week crossing 10% / 50% / 80% coverage_ge1; mean coverage in wave/baseline."""
    if coverage_weekly.empty:
        return pd.DataFrame()

    wave_iso = iso_weeks_in_period(str(wave["start"]), str(wave["end"]))
    base_iso = iso_weeks_in_period(str(baseline["start"]), str(baseline["end"]))

    rows: list[dict] = []
    for ab in sorted(coverage_weekly["age_bin"].unique(), key=lambda x: (x != "all", str(x))):
        sub = coverage_weekly[coverage_weekly["age_bin"] == ab]
        row: dict = {
            "age_bin": ab,
            "first_iso_week_coverage_ge1_gt_0.1": _first_iso_above(sub, "coverage_ge1", 0.1),
            "first_iso_week_coverage_ge1_gt_0.5": _first_iso_above(sub, "coverage_ge1", 0.5),
            "first_iso_week_coverage_ge1_gt_0.8": _first_iso_above(sub, "coverage_ge1", 0.8),
        }
        wsub = sub[sub["iso_week"].isin(wave_iso)]
        bsub = sub[sub["iso_week"].isin(base_iso)]
        row["mean_coverage_ge1_wave"] = (
            float(wsub["coverage_ge1"].mean()) if len(wsub) and wsub["coverage_ge1"].notna().any() else np.nan
        )
        row["mean_coverage_ge1_baseline"] = (
            float(bsub["coverage_ge1"].mean()) if len(bsub) and bsub["coverage_ge1"].notna().any() else np.nan
        )
        rows.append(row)

    return pd.DataFrame(rows)


def log_vaccine_coverage_console(
    coverage_weekly: pd.DataFrame,
    log: Any,
    *,
    wave_start: str | None = None,
    baseline_start: str | None = None,
) -> None:
    """Max ``coverage_ge1`` by ``age_bin``; optional snapshot at key ISO weeks."""
    if coverage_weekly.empty:
        return
    log("vaccine coverage (descriptive, enrollment age_bin): max coverage_ge1 by age_bin …")
    for ab in sorted(coverage_weekly["age_bin"].unique(), key=lambda x: (x != "all", str(x))):
        sub = coverage_weekly[coverage_weekly["age_bin"] == ab]
        v = sub["coverage_ge1"]
        if not v.notna().any():
            continue
        im = v.idxmax()
        mx = float(sub.loc[im, "coverage_ge1"])
        wk = sub.loc[im, "iso_week"]
        log(f"  {ab}: max coverage_ge1={mx:.4f} at {wk}")
    if wave_start:
        snap = coverage_weekly[coverage_weekly["iso_week"] == wave_start]
        if not snap.empty:
            log(f"  snapshot wave start {wave_start}: " + ", ".join(f"{r['age_bin']}={r['coverage_ge1']:.4f}" for _, r in snap.iterrows() if pd.notna(r["coverage_ge1"])))
    if baseline_start:
        snap = coverage_weekly[coverage_weekly["iso_week"] == baseline_start]
        if not snap.empty:
            log(
                "  snapshot baseline start "
                + baseline_start
                + ": "
                + ", ".join(
                    f"{r['age_bin']}={r['coverage_ge1']:.4f}"
                    for _, r in snap.iterrows()
                    if pd.notna(r["coverage_ge1"])
                )
            )

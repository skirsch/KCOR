"""Orchestration: stability checks, KM table, time-since-dose weekly slice."""

from __future__ import annotations

import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd

from cohort_builder import cohort_mask, iter_followup_mondays, monday_to_iso_week, weeks_between
from metrics import build_weekly_metrics


def stability_check_quiet_period(
    weekly: pd.DataFrame,
    *,
    baseline_start: str,
    baseline_end: str,
    cohort: str = "dose0",
    age_bin: str = "all",
) -> None:
    sub = weekly[
        (weekly["cohort"] == cohort) & (weekly["age_bin"] == age_bin)
    ].copy()
    base_weeks = {monday_to_iso_week(d) for d in iter_followup_mondays(baseline_start, baseline_end)}

    quiet = sub[sub["iso_week"].isin(base_weeks)]
    if len(quiet) == 0:
        mean_cfr = float("nan")
        mean_case = float("nan")
    else:
        mean_cfr = float(quiet["cfr_covid"].mean(skipna=True))
        mean_case = float(quiet["case_rate"].mean(skipna=True))
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[cfr {ts}] [stability] Quiet {baseline_start}–{baseline_end} ({cohort}, {age_bin}): "
        f"mean case_rate={mean_case:.6f}, mean CFR_covid={mean_cfr:.6f}"
    )


def build_km_post_infection_table(
    df: pd.DataFrame,
    *,
    followup_end: str,
    cohorts: list[str],
) -> tuple[pd.DataFrame, str]:
    """
    Per cohort KM from first infection to all-cause death; censored at follow-up end.
    Returns (table, skip_reason). skip_reason is empty string on success.
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return pd.DataFrame(), "lifelines is not installed (pip install lifelines or use make install)"

    end_monday = iter_followup_mondays(followup_end, followup_end)[0]
    censor_horizon = end_monday + timedelta(days=6)

    rows = []
    timelines = np.arange(0, 41, 0.5)
    any_infected = False
    for cohort in cohorts:
        m = cohort_mask(df, cohort) & df["infection_monday"].notna()
        sub = df[m].copy()
        if len(sub) == 0:
            continue
        any_infected = True
        t_obs = []
        e_obs = []
        for _, row in sub.iterrows():
            inf = row["infection_monday"]
            dth = row["death_monday"]
            if pd.isna(inf):
                continue
            if pd.notna(dth) and dth >= inf:
                w = weeks_between(inf, dth)
                if w < 0:
                    continue
                t_obs.append(float(w))
                e_obs.append(1)
            else:
                w = weeks_between(inf, censor_horizon)
                t_obs.append(float(max(w, 0)))
                e_obs.append(0)
        if not t_obs:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(np.array(t_obs), np.array(e_obs), label=cohort)
        for tw in timelines:
            try:
                s = float(kmf.predict(tw))
            except Exception:
                s = np.nan
            rows.append({"cohort": cohort, "timeline": tw, "KM_estimate": s})
    if not rows:
        if not any_infected:
            return (
                pd.DataFrame(),
                "no infected persons in requested cohorts after filters (infection_monday missing for everyone)",
            )
        return pd.DataFrame(), "KM fit produced no rows (no valid event/censor times)"
    return pd.DataFrame(rows), ""


def run_time_since_dose2_analysis(
    df: pd.DataFrame,
    *,
    bins: list[list[int]],
    followup_start: str,
    followup_end: str,
    age_bins_config: list[list[int]],
    baseline_start: str,
    baseline_end: str,
    wave_start: str,
    wave_end: str,
    reference_cohort: str = "dose0",
    metrics_workers: int = 1,
) -> pd.DataFrame:
    w = df["weeks_second_dose_to_enrollment"]
    masks: dict[str, pd.Series] = {"dose0": df["cohort_dose0"]}
    cohort_names = ["dose0"]
    for lo, hi in bins:
        name = f"dose2_ts_{lo}_{hi}"
        m = df["cohort_dose2"] & w.notna() & (w >= lo) & (w <= hi)
        if m.any():
            masks[name] = m
            cohort_names.append(name)
    weekly_ts, _ = build_weekly_metrics(
        df,
        followup_start=followup_start,
        followup_end=followup_end,
        cohorts=cohort_names,
        age_bins_config=age_bins_config,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        wave_start=wave_start,
        wave_end=wave_end,
        cohort_masks=masks,
        reference_cohort=reference_cohort,
        metrics_workers=metrics_workers,
    )
    return weekly_ts

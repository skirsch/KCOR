"""Console QA: follow-up aggregates and optional spot-check vs external death tables."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date

import numpy as np
import pandas as pd

from cohort_builder import (
    PRIMARY_ENROLLMENT_COHORTS,
    iter_followup_mondays,
    monday_to_iso_week,
)


def _death_monday_iso(d: object) -> str | None:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    try:
        if pd.isna(d):
            return None
    except (ValueError, TypeError):
        return None
    if isinstance(d, date):
        return monday_to_iso_week(d)
    return None


def _person_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ID in study (dose0 | dose1 | dose2 enrollment strata only; not dose3)."""
    m_study = df["cohort_dose0"] | df["cohort_dose1"] | df["cohort_dose2"]
    sub = df[m_study]
    if sub.empty:
        return pd.DataFrame()
    return sub.groupby("ID", sort=False).agg(
        death_monday=("death_monday", "first"),
        death_monday_allcause=("death_monday_allcause", "first"),
        covid_death_monday=("covid_death_monday", "first"),
        birth_band_start=("birth_band_start", "first"),
        cohort_dose0=("cohort_dose0", "max"),
        cohort_dose1=("cohort_dose1", "max"),
        cohort_dose2=("cohort_dose2", "max"),
    ).assign(
        vaxxed_enrollment=lambda x: x["cohort_dose1"] | x["cohort_dose2"],
    )


def _followup_iso_weeks(followup_start: str, followup_end: str) -> set[str]:
    return {monday_to_iso_week(w) for w in iter_followup_mondays(followup_start, followup_end)}


def _infection_episode_counts(df: pd.DataFrame, follow_iso: set[str]) -> tuple[int, int, int]:
    """(all, unvaxxed, vaxxed) = rows with infection_monday in follow-up ISO weeks (dose0–2 only)."""
    m_study = df["cohort_dose0"] | df["cohort_dose1"] | df["cohort_dose2"]
    sub = df[m_study]
    iso = sub["infection_monday"].map(_death_monday_iso)
    in_p = iso.isin(follow_iso)
    n_all = int(in_p.sum())
    n_u = int((in_p & sub["cohort_dose0"]).sum())
    n_v = int((in_p & (sub["cohort_dose1"] | sub["cohort_dose2"])).sum())
    return n_all, n_u, n_v


def _person_death_counts(
    pers: pd.DataFrame,
    follow_iso: set[str],
    mask: pd.Series,
) -> tuple[int, int, int]:
    """
    Distinct persons in ``mask`` for ISO week set ``follow_iso``:

    * **COVID deaths** — ``covid_death_monday`` falls in ``follow_iso``.
    * **ACM deaths** — **union**: ``DateOfDeath`` **or** ``Date_COVID_death`` week is in ``follow_iso``.
      (Coalescing LPZ-first can put ACM in a different week than the COVID register; union keeps
      ``covid_deaths <= acm_deaths``.)
    * **Non-COVID** — ACM-in-window by union but COVID week not in ``follow_iso``.
    """
    p = pers.loc[mask]
    if len(p) == 0:
        return 0, 0, 0
    lpz_iso = p["death_monday"].map(_death_monday_iso)
    cv_iso = p["covid_death_monday"].map(_death_monday_iso)
    in_cv = cv_iso.isin(follow_iso)
    in_acm = lpz_iso.isin(follow_iso) | in_cv
    n_cv = int(in_cv.sum())
    n_acm = int(in_acm.sum())
    n_noncov = int((in_acm & ~in_cv).sum())
    return n_cv, n_acm, n_noncov


def _person_table_all_ids(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct ``ID`` (full extract after enrollment) for file-wide death QA."""
    if df.empty or "ID" not in df.columns:
        return pd.DataFrame()
    return df.groupby("ID", sort=False).agg(
        death_monday=("death_monday", "first"),
        covid_death_monday=("covid_death_monday", "first"),
    )


def _spot_week_metrics(
    p: pd.DataFrame,
    spot_set: set[str],
    mask: pd.Series,
) -> tuple[int, int, int, int]:
    """
    Single ISO-week window, distinct persons in ``mask``:

    * ``covid_week`` — ``Date_COVID_death`` / Umrti column week in ``spot_set``.
    * ``acm_lpz_week`` — ``DateOfDeath`` / DatumUmrtiLPZ week in ``spot_set`` (usual ``514`` total).
    * ``acm_union`` — LPZ or COVID week in ``spot_set`` (sanity; not compared to YAML by default).
    * ``noncov_lpz_week`` — LPZ week in ``spot_set`` but COVID week not in ``spot_set``.
    """
    x = p.loc[mask]
    if len(x) == 0:
        return 0, 0, 0, 0
    lpz = x["death_monday"].map(_death_monday_iso)
    cv = x["covid_death_monday"].map(_death_monday_iso)
    in_cv = cv.isin(spot_set)
    in_lpz = lpz.isin(spot_set)
    n_cv = int(in_cv.sum())
    n_lpz = int(in_lpz.sum())
    n_union = int((in_lpz | in_cv).sum())
    n_nc = int((in_lpz & ~in_cv).sum())
    return n_cv, n_lpz, n_union, n_nc


def _weekly_total_pop_by_iso(weekly: pd.DataFrame) -> pd.Series:
    """Sum ``population_at_risk`` by ISO week (``age_bin == all``, dose0–2 cohort rows only)."""
    w = weekly[
        (weekly["age_bin"] == "all") & (weekly["cohort"].isin(PRIMARY_ENROLLMENT_COHORTS))
    ]
    if w.empty:
        return pd.Series(dtype=float)
    return w.groupby("iso_week", sort=True)["population_at_risk"].sum()


def _avg_pop_at_risk_stratum(
    weekly: pd.DataFrame,
    stratum: str,
    *,
    iso_weeks: set[str] | None = None,
) -> float:
    """Mean weekly population at risk; vaccinated = total − dose0 (same week). Optional ``iso_weeks`` restricts averaging."""
    w = weekly[weekly["age_bin"] == "all"]
    if w.empty:
        return float("nan")
    by_iso = _weekly_total_pop_by_iso(weekly)
    if by_iso.empty:
        return float("nan")
    if iso_weeks is not None:
        by_iso = by_iso[by_iso.index.isin(iso_weeks)]
        if by_iso.empty:
            return float("nan")
    if stratum == "everyone":
        return float(by_iso.mean())
    wd0 = w[w["cohort"] == "dose0"].groupby("iso_week", sort=True)["population_at_risk"].sum()
    if iso_weeks is not None:
        wd0 = wd0[wd0.index.isin(iso_weeks)]
    d0 = wd0.reindex(by_iso.index, fill_value=0)
    if stratum == "unvaxxed":
        return float(d0.mean())
    if stratum == "vaxxed":
        return float((by_iso - d0).mean())
    return float("nan")


def _weekly_age_all_cohort_sets(weekly: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    """Cohort name sets (dose0–2 only) for summing ``weekly_metrics`` with ``age_bin == all``."""
    w = weekly[weekly["age_bin"] == "all"]
    cohorts = set(w["cohort"].unique()) & PRIMARY_ENROLLMENT_COHORTS
    everyone = cohorts
    unvaxxed = {"dose0"} & everyone
    vaxxed = everyone - {"dose0"}
    return everyone, unvaxxed, vaxxed


def _period_allcause_deaths_per_person_week(
    weekly: pd.DataFrame,
    cohort: str,
    *,
    iso_weeks: set[str],
) -> tuple[float, int, int]:
    """
    Over ``iso_weeks`` only: ``sum(deaths_all) / sum(population_at_risk)`` for one cohort
    (``age_bin == all``). Matches “3 deaths on 100 at risk + 2 on 90 → (3+2)/(100+90)”.
    """
    if cohort not in PRIMARY_ENROLLMENT_COHORTS:
        return float("nan"), 0, 0
    sub = weekly[
        (weekly["age_bin"] == "all")
        & (weekly["cohort"] == cohort)
        & (weekly["iso_week"].isin(iso_weeks))
    ]
    if sub.empty:
        return float("nan"), 0, 0
    tot_d = int(sub["deaths_all"].sum())
    tot_p = int(sub["population_at_risk"].sum())
    rate = tot_d / tot_p if tot_p > 0 else float("nan")
    return rate, tot_d, tot_p


def _weekly_aggregate_case_rate_cfr(
    weekly: pd.DataFrame,
    cohorts: set[str],
    *,
    iso_weeks: set[str] | None = None,
) -> tuple[float, float]:
    """
    ``age_bin == all``: case_rate = sum(cases)/sum(population_at_risk);
    episode CFR = sum(cfr_covid * cases) / sum(cases). Optional ``iso_weeks`` limits rows.
    """
    if not cohorts:
        return float("nan"), float("nan")
    use = cohorts & PRIMARY_ENROLLMENT_COHORTS
    sub = weekly[(weekly["age_bin"] == "all") & (weekly["cohort"].isin(use))]
    if iso_weeks is not None:
        sub = sub[sub["iso_week"].isin(iso_weeks)]
    if sub.empty:
        return float("nan"), float("nan")
    c = sub["cases"].to_numpy(dtype=float)
    p = sub["population_at_risk"].to_numpy(dtype=float)
    f = sub["cfr_covid"].to_numpy(dtype=float)
    tot_c = float(np.nansum(c))
    tot_p = float(np.nansum(p))
    tot_num = float(np.nansum(np.where((c > 0) & np.isfinite(f), f * c, 0.0)))
    case_rate = tot_c / tot_p if tot_p > 0 else float("nan")
    cfr = tot_num / tot_c if tot_c > 0 else float("nan")
    return case_rate, cfr


def _birth_year_eq(series: pd.Series, year: int) -> pd.Series:
    def _one(v: object) -> bool:
        if pd.isna(v):
            return False
        try:
            return int(float(v)) == year
        except (TypeError, ValueError):
            return False

    return series.map(_one)


def log_qa_summary(
    df: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    cohort_followup_start: str,
    cohort_followup_end: str,
    qa_period_start: str,
    qa_period_end: str,
    log: Callable[[str], None],
    qa_cfg: Mapping[str, object] | None = None,
    df_enrollment_all: pd.DataFrame | None = None,
) -> None:
    """
    Log QA for the **period of interest** (``qa_period_start``–``qa_period_end``): infections, deaths, rates.

    Cohort ``population_at_risk`` in ``weekly_metrics`` is built from enrollment
    (``cohort_followup_start``–``cohort_followup_end``); the table below aggregates infections
    and deaths only over the QA ISO-week window.

    * Unvaxxed = enrollment ``dose0``; vaxxed = ``dose1`` | ``dose2`` (dose3 stratum not used).
    * Deaths: distinct ``ID``; ACM-in-window = LPZ or COVID week union for period totals.
    * Avg pop / case_rate / CFR = mean or sums over **QA period weeks** only (``age_bin == all``).
    """
    qa_iso = _followup_iso_weeks(qa_period_start, qa_period_end)
    n_qa = len(qa_iso)

    log(
        f"QA summary — period of interest {qa_period_start}–{qa_period_end} ({n_qa} ISO weeks): "
        "infections, COVID/ACM deaths, avg_pop_at_risk, case_rate, cfr_covid use this window only. "
        f"Weekly cohort tracking (alive per week) runs from enrollment follow-up "
        f"{cohort_followup_start}–{cohort_followup_end}."
    )
    log(
        "  infections=positive-test rows in window; deaths=distinct persons; "
        "non-COVID=died in window (LPZ or COVID week) but COVID week not in window; "
        "unvaxxed=dose0, vaxxed=dose1|dose2."
    )

    inf_a, inf_u, inf_v = _infection_episode_counts(df, qa_iso)
    pers = _person_table(df)
    ev_coh, u_coh, v_coh = _weekly_age_all_cohort_sets(weekly)

    rows: list[tuple[str, int, int, int, int, float, float, float]] = []
    if len(pers) == 0:
        log("QA summary: no persons in enrollment cohorts (unexpected).")
        return

    m_all = pd.Series(True, index=pers.index)
    m_u = pers["cohort_dose0"].astype(bool)
    m_v = pers["vaxxed_enrollment"].astype(bool)

    for label, m, pop_key, cset in (
        ("everyone", m_all, "everyone", ev_coh),
        ("unvaxxed (dose0)", m_u, "unvaxxed", u_coh),
        ("vaxxed (dose1+)", m_v, "vaxxed", v_coh),
    ):
        n_cv, n_acm, n_nc = _person_death_counts(pers, qa_iso, m)
        if label == "everyone":
            n_inf = inf_a
        elif label == "unvaxxed (dose0)":
            n_inf = inf_u
        else:
            n_inf = inf_v
        avg_pop = _avg_pop_at_risk_stratum(weekly, pop_key, iso_weeks=qa_iso)
        cr, cfr = _weekly_aggregate_case_rate_cfr(weekly, cset, iso_weeks=qa_iso)
        rows.append((label, n_inf, n_cv, n_nc, n_acm, avg_pop, cr, cfr))

    col_w = max(len(r[0]) for r in rows)
    log(
        f"{'stratum'.ljust(col_w)}  infections  covid_deaths  noncovid_deaths  acm_deaths  "
        f"avg_pop_at_risk   case_rate   cfr_covid"
    )
    for label, n_inf, n_cv, n_nc, n_acm, avg_pop, cr, cfr in rows:
        ap = f"{avg_pop:,.1f}" if np.isfinite(avg_pop) else "n/a"
        crs = f"{cr:.6f}" if np.isfinite(cr) else "n/a"
        cfrs = f"{cfr:.6f}" if np.isfinite(cfr) else "n/a"
        log(
            f"{label.ljust(col_w)}  {n_inf:>10,}  {n_cv:>12,}  {n_nc:>15,}  {n_acm:>10,}  {ap:>15}  {crs:>10}  {cfrs:>10}"
        )

    log(
        f"QA period all-cause mortality (weekly_metrics): sum(deaths_all) / sum(population_at_risk) "
        f"over {qa_period_start}–{qa_period_end}, age_bin=all (person-weeks = weekly denominators)."
    )
    for c in ("dose0", "dose1", "dose2"):
        r, td, tp = _period_allcause_deaths_per_person_week(weekly, c, iso_weeks=qa_iso)
        rs = f"{r:.8f}" if np.isfinite(r) else "n/a"
        log(f"  {c}: deaths={td:,}  person_weeks={tp:,}  deaths/person_week={rs}")

    if not qa_cfg:
        return

    spot = qa_cfg.get("spot_check")
    if not spot or not isinstance(spot, dict):
        return

    iso_w = str(spot.get("iso_week", "")).strip()
    if not iso_w:
        return

    if df_enrollment_all is None or len(df_enrollment_all) == 0:
        log("QA spot-check: skip (no df_enrollment_all — need full post-enrollment frame)")
        return

    spot_set = {iso_w}
    p_all = _person_table_all_ids(df_enrollment_all)
    m_all = pd.Series(True, index=p_all.index)
    n_cv, n_lpz, n_union, n_nc = _spot_week_metrics(p_all, spot_set, m_all)

    log(
        f"QA spot-check — ISO week {iso_w}, all distinct IDs in extract after enrollment "
        "(same load/mRNA filters as pipeline)"
    )
    log(f"  distinct IDs: {len(p_all):,}")
    log(
        f"  total covid_week (Umrti→Date_COVID_death)={n_cv}  "
        f"total acm_lpz_week (DatumUmrtiLPZ→DateOfDeath)={n_lpz}  "
        f"acm_union={n_union}  noncov_lpz_week={n_nc}"
    )

    exp = spot.get("expected")
    if not exp or not isinstance(exp, dict):
        return

    def _get_covid_acm(d: Mapping[str, object]) -> tuple[int | None, int | None]:
        ev = ea = None
        for k in ("covid_deaths", "all_covid"):
            if k not in d or d[k] is None:
                continue
            try:
                ev = int(d[k])
                break
            except (TypeError, ValueError):
                continue
        for k in ("acm_deaths", "all_acm"):
            if k not in d or d[k] is None:
                continue
            try:
                ea = int(d[k])
                break
            except (TypeError, ValueError):
                continue
        return ev, ea

    ev_cov, ev_acm = _get_covid_acm(exp)
    checks: list[tuple[str, int, int]] = []
    if ev_cov is not None:
        checks.append(("total covid_week", n_cv, ev_cov))
    if ev_acm is not None:
        checks.append(("total acm_lpz_week", n_lpz, ev_acm))
    if not checks:
        return
    bad = [f"{nm}: got {g} expected {e}" for nm, g, e in checks if g != e]
    if bad:
        log("QA spot-check expected: FAIL - " + "; ".join(bad))
    else:
        log("QA spot-check expected: OK (covid_deaths & acm_deaths match)")

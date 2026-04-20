"""Console QA: follow-up aggregates and optional spot-check vs external death tables."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_builder import (
    PRIMARY_ENROLLMENT_COHORTS,
    iso_week_str_to_monday,
    iter_followup_mondays,
    monday_to_iso_week,
)

from metrics import _compute_weekly_stratum_rows, _week_index_map


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


def _infection_episode_counts_dose012(df: pd.DataFrame, follow_iso: set[str]) -> tuple[int, int, int, int]:
    """(all, dose0, dose1, dose2) = infection rows in window (enrollment strata only)."""
    m_study = df["cohort_dose0"] | df["cohort_dose1"] | df["cohort_dose2"]
    sub = df[m_study]
    iso = sub["infection_monday"].map(_death_monday_iso)
    in_p = iso.isin(follow_iso)
    n0 = int((in_p & sub["cohort_dose0"]).sum())
    n1 = int((in_p & sub["cohort_dose1"]).sum())
    n2 = int((in_p & sub["cohort_dose2"]).sum())
    return int(in_p.sum()), n0, n1, n2


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
    """One row per distinct ``ID`` (full extract after enrollment) for death QA / spot-check."""
    if df.empty or "ID" not in df.columns:
        return pd.DataFrame()
    agg: dict[str, tuple[str, str]] = {
        "death_monday": ("death_monday", "first"),
        "covid_death_monday": ("covid_death_monday", "first"),
    }
    if "birth_band_start" in df.columns:
        agg["birth_band_start"] = ("birth_band_start", "first")
    return df.groupby("ID", sort=False).agg(**agg)


def _birth_year_range_mask(series: pd.Series, lo: int, hi: int) -> pd.Series:
    """True where ``birth_band_start`` parses to an integer year in ``[lo, hi]``; NaN / bad → False."""

    def _ok(v: object) -> bool:
        if v is None:
            return False
        try:
            if pd.isna(v):
                return False
        except (ValueError, TypeError):
            return False
        try:
            y = int(float(v))
        except (TypeError, ValueError):
            return False
        return lo <= y <= hi

    return series.map(_ok)


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
    if stratum in PRIMARY_ENROLLMENT_COHORTS:
        wc = w[w["cohort"] == stratum].groupby("iso_week", sort=True)["population_at_risk"].sum()
        if iso_weeks is not None:
            wc = wc[wc.index.isin(iso_weeks)]
        if wc.empty:
            return float("nan")
        return float(wc.mean())
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


def _period_covid_deaths_per_person_week(
    weekly: pd.DataFrame,
    cohort: str,
    *,
    iso_weeks: set[str],
) -> tuple[float, int, int]:
    """
    Over ``iso_weeks`` only: ``sum(deaths_covid) / sum(population_at_risk)`` for one cohort
    (``age_bin == all``; weekly ``deaths_covid`` from ``covid_death_monday``).
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
    tot_d = int(sub["deaths_covid"].sum())
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


def _mini_weekly_all_cohorts(
    df_slice: pd.DataFrame,
    *,
    followup_start: str,
    followup_end: str,
    cohorts: list[str],
) -> pd.DataFrame:
    """Rebuild ``age_bin == all`` weekly rows for ``cohorts`` on a person subset (e.g. birth-year slice)."""
    weeks = list(iter_followup_mondays(followup_start, followup_end))
    wmap = _week_index_map(weeks)
    iso_labels = [monday_to_iso_week(w) for w in weeks]
    parts: list[dict] = []
    for co in cohorts:
        parts.extend(
            _compute_weekly_stratum_rows(
                df_slice,
                co,
                "all",
                weeks=weeks,
                wmap=wmap,
                iso_labels=iso_labels,
                cohort_masks=None,
            )
        )
    return pd.DataFrame(parts)


_DOSE_STR_TO_INT = {"dose0": 0, "dose1": 1, "dose2": 2, "dose3": 3}


def write_debug_birth_cohort_weekly_csv(
    df_model: pd.DataFrame,
    *,
    followup_start: str,
    followup_end: str,
    birth_year_min: int,
    birth_year_max: int,
    cohorts: list[str],
    out_path: Path,
    log: Callable[[str], None] | None = None,
) -> bool:
    """
    One row per ISO week × enrollment stratum: week-start **date**, numeric **dose** (0–3),
    **alive_at_week_start**, **died_in_week** (all-cause / ``deaths_all``),
    **covid_deaths_in_week** (``deaths_covid`` / ``covid_death_monday`` histogram).
    Birth-year slice only — for reconciling QA vs external death tables.
    """
    _lg = log or (lambda _m: None)
    if "birth_band_start" not in df_model.columns:
        _lg("debug enrollment weekly CSV: skip (no birth_band_start)")
        return False
    m_birth = _birth_year_range_mask(df_model["birth_band_start"], birth_year_min, birth_year_max)
    df_slice = df_model.loc[m_birth]
    if df_slice.empty:
        _lg(
            f"debug enrollment weekly CSV: skip (no rows with birth {birth_year_min}–{birth_year_max})"
        )
        return False
    use_cohorts = [c for c in ("dose0", "dose1", "dose2") if c in cohorts and c in PRIMARY_ENROLLMENT_COHORTS]
    if not use_cohorts:
        _lg("debug enrollment weekly CSV: skip (no dose0–2 cohorts in cfg)")
        return False
    mini = _mini_weekly_all_cohorts(
        df_slice,
        followup_start=followup_start,
        followup_end=followup_end,
        cohorts=use_cohorts,
    )
    if mini.empty:
        _lg("debug enrollment weekly CSV: skip (empty weekly slice)")
        return False
    dose_num = mini["cohort"].map(lambda c: _DOSE_STR_TO_INT.get(str(c), pd.NA))
    out = mini.assign(
        date=mini["week_monday"],
        dose=dose_num,
        alive_at_week_start=mini["population_at_risk"],
        died_in_week=mini["deaths_all"],
        covid_deaths_in_week=mini["deaths_covid"],
    )[
        [
            "date",
            "dose",
            "alive_at_week_start",
            "died_in_week",
            "covid_deaths_in_week",
        ]
    ]
    out = out.sort_values(["date", "dose"], kind="mergesort").reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    _lg(
        f"debug enrollment weekly CSV: wrote {len(out):,} rows "
        f"(birth {birth_year_min}–{birth_year_max}, follow-up {followup_start}–{followup_end}) → {out_path}"
    )
    return True


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

    * Table strata: ``everyone`` then ``dose0`` / ``dose1`` / ``dose2`` at enrollment (dose3 not used).
    * Deaths: distinct ``ID``; ACM-in-window = LPZ or COVID week union for period totals.
    * Avg pop / case_rate / CFR = mean or sums over **QA period weeks** only (``age_bin == all``).

    If ``qa_cfg`` sets ``period_birth_year_min`` / ``period_birth_year_max``, the QA table and
    weekly-derived columns use only rows with ``birth_band_start`` in that inclusive range; the
    spot-check block is unchanged (its own birth window).
    """
    qa_iso = _followup_iso_weeks(qa_period_start, qa_period_end)
    n_qa = len(qa_iso)

    p_lo = qa_cfg.get("period_birth_year_min") if qa_cfg else None
    p_hi = qa_cfg.get("period_birth_year_max") if qa_cfg else None
    use_period_birth = p_lo is not None and p_hi is not None
    if use_period_birth:
        try:
            pb_lo, pb_hi = int(p_lo), int(p_hi)
        except (TypeError, ValueError):
            use_period_birth = False
            log("QA summary: invalid period_birth_year_min/max — using all birth years for QA table")
    if use_period_birth and "birth_band_start" not in df.columns:
        use_period_birth = False
        log("QA summary: no birth_band_start column — using all birth years for QA table")

    if use_period_birth:
        df_period = df.loc[_birth_year_range_mask(df["birth_band_start"], pb_lo, pb_hi)].copy()
        birth_note = f"birth years {pb_lo}–{pb_hi} only (birth_band_start)"
    else:
        df_period = df
        birth_note = "all birth years"

    log(
        f"QA summary — period of interest {qa_period_start}–{qa_period_end} ({n_qa} ISO weeks), "
        f"{birth_note}: infections, COVID/ACM deaths, avg_pop_at_risk, case_rate, cfr_covid use this window only. "
        f"Weekly cohort tracking (alive per week) runs from enrollment follow-up "
        f"{cohort_followup_start}–{cohort_followup_end}."
    )
    log(
        "  infections=positive-test rows in window; deaths=distinct persons; "
        "non-COVID=died in window (LPZ or COVID week) but COVID week not in window; "
        "strata=dose0|dose1|dose2 at enrollment."
    )

    inf_all, inf_d0, inf_d1, inf_d2 = _infection_episode_counts_dose012(df_period, qa_iso)
    pers = _person_table(df_period)
    ev_coh, _, _ = _weekly_age_all_cohort_sets(weekly)
    cohorts_for_slice = [c for c in ("dose0", "dose1", "dose2") if c in ev_coh]
    if use_period_birth and len(cohorts_for_slice):
        weekly_for_qa = _mini_weekly_all_cohorts(
            df_period,
            followup_start=cohort_followup_start,
            followup_end=cohort_followup_end,
            cohorts=cohorts_for_slice,
        )
    else:
        weekly_for_qa = weekly

    rows: list[tuple[str, int, int, int, int, float, float, float]] = []
    if len(pers) == 0:
        log("QA summary: no persons in enrollment cohorts for this slice (unexpected).")
        return

    m_all = pd.Series(True, index=pers.index)

    for label, m, pop_key, cset, n_inf in (
        ("everyone", m_all, "everyone", ev_coh, inf_all),
        ("dose0", pers["cohort_dose0"].astype(bool), "dose0", {"dose0"}, inf_d0),
        ("dose1", pers["cohort_dose1"].astype(bool), "dose1", {"dose1"}, inf_d1),
        ("dose2", pers["cohort_dose2"].astype(bool), "dose2", {"dose2"}, inf_d2),
    ):
        n_cv, n_acm, n_nc = _person_death_counts(pers, qa_iso, m)
        avg_pop = _avg_pop_at_risk_stratum(weekly_for_qa, pop_key, iso_weeks=qa_iso)
        cr, cfr = _weekly_aggregate_case_rate_cfr(weekly_for_qa, cset, iso_weeks=qa_iso)
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
        f"QA period COVID mortality ({birth_note}): sum(deaths_covid) / sum(population_at_risk) "
        f"over {qa_period_start}–{qa_period_end}, age_bin=all (person-weeks = weekly denominators)."
    )
    for c in ("dose0", "dose1", "dose2"):
        r, td, tp = _period_covid_deaths_per_person_week(weekly_for_qa, c, iso_weeks=qa_iso)
        rs = f"{r:.8f}" if np.isfinite(r) else "n/a"
        log(f"  {c}: covid_deaths={td:,}  person_weeks={tp:,}  covid_deaths/person_week={rs}")

    if not qa_cfg:
        return

    spot = qa_cfg.get("spot_check")
    if not spot or not isinstance(spot, dict):
        return

    iso_w_raw = str(spot.get("iso_week", "")).strip()
    if not iso_w_raw:
        return

    # Anchored QA: exactly one ISO calendar week (e.g. 2021-48), not a range.
    ts_spot = iso_week_str_to_monday(iso_w_raw)
    if pd.isna(ts_spot):
        log(
            f"QA spot-check: skip (iso_week {iso_w_raw!r} must be a single ISO week label, e.g. 2021-48)"
        )
        return
    spot_monday = ts_spot.date()
    iso_w = monday_to_iso_week(spot_monday)
    spot_set = {iso_w}

    y_lo = spot.get("birth_year_min")
    y_hi = spot.get("birth_year_max")
    if y_lo is None or y_hi is None:
        log(
            "QA spot-check: skip (requires birth_year_min and birth_year_max with iso_week — "
            "one ISO week × birth-year slice only, e.g. 1930–1939 for 193x)"
        )
        return
    try:
        y_lo_i, y_hi_i = int(y_lo), int(y_hi)
    except (TypeError, ValueError):
        log("QA spot-check: skip (invalid birth_year_min / birth_year_max)")
        return

    if df_enrollment_all is None or len(df_enrollment_all) == 0:
        log("QA spot-check: skip (no df_enrollment_all — need full post-enrollment frame)")
        return

    p_all = _person_table_all_ids(df_enrollment_all)
    if "birth_band_start" not in p_all.columns:
        log("QA spot-check: skip (no birth_band_start on frame; cannot apply birth cohort filter)")
        return
    m_cohort = _birth_year_range_mask(p_all["birth_band_start"], y_lo_i, y_hi_i)

    n_cv, n_lpz, n_union, n_nc = _spot_week_metrics(p_all, spot_set, m_cohort)

    log(
        f"QA spot-check — scope: exactly ISO week {iso_w} (week starting {spot_monday.isoformat()}), "
        f"birth years {y_lo_i}–{y_hi_i} inclusive (distinct IDs in that slice only); "
        "same load/mRNA filters as pipeline"
    )
    log(f"  distinct IDs in birth×week slice: {int(m_cohort.sum()):,}  (all IDs in table: {len(p_all):,})")
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
    scope = f"ISO {iso_w}, birth {y_lo_i}–{y_hi_i}"
    if ev_cov is not None:
        checks.append((f"covid_week [{scope}]", n_cv, ev_cov))
    if ev_acm is not None:
        checks.append((f"acm_lpz_week [{scope}]", n_lpz, ev_acm))
    if not checks:
        return
    bad = [f"{nm}: got {g} expected {e}" for nm, g, e in checks if g != e]
    if bad:
        log("QA spot-check expected: FAIL - " + "; ".join(bad))
    else:
        log(f"QA spot-check expected: OK for {scope} (covid_deaths & acm_deaths)")

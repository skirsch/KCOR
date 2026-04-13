"""Landmark all-cause Kaplan–Meier from a fixed ISO week (e.g. enrollment week 2021-24)."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_builder import iso_week_str_to_monday, iter_followup_mondays, weeks_between
from metrics import _sub_one_row_per_person

_REPO_CODE = Path(__file__).resolve().parents[3] / "code"
if str(_REPO_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_CODE))

from mfg_codes import MFG_DICT, MODERNA, OTHER, PFIZER, UNVAX  # noqa: E402

FIRST_MFG_COHORTS = ("unvax", "pfizer", "moderna", "other")
DOSE_COHORTS = ("dose0", "dose1", "dose2")


def _landmark_monday(landmark_iso_week: str) -> date:
    ts = iso_week_str_to_monday(landmark_iso_week)
    if pd.isna(ts):
        raise ValueError(f"Invalid landmark_iso_week: {landmark_iso_week!r}")
    return ts.date()


def _censor_horizon_date(followup_end_iso: str) -> date:
    end_monday = iter_followup_mondays(followup_end_iso, followup_end_iso)[0]
    return end_monday + timedelta(days=6)


def _as_date(d: object) -> date | None:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    if pd.isna(d):
        return None
    if isinstance(d, date) and not isinstance(d, pd.Timestamp):
        return d
    ts = pd.Timestamp(d)
    if pd.isna(ts):
        return None
    return ts.date()


def _min_time_event(pairs: list[tuple[float, int]]) -> tuple[float, int]:
    """Earliest time; ties break toward death (event) over censor."""
    if not pairs:
        return 0.0, 0
    return min(pairs, key=lambda x: (x[0], 0 if x[1] == 1 else 1))


def _mfg_from_code(code: object) -> str:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return UNVAX
    s = str(code).strip()
    return MFG_DICT.get(s, OTHER)


def _horizon_weeks(landmark: date, censor_d: date) -> float:
    return float(max(weeks_between(landmark, censor_d), 0))


def _assign_first_mfg_stratum(
    first_dose_monday: object,
    vaccine_code_first: object,
    landmark: date,
) -> str:
    fd = _as_date(first_dose_monday)
    if fd is None or fd > landmark:
        return "unvax"
    m = _mfg_from_code(vaccine_code_first)
    if m == PFIZER:
        return "pfizer"
    if m == MODERNA:
        return "moderna"
    return "other"


def _assign_dose_stratum(
    first_dose_monday: object,
    second_dose_monday: object,
    landmark: date,
) -> str:
    fd = _as_date(first_dose_monday)
    sd = _as_date(second_dose_monday)
    if fd is None or fd > landmark:
        return "dose0"
    if sd is None or sd > landmark:
        return "dose1"
    return "dose2"


def resolve_landmark_age_bins(km_land_cfg: dict, age_labels: list[str]) -> list[str]:
    """
    Which age_bin labels to run landmark KM for.

    * ``age_bins: all`` (string) or ``age_bins: [ "40-49", ... ]`` — explicit list or all cfg age bands.
    * ``age_bin: all`` — same as all bands.
    * ``age_bin: "70-120"`` — single band (legacy).
    """
    raw_bins = km_land_cfg.get("age_bins")
    if raw_bins is not None:
        if isinstance(raw_bins, str) and raw_bins.strip().lower() == "all":
            return list(age_labels)
        if raw_bins is True:
            return list(age_labels)
        return [str(x) for x in raw_bins]
    ab = km_land_cfg.get("age_bin", "all")
    if isinstance(ab, str) and ab.strip().lower() == "all":
        return list(age_labels)
    return [str(ab)]


def _fit_km_to_table(
    cohort_to_te: dict[str, tuple[list[float], list[int]]],
    *,
    age_bin: str,
    timeline_step: float = 0.5,
    timeline_pad_weeks: float = 6.0,
) -> tuple[pd.DataFrame, str]:
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return pd.DataFrame(), "lifelines is not installed (pip install lifelines or use make install)"

    max_t = 0.0
    for _cohort, (t_obs, e_obs) in cohort_to_te.items():
        if t_obs:
            max_t = max(max_t, float(np.max(t_obs)))
    timelines = np.arange(0.0, max_t + timeline_pad_weeks + timeline_step, timeline_step)

    rows: list[dict] = []
    for cohort, (t_obs, e_obs) in cohort_to_te.items():
        if not t_obs:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(np.array(t_obs), np.array(e_obs), label=cohort)
        for tw in timelines:
            try:
                s = float(kmf.predict(tw))
            except Exception:
                s = np.nan
            rows.append(
                {
                    "cohort": cohort,
                    "timeline": float(tw),
                    "KM_estimate": s,
                    "age_bin": age_bin,
                }
            )
    if not rows:
        return pd.DataFrame(), "KM fit produced no rows (no valid event/censor times)"
    return pd.DataFrame(rows), ""


def build_km_landmark_first_mfg_table(
    df: pd.DataFrame,
    *,
    landmark_iso_week: str,
    followup_end: str,
    age_bin: str,
) -> tuple[pd.DataFrame, str]:
    """
    Landmark KM: strata by first-dose manufacturer as of the landmark week (fixed enrollment).
    Unvaccinated at landmark = no first dose on or before landmark Monday.
    Death vs administrative censor at end of follow-up (no dose-based censoring).
    Deaths recorded after ``followup_end`` are treated as alive through the horizon (not events).
    """
    landmark = _landmark_monday(landmark_iso_week)
    censor_d = _censor_horizon_date(followup_end)
    wh = _horizon_weeks(landmark, censor_d)

    if "age_bin" not in df.columns:
        return pd.DataFrame(), "age_bin column missing (build enrollment table first)"

    sub = df.loc[df["age_bin"] == age_bin].copy()
    if sub.empty:
        return pd.DataFrame(), f"no rows with age_bin={age_bin!r}"
    subp = _sub_one_row_per_person(sub)

    dm = subp["death_monday_allcause"].to_numpy()
    fd = subp["first_dose_monday"].to_numpy()
    vc = subp["VaccineCode_FirstDose"].to_numpy() if "VaccineCode_FirstDose" in subp.columns else np.array([""] * len(subp), dtype=object)

    n_at_risk = 0
    cohort_to_te: dict[str, tuple[list[float], list[int]]] = {c: ([], []) for c in FIRST_MFG_COHORTS}

    for i in range(len(subp)):
        dth = _as_date(dm[i])
        if dth is not None and dth < landmark:
            continue
        n_at_risk += 1
        st = _assign_first_mfg_stratum(fd[i], vc[i], landmark)
        # Deaths after follow-up end are not observed in this window: administrative censor at horizon.
        if dth is not None and landmark <= dth <= censor_d:
            t = float(max(weeks_between(landmark, dth), 0))
            e = 1
        else:
            t, e = wh, 0
        tt, ee = cohort_to_te[st]
        tt.append(t)
        ee.append(e)

    if n_at_risk == 0:
        return pd.DataFrame(), "no one alive at landmark (all deaths strictly before landmark week)"

    return _fit_km_to_table(cohort_to_te, age_bin=age_bin)


def build_km_landmark_dose_nextdose_censor_table(
    df: pd.DataFrame,
    *,
    landmark_iso_week: str,
    followup_end: str,
    age_bin: str,
) -> tuple[pd.DataFrame, str]:
    """
    Landmark KM with dose0 | dose1 | dose2 at landmark; censor at next dose (2nd or 3rd) or follow-up end.
    Death is the event; dose transitions are non-informative censoring (standard KM).
    Times and deaths are truncated at ``followup_end``; dose steps after that week are ignored.
    """
    landmark = _landmark_monday(landmark_iso_week)
    censor_d = _censor_horizon_date(followup_end)

    if "age_bin" not in df.columns:
        return pd.DataFrame(), "age_bin column missing (build enrollment table first)"

    sub = df.loc[df["age_bin"] == age_bin].copy()
    if sub.empty:
        return pd.DataFrame(), f"no rows with age_bin={age_bin!r}"
    subp = _sub_one_row_per_person(sub)

    dm = subp["death_monday_allcause"].to_numpy()
    fd = subp["first_dose_monday"].to_numpy()
    sd = subp["second_dose_monday"].to_numpy()
    td = subp["third_dose_monday"].to_numpy() if "third_dose_monday" in subp.columns else np.full(len(subp), np.nan, dtype=object)

    cohort_to_te: dict[str, tuple[list[float], list[int]]] = {c: ([], []) for c in DOSE_COHORTS}

    wh = _horizon_weeks(landmark, censor_d)

    for i in range(len(subp)):
        dth = _as_date(dm[i])
        if dth is not None and dth < landmark:
            continue

        stratum = _assign_dose_stratum(fd[i], sd[i], landmark)
        d1 = _as_date(fd[i])
        d2 = _as_date(sd[i])
        d3 = _as_date(td[i])

        pairs: list[tuple[float, int]] = [(wh, 0)]

        if stratum == "dose0":
            if dth is not None and landmark <= dth <= censor_d:
                pairs.append((float(max(weeks_between(landmark, dth), 0)), 1))
            if d1 is not None and landmark < d1 <= censor_d:
                pairs.append((float(max(weeks_between(landmark, d1), 0)), 0))
        elif stratum == "dose1":
            if dth is not None and landmark <= dth <= censor_d:
                pairs.append((float(max(weeks_between(landmark, dth), 0)), 1))
            if d2 is not None and landmark < d2 <= censor_d:
                pairs.append((float(max(weeks_between(landmark, d2), 0)), 0))
        else:  # dose2
            if dth is not None and landmark <= dth <= censor_d:
                pairs.append((float(max(weeks_between(landmark, dth), 0)), 1))
            if d3 is not None and landmark < d3 <= censor_d:
                pairs.append((float(max(weeks_between(landmark, d3), 0)), 0))

        t, e = _min_time_event(pairs)
        tt, ee = cohort_to_te[stratum]
        tt.append(t)
        ee.append(e)

    n_at_risk = sum(len(cohort_to_te[c][0]) for c in DOSE_COHORTS)
    if n_at_risk == 0:
        return pd.DataFrame(), "no one alive at landmark for dose-stratified KM"

    return _fit_km_to_table(cohort_to_te, age_bin=age_bin)

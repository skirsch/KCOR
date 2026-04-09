"""Enrollment cohorts, age bins, parsed event dates (ISO week → Monday).

Date parsing follows KCOR_CMR.py: vectorized ``pd.to_datetime(s + '-1', format='%G-%V-%u')``
on whole columns — avoid ``.map()`` / per-row Python on multi-million-row frames.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Enrollment ~mid-2021 (e.g. ISO 2021-24): analyses use dose0 | dose1 | dose2 only.
# A third dose by enrollment is negligible; ``cohort_dose3`` is still stored but must not be
# listed in ``cfg["cohorts"]`` for this design (avoids overlapping dose2/dose3 strata).
PRIMARY_ENROLLMENT_COHORTS: frozenset[str] = frozenset({"dose0", "dose1", "dose2"})


def iso_week_str_to_monday(s: str | float | None) -> pd.Timestamp | pd.NaT:
    """Scalar ISO week YYYY-WW → Monday Timestamp (for config strings, small loops)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return pd.NaT
    t = str(s).strip()
    if not t:
        return pd.NaT
    t = "".join(c for c in t if c.isalnum() or c in "-")
    if len(t) < 6:
        return pd.NaT
    try:
        return pd.to_datetime(t + "-1", format="%G-%V-%u", errors="coerce")
    except Exception:
        return pd.NaT


def _iso_week_str_series_to_timestamp(series: pd.Series) -> pd.Series:
    """Vectorized YYYY-WW → Monday Timestamp (same as KCOR_CMR dose columns)."""
    x = series.fillna("").astype(str).str.strip()
    x = x.replace({"nan": "", "NaT": "", "<NA>": "", "None": ""})
    return pd.to_datetime(x + "-1", format="%G-%V-%u", errors="coerce")


def _lpz_death_series_to_timestamp(series: pd.Series) -> pd.Series:
    """LPZ death field: strip non-(digits, hyphen) then ISO-week parse (KCOR_CMR style)."""
    x = series.fillna("").astype(str).str.replace(r"[^0-9-]", "", regex=True)
    x = x.replace({"nan": "", "NaT": "", "<NA>": ""})
    return pd.to_datetime(x + "-1", format="%G-%V-%u", errors="coerce")


def _covid_death_series_to_timestamp(series: pd.Series) -> pd.Series:
    """Date_COVID_death: ISO week if present; single-digit flags → NaT."""
    x = series.fillna("").astype(str).str.strip()
    x = x.replace({"nan": "", "NaT": "", "<NA>": ""})
    bad = x.str.match(r"^\d$", na=False)
    ts = pd.to_datetime(x + "-1", format="%G-%V-%u", errors="coerce")
    return ts.mask(bad, pd.NaT)


def monday_to_iso_week(monday: date) -> str:
    iso = monday.isocalendar()
    return f"{iso.year}-{iso.week:02d}"


def week_index(monday: pd.Timestamp | date) -> tuple[int, int]:
    if hasattr(monday, "date"):
        monday = monday.date()
    ic = monday.isocalendar()
    return (ic.year, ic.week)


def weeks_between(start_monday: date, end_monday: date) -> int:
    if end_monday < start_monday:
        return -1
    sm = start_monday - timedelta(days=start_monday.weekday())
    em = end_monday - timedelta(days=end_monday.weekday())
    return (em - sm).days // 7


def iter_followup_mondays(start_iso: str, end_iso: str) -> list[date]:
    d0 = iso_week_str_to_monday(start_iso)
    d1 = iso_week_str_to_monday(end_iso)
    if pd.isna(d0) or pd.isna(d1):
        raise ValueError(f"Bad ISO week range {start_iso} {end_iso}")
    a = d0.date()
    b = d1.date()
    out = []
    cur = a
    while cur <= b:
        out.append(cur)
        cur = cur + timedelta(days=7)
    return out


def _assign_age_bin_vectorized(
    age: pd.Series,
    age_bins: list[list[int]],
) -> pd.Series:
    """Assign age_bin labels from (inclusive) YAML bins without per-row Python."""
    out = pd.Series(pd.NA, index=age.index, dtype="string")
    for lo, hi in age_bins:
        m = age.notna() & (age >= lo) & (age <= hi)
        out = out.mask(m, f"{lo}-{hi}")
    return out


def build_enrollment_table(
    df: pd.DataFrame,
    *,
    enrollment_week: str,
    age_bins: list[list[int]],
    progress_log: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Add columns: enrollment_monday, birth_band_start, age_at_enrollment, age_bin,
    cohort_dose0, cohort_dose1, cohort_dose2, cohort_dose3,
    first_dose_monday, second_dose_monday, third_dose_monday, infection_monday, death_monday,
    covid_death_monday, death_monday_allcause, etc.

    progress_log: optional ``print``-like callback for long runs (flush in caller).
    """
    log = progress_log or (lambda _m: None)

    def _p(msg: str) -> None:
        log(msg)

    out = df.copy()
    enroll_mon = iso_week_str_to_monday(enrollment_week)
    if pd.isna(enroll_mon):
        raise ValueError(f"Bad enrollment_week {enrollment_week}")
    enroll_d = enroll_mon.date()
    enroll_iso_year = enroll_d.isocalendar().year
    enroll_ts = pd.Timestamp(enroll_d).normalize()

    _p(f"enrollment: birth year + age bins ({len(out):,} rows) …")
    out["birth_band_start"] = (
        out["YearOfBirth"].astype(str).str.extract(r"(\d{4})", expand=False).astype(float)
    )
    out["age_at_enrollment"] = enroll_iso_year - out["birth_band_start"]
    out["age_bin"] = _assign_age_bin_vectorized(out["age_at_enrollment"], age_bins)

    _p("enrollment: vectorized dose 1 / 2 / 3 ISO dates (KCOR_CMR pattern) …")
    dose1_ts = _iso_week_str_series_to_timestamp(out["Date_FirstDose"]).dt.normalize()
    dose2_ts = _iso_week_str_series_to_timestamp(out["Date_SecondDose"]).dt.normalize()
    dose3_ts = _iso_week_str_series_to_timestamp(out["Date_ThirdDose"]).dt.normalize()
    out["first_dose_monday"] = dose1_ts.dt.date
    out["second_dose_monday"] = dose2_ts.dt.date
    out["third_dose_monday"] = dose3_ts.dt.date

    _p("enrollment: LPZ death (DateOfDeath) …")
    death_ts = _lpz_death_series_to_timestamp(out["DateOfDeath"]).dt.normalize()
    out["death_monday"] = death_ts.dt.date

    _p("enrollment: Date_COVID_death …")
    covid_ts = _covid_death_series_to_timestamp(out["Date_COVID_death"]).dt.normalize()
    out["covid_death_monday"] = covid_ts.dt.date

    # All-cause mortality timing: prefer LPZ DateOfDeath; if missing, use Date_COVID_death when present.
    # Registry rows often have COVID week filled while DateOfDeath is empty — without this, ACM=0 but COVID>0 in QA.
    _dm = pd.to_datetime(out["death_monday"], errors="coerce")
    _cdm = pd.to_datetime(out["covid_death_monday"], errors="coerce")
    out["death_monday_allcause"] = _dm.fillna(_cdm).dt.date

    _p("enrollment: DateOfPositiveTest …")
    inf_ts = _iso_week_str_series_to_timestamp(out["DateOfPositiveTest"]).dt.normalize()
    out["infection_monday"] = inf_ts.dt.date

    out["enrollment_monday"] = enroll_d

    # Alive at start of enrollment week: no LPZ death, or death in that week or later.
    # Must match KCOR_CMR.py (alive_at_enroll): DateOfDeath.isna() | (DateOfDeath >= enrollment_date).
    # BUG was death_ts <= enroll_ts, which excluded everyone who died after enrollment and included pre-enrollment deaths.
    alive_enroll = death_ts.isna() | (death_ts >= enroll_ts)
    out["eligible_enrollment"] = alive_enroll

    has_first = dose1_ts.notna() & (dose1_ts <= enroll_ts)
    has_second = dose2_ts.notna() & (dose2_ts <= enroll_ts)
    has_third = dose3_ts.notna() & (dose3_ts <= enroll_ts)

    # Mutually exclusive for main strata: unvaccinated | 1 dose | 2+ doses (primary course).
    # ``cohort_dose3`` flags third dose by enrollment (rare in 2021-24); CFR runs use PRIMARY_ENROLLMENT_COHORTS only.
    out["cohort_dose0"] = out["eligible_enrollment"] & ~has_first
    out["cohort_dose1"] = out["eligible_enrollment"] & has_first & ~has_second
    out["cohort_dose2"] = out["eligible_enrollment"] & has_second
    out["cohort_dose3"] = out["eligible_enrollment"] & has_third

    _p("enrollment: ISO week string columns for infection/death …")
    out["infection_iso_week"] = inf_ts.dt.strftime("%G-%V").fillna("")
    out["death_iso_week"] = _dm.fillna(_cdm).dt.strftime("%G-%V").fillna("")

    _p("enrollment: weeks second dose → enrollment …")
    valid_w = has_second
    delta_days = (enroll_ts - dose2_ts).dt.days
    out["weeks_second_dose_to_enrollment"] = (delta_days // 7).where(valid_w, np.nan)

    _p("enrollment: done")
    return out


def cohort_mask(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "dose0":
        return df["cohort_dose0"]
    if name == "dose1":
        return df["cohort_dose1"]
    if name == "dose2":
        return df["cohort_dose2"]
    if name == "dose3":
        return df["cohort_dose3"]
    raise ValueError(f"Unknown cohort {name}")

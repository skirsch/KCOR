"""Period-integrated wave summary and VE vs reference (synthetic weekly)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

CFR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CFR_DIR / "code"))

from metrics import (  # noqa: E402
    build_period_aggregate_summary,
    build_period_ve_summary,
    expected_weeks_in_period,
    iso_weeks_in_period,
)


def _tiny_weekly() -> pd.DataFrame:
    """Two ISO weeks 2021-41–42, dose0 vs dose2, age_bin=all."""
    rows = [
        dict(
            iso_week="2021-41",
            cohort="dose0",
            age_bin="all",
            population_at_risk=100,
            cases=10,
            deaths_covid=1,
            deaths_all=2,
            deaths_non_covid=1,
        ),
        dict(
            iso_week="2021-42",
            cohort="dose0",
            age_bin="all",
            population_at_risk=100,
            cases=10,
            deaths_covid=1,
            deaths_all=2,
            deaths_non_covid=1,
        ),
        dict(
            iso_week="2021-41",
            cohort="dose2",
            age_bin="all",
            population_at_risk=200,
            cases=5,
            deaths_covid=1,
            deaths_all=2,
            deaths_non_covid=1,
        ),
        dict(
            iso_week="2021-42",
            cohort="dose2",
            age_bin="all",
            population_at_risk=200,
            cases=5,
            deaths_covid=1,
            deaths_all=2,
            deaths_non_covid=1,
        ),
    ]
    w = pd.DataFrame(rows)
    w["mortality_rate_all_cause"] = w["deaths_all"] / w["population_at_risk"]
    return w


def test_iso_weeks_and_expected_count() -> None:
    s = iso_weeks_in_period("2021-41", "2021-42")
    assert s == {"2021-41", "2021-42"}
    assert expected_weeks_in_period("2021-41", "2021-42") == 2


def test_build_period_aggregate_summary() -> None:
    w = _tiny_weekly()
    sm = build_period_aggregate_summary(
        w,
        period_start="2021-41",
        period_end="2021-42",
        period_name="wave",
        rate_suffix="_wave",
    )
    assert len(sm) == 2
    d0 = sm.loc[sm["cohort"] == "dose0"].iloc[0]
    assert d0["total_person_weeks"] == 200
    assert d0["total_cases"] == 20
    assert d0["total_covid_deaths"] == 2
    assert d0["total_allcause_deaths"] == 4
    assert d0["weeks_in_period"] == 2
    assert d0["total_rows"] == 2
    assert d0["expected_weeks_in_period"] == 2
    assert abs(float(d0["case_rate_wave"]) - 20 / 200) < 1e-12
    assert abs(float(d0["covid_death_rate_wave"]) - 2 / 200) < 1e-12
    assert abs(float(d0["cfr_covid_wave"]) - 2 / 20) < 1e-12


def test_build_period_ve_summary_rr_ve_diff() -> None:
    w = _tiny_weekly()
    sm = build_period_aggregate_summary(
        w,
        period_start="2021-41",
        period_end="2021-42",
        period_name="wave",
        rate_suffix="_wave",
    )
    ve = build_period_ve_summary(sm, reference_cohort="dose0", rate_suffix="_wave")
    assert len(ve) == 1
    r = ve.iloc[0]
    assert r["cohort"] == "dose2"
    # dose0: case_rate 0.1, covid_dr 0.01, cfr_covid 0.1
    # dose2: case_rate 10/400=0.025, covid_dr 2/400=0.005, cfr_covid 2/10=0.2
    assert abs(float(r["rr_case_rate"]) - 0.25) < 1e-9
    assert abs(float(r["ve_case_rate"]) - 0.75) < 1e-9
    assert abs(float(r["diff_case_rate"]) - (-0.075)) < 1e-9
    assert abs(float(r["rr_covid_death_rate"]) - 0.5) < 1e-9
    assert abs(float(r["ve_covid_death_rate"]) - 0.5) < 1e-9
    assert abs(float(r["rr_cfr_covid"]) - 2.0) < 1e-9
    assert abs(float(r["ve_cfr_covid"]) - (-1.0)) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

"""Weekly vaccine coverage: denominators, dose timing, death, aggregate ``all``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CFR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CFR_DIR / "code"))

from cohort_builder import iter_followup_mondays  # noqa: E402
from coverage import build_weekly_vaccine_coverage  # noqa: E402


def _weeks_4142() -> tuple:
    w = list(iter_followup_mondays("2021-41", "2021-42"))
    assert len(w) == 2
    return w[0], w[1]


def test_coverage_ge1_bounds_and_dose_week() -> None:
    w0, w1 = _weeks_4142()
    df = pd.DataFrame(
        {
            "ID": [1, 2],
            "age_bin": ["40-49", "40-49"],
            "Infection": [1, 1],
            "death_monday_allcause": [pd.NaT, pd.NaT],
            "first_dose_monday": [pd.NaT, w1],
            "second_dose_monday": [pd.NaT, pd.NaT],
            "third_dose_monday": [pd.NaT, pd.NaT],
        }
    )
    out = build_weekly_vaccine_coverage(
        df,
        followup_start="2021-41",
        followup_end="2021-42",
        age_bins_config=[[40, 49]],
    )
    sub = out[out["age_bin"] == "40-49"].set_index("iso_week")
    assert sub.loc["2021-41", "population_at_risk"] == 2
    assert sub.loc["2021-41", "vaccinated_ge1"] == 0
    assert sub.loc["2021-41", "coverage_ge1"] == 0.0
    assert sub.loc["2021-42", "vaccinated_ge1"] == 1
    assert abs(float(sub.loc["2021-42", "coverage_ge1"]) - 0.5) < 1e-12
    cov = out["coverage_ge1"].to_numpy()
    assert np.all((cov >= 0) & (cov <= 1) | np.isnan(cov))


def test_death_excludes_later_denominator() -> None:
    w0, w1 = _weeks_4142()
    df = pd.DataFrame(
        {
            "ID": [1],
            "age_bin": ["40-49"],
            "Infection": [1],
            "death_monday_allcause": [w0],
            "first_dose_monday": [pd.NaT],
            "second_dose_monday": [pd.NaT],
            "third_dose_monday": [pd.NaT],
        }
    )
    out = build_weekly_vaccine_coverage(
        df,
        followup_start="2021-41",
        followup_end="2021-42",
        age_bins_config=[[40, 49]],
    )
    sub = out[out["age_bin"] == "40-49"].set_index("iso_week")
    assert sub.loc["2021-41", "population_at_risk"] == 1
    assert sub.loc["2021-42", "population_at_risk"] == 0
    assert np.isnan(float(sub.loc["2021-42", "coverage_ge1"]))


def test_age_bin_all_matches_sum_of_strata() -> None:
    w0, _ = _weeks_4142()
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "age_bin": ["40-49", "40-49", "50-59"],
            "Infection": [1, 1, 1],
            "death_monday_allcause": [pd.NaT, pd.NaT, pd.NaT],
            "first_dose_monday": [pd.NaT, w0, pd.NaT],
            "second_dose_monday": [pd.NaT, pd.NaT, pd.NaT],
            "third_dose_monday": [pd.NaT, pd.NaT, pd.NaT],
        }
    )
    out = build_weekly_vaccine_coverage(
        df,
        followup_start="2021-41",
        followup_end="2021-41",
        age_bins_config=[[40, 49], [50, 59]],
    )
    row_all = out[out["age_bin"] == "all"].iloc[0]
    p40 = out[(out["age_bin"] == "40-49") & (out["iso_week"] == "2021-41")].iloc[0]
    p50 = out[(out["age_bin"] == "50-59") & (out["iso_week"] == "2021-41")].iloc[0]
    assert row_all["population_at_risk"] == p40["population_at_risk"] + p50["population_at_risk"]
    assert row_all["vaccinated_ge1"] == p40["vaccinated_ge1"] + p50["vaccinated_ge1"]
    den = row_all["population_at_risk"]
    assert den == 3
    assert row_all["vaccinated_ge1"] == 1
    assert abs(float(row_all["coverage_ge1"]) - 1.0 / 3.0) < 1e-12


def test_reinfection_rows_deduped_for_population() -> None:
    w0, _ = _weeks_4142()
    df = pd.DataFrame(
        {
            "ID": [1, 1],
            "age_bin": ["40-49", "40-49"],
            "Infection": [1, 2],
            "death_monday_allcause": [pd.NaT, pd.NaT],
            "first_dose_monday": [w0, w0],
            "second_dose_monday": [pd.NaT, pd.NaT],
            "third_dose_monday": [pd.NaT, pd.NaT],
        }
    )
    out = build_weekly_vaccine_coverage(
        df,
        followup_start="2021-41",
        followup_end="2021-41",
        age_bins_config=[[40, 49]],
    )
    sub = out[out["age_bin"] == "40-49"].iloc[0]
    assert sub["population_at_risk"] == 1
    assert sub["vaccinated_ge1"] == 1
    assert float(sub["coverage_ge1"]) == 1.0

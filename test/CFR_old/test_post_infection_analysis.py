from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parent / "code")))

from analysis import (  # noqa: E402
    build_infected_cohort_age_composition_table,
    build_km_post_infection_age_bin_table,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort_dose0": [True, True, False, False],
            "cohort_dose1": [False, False, True, False],
            "cohort_dose2": [False, False, False, True],
            "infection_monday": [
                pd.to_datetime("2021-10-11").date(),
                pd.to_datetime("2021-10-11").date(),
                pd.to_datetime("2021-10-18").date(),
                pd.to_datetime("2021-10-18").date(),
            ],
            "death_monday_allcause": [
                pd.to_datetime("2021-10-18").date(),
                pd.NaT,
                pd.NaT,
                pd.to_datetime("2021-11-01").date(),
            ],
            "age_bin": ["40-49", "70-120", "40-49", "70-120"],
            "age_at_enrollment": [45, 75, 48, 78],
        }
    )


def test_infected_cohort_age_composition_table_summarizes_counts_and_shares() -> None:
    out = build_infected_cohort_age_composition_table(_toy_df(), cohorts=["dose0", "dose1", "dose2"])
    assert not out.empty
    dose0_all = out[(out["cohort"] == "dose0") & (out["age_bin"] == "all")].iloc[0]
    assert int(dose0_all["infected_n"]) == 2
    assert float(dose0_all["infected_share_within_cohort"]) == 1.0
    dose0_40s = out[(out["cohort"] == "dose0") & (out["age_bin"] == "40-49")].iloc[0]
    assert int(dose0_40s["infected_n"]) == 1
    assert float(dose0_40s["infected_share_within_cohort"]) == 0.5


def test_km_post_infection_age_bin_table_keeps_age_bin_dimension() -> None:
    out, reason = build_km_post_infection_age_bin_table(
        _toy_df(),
        followup_end="2021-52",
        cohorts=["dose0", "dose1", "dose2"],
        age_bins=["40-49", "70-120"],
    )
    assert reason == ""
    assert not out.empty
    assert set(out["age_bin"]) == {"40-49", "70-120"}
    assert set(out["cohort"]) >= {"dose0", "dose1", "dose2"}

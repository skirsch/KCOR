from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parent / "code")))

from cohort_builder import add_prior_infection_before_enrollment_flag


def test_add_prior_infection_before_enrollment_flag_uses_earliest_episode() -> None:
    df = pd.DataFrame(
        {
            "ID": ["a", "a", "b", "c"],
            "infection_monday": [
                pd.to_datetime("2021-06-07").date(),
                pd.to_datetime("2021-10-11").date(),
                pd.to_datetime("2021-10-11").date(),
                pd.NaT,
            ],
            "enrollment_monday": [
                pd.to_datetime("2021-06-14").date(),
                pd.to_datetime("2021-06-14").date(),
                pd.to_datetime("2021-06-14").date(),
                pd.to_datetime("2021-06-14").date(),
            ],
        }
    )
    out = add_prior_infection_before_enrollment_flag(df)
    flags = dict(zip(out["ID"], out["prior_infection_before_enrollment"], strict=False))
    assert flags["a"] is True
    assert flags["b"] is False
    assert flags["c"] is False

"""Landmark KM: first-dose manufacturer strata and dose strata with next-dose censoring."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parent / "code")))

from km_landmark import (  # noqa: E402
    build_km_landmark_dose_nextdose_censor_table,
    build_km_landmark_first_mfg_table,
    resolve_landmark_age_bins,
)


def _tiny_df() -> pd.DataFrame:
    # ISO 2021-24 Monday = 2021-06-14
    L = date(2021, 6, 14)
    w_before = date(2021, 6, 7)
    w_after = date(2021, 7, 5)
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "age_bin": ["70-120"] * 6,
            "Infection": [1] * 6,
            "death_monday_allcause": [pd.NaT, L, w_after, pd.NaT, pd.NaT, pd.NaT],
            "first_dose_monday": [pd.NaT, L, L, L, w_after, w_before],
            "second_dose_monday": [pd.NaT, pd.NaT, w_after, w_after, pd.NaT, L],
            "third_dose_monday": [pd.NaT] * 6,
            "VaccineCode_FirstDose": ["", "CO01", "CO02", "CO03", "CO01", "CO01"],
        }
    )


def test_landmark_first_mfg_table():
    df = _tiny_df()
    tbl, reason = build_km_landmark_first_mfg_table(
        df,
        landmark_iso_week="2021-24",
        followup_end="2022-20",
        age_bin="70-120",
    )
    assert reason == ""
    assert not tbl.empty
    assert set(tbl["cohort"].unique()) <= {"unvax", "pfizer", "moderna", "OTHER"}
    # ID 1 died before landmark → excluded
    # ID 2 Pfizer at L, death at L → event t=0 in pfizer
    # ID 3 Moderna at L, death after → event in moderna
    # ID 4 AZ at L, censored
    # ID 5 unvax at L (first dose after L), censored
    pf = tbl[tbl["cohort"] == "pfizer"]
    assert len(pf) > 0
    assert float(pf["KM_estimate"].min()) < 1.0


def test_landmark_dose_censor_table():
    df = _tiny_df()
    tbl, reason = build_km_landmark_dose_nextdose_censor_table(
        df,
        landmark_iso_week="2021-24",
        followup_end="2022-20",
        age_bin="70-120",
    )
    assert reason == ""
    assert not tbl.empty
    assert set(tbl["cohort"].unique()) <= {"dose0", "dose1", "dose2"}
    assert "dose2" in set(tbl["cohort"].unique())


def test_resolve_landmark_age_bins():
    labels = ["40-49", "50-59", "70-120"]
    assert resolve_landmark_age_bins({"age_bins": "all"}, labels) == labels
    assert resolve_landmark_age_bins({"age_bin": "all"}, labels) == labels
    assert resolve_landmark_age_bins({"age_bin": "70-120"}, labels) == ["70-120"]
    assert resolve_landmark_age_bins({"age_bins": ["40-49", "70-120"]}, labels) == ["40-49", "70-120"]

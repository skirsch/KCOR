"""Population / death tallies: one person per ID despite multiple infection rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CFR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CFR_DIR / "code"))

from cohort_builder import iter_followup_mondays, monday_to_iso_week  # noqa: E402
from metrics import _compute_weekly_stratum_rows, _week_index_map  # noqa: E402


def _two_week_frame() -> tuple[list, dict, list[str]]:
    weeks = list(iter_followup_mondays("2021-41", "2021-42"))
    wmap = _week_index_map(weeks)
    iso_labels = [monday_to_iso_week(w) for w in weeks]
    return weeks, wmap, iso_labels


def test_population_and_deaths_dedupe_duplicate_infection_rows() -> None:
    """Same ID, Infection 1+2: pop and deaths count once; cases still episode-based."""
    weeks, wmap, iso_labels = _two_week_frame()
    w0, w1 = weeks[0], weeks[1]

    df = pd.DataFrame(
        {
            "ID": [1, 1, 1, 1],
            "Infection": [1, 2, 1, 2],
            "cohort_dose0": [True, True, False, False],
            "cohort_dose1": [False, False, False, False],
            "cohort_dose2": [False, False, True, True],
            "cohort_dose3": [False, False, False, False],
            "age_bin": ["70-120"] * 4,
            "infection_monday": [w0, w0, w1, w1],
            "death_monday_allcause": [w1, w1, pd.NaT, pd.NaT],
            "covid_death_monday": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        }
    )

    rows0 = _compute_weekly_stratum_rows(
        df, "dose0", "70-120", weeks=weeks, wmap=wmap, iso_labels=iso_labels, cohort_masks=None
    )
    by_iso = {r["iso_week"]: r for r in rows0}
    # Week 41: one person alive (two episode rows, one human)
    assert by_iso["2021-41"]["population_at_risk"] == 1
    # Two infection rows dated week 41 → 2 episode cases
    assert by_iso["2021-41"]["cases"] == 2
    # One death in week 42, not double-counted from two rows
    assert by_iso["2021-42"]["deaths_all"] == 1

    rows2 = _compute_weekly_stratum_rows(
        df, "dose2", "70-120", weeks=weeks, wmap=wmap, iso_labels=iso_labels, cohort_masks=None
    )
    by2 = {r["iso_week"]: r for r in rows2}
    assert by2["2021-41"]["population_at_risk"] == 1
    assert by2["2021-42"]["cases"] == 2

"""QA spot-check helpers: file-wide totals for one ISO week."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

CFR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CFR_DIR / "code"))

from cohort_builder import iso_week_str_to_monday, monday_to_iso_week  # noqa: E402
from qa_summary import (  # noqa: E402
    _birth_year_range_mask,
    _period_covid_deaths_per_person_week,
    _person_table_all_ids,
    _spot_week_metrics,
)


def test_czech_yaml_spot_expected_keys() -> None:
    cfg_path = CFR_DIR / "config" / "czech.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    qs = cfg["qa_summary"]
    assert qs["period_birth_year_min"] == 1940 and qs["period_birth_year_max"] == 1949
    dbg = qs["debug_enrollment_weekly_csv"]
    assert dbg["enabled"] and dbg["birth_year_min"] == 1930 and dbg["birth_year_max"] == 1939
    sc = qs["spot_check"]
    assert sc["birth_year_min"] == 1930 and sc["birth_year_max"] == 1939
    exp = sc["expected"]
    assert exp["covid_deaths"] == 242
    assert exp["acm_deaths"] == 947


def test_person_table_all_ids_dedupes() -> None:
    df = pd.DataFrame(
        {
            "ID": ["a", "a", "b"],
            "death_monday": [date(2021, 11, 29), date(2021, 12, 6), pd.NaT],
            "covid_death_monday": [date(2021, 11, 29), pd.NaT, date(2021, 11, 29)],
        }
    )
    p = _person_table_all_ids(df)
    assert len(p) == 2
    assert list(p.index) == ["a", "b"]


def test_spot_week_total_202148() -> None:
    """Two IDs with death in 2021-48 (Monday 2021-11-29)."""
    df = pd.DataFrame(
        {
            "ID": [1, 2],
            "death_monday": [date(2021, 11, 29), date(2021, 11, 29)],
            "covid_death_monday": [date(2021, 11, 29), pd.NaT],
        }
    )
    p = _person_table_all_ids(df)
    m = pd.Series(True, index=p.index)
    n_cv, n_lpz, n_union, n_nc = _spot_week_metrics(p, {"2021-48"}, m)
    assert n_cv == 1
    assert n_lpz == 2
    assert n_union == 2
    assert n_nc == 1


def test_spot_check_iso_week_is_single_parsable_week() -> None:
    """Config iso_week must parse to one Monday; label matches pipeline ISO week strings."""
    raw = "2021-48"
    ts = iso_week_str_to_monday(raw)
    assert not pd.isna(ts)
    iso_w = monday_to_iso_week(ts.date())
    assert iso_w == "2021-48"


def test_birth_year_range_mask_193x() -> None:
    s = pd.Series([1929, 1930, 1935, 1939, 1940, float("nan"), None])
    m = _birth_year_range_mask(s, 1930, 1939)
    assert list(m) == [False, True, True, True, False, False, False]


def test_person_table_all_ids_includes_birth_when_present() -> None:
    df = pd.DataFrame(
        {
            "ID": ["a", "a"],
            "death_monday": [pd.NaT, pd.NaT],
            "covid_death_monday": [pd.NaT, pd.NaT],
            "birth_band_start": [1935.0, 1935.0],
        }
    )
    p = _person_table_all_ids(df)
    assert "birth_band_start" in p.columns
    assert p.loc["a", "birth_band_start"] == 1935.0


def test_period_covid_deaths_per_person_week() -> None:
    """(3+2)/(100+90) over two ISO weeks for dose0 using deaths_covid."""
    weekly = pd.DataFrame(
        {
            "iso_week": ["2021-40", "2021-41"],
            "cohort": ["dose0", "dose0"],
            "age_bin": ["all", "all"],
            "population_at_risk": [100, 90],
            "deaths_covid": [3, 2],
        }
    )
    r, td, tp = _period_covid_deaths_per_person_week(
        weekly, "dose0", iso_weeks={"2021-40", "2021-41"}
    )
    assert td == 5 and tp == 190
    assert abs(r - 5 / 190) < 1e-12

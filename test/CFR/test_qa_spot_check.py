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

from qa_summary import (  # noqa: E402
    _period_allcause_deaths_per_person_week,
    _person_table_all_ids,
    _spot_week_metrics,
)


def test_czech_yaml_spot_expected_keys() -> None:
    cfg_path = CFR_DIR / "config" / "czech.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    exp = cfg["qa_summary"]["spot_check"]["expected"]
    assert "covid_deaths" in exp and "acm_deaths" in exp
    assert exp["covid_deaths"] == 145
    assert exp["acm_deaths"] == 514
    assert "birth_year" not in cfg["qa_summary"]["spot_check"]


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


def test_period_allcause_deaths_per_person_week() -> None:
    """(3+2)/(100+90) over two ISO weeks for dose0."""
    weekly = pd.DataFrame(
        {
            "iso_week": ["2021-40", "2021-41"],
            "cohort": ["dose0", "dose0"],
            "age_bin": ["all", "all"],
            "population_at_risk": [100, 90],
            "deaths_all": [3, 2],
        }
    )
    r, td, tp = _period_allcause_deaths_per_person_week(
        weekly, "dose0", iso_weeks={"2021-40", "2021-41"}
    )
    assert td == 5 and tp == 190
    assert abs(r - 5 / 190) < 1e-12

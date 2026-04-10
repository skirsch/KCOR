from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parent / "code")))

from falsification import (
    build_age_group_weekly,
    build_coverage_dilution_summary,
    build_incidence_severity_decomposition_summary,
    build_multi_split_falsification_summary,
    build_negative_control_rank_summary,
    build_old_young_difference_weekly,
    build_quantitative_scenario_bounds,
    build_ve_death_signal_bounds,
)


def test_build_age_group_weekly_aggregates_across_cohorts() -> None:
    weekly = pd.DataFrame(
        {
            "iso_week": ["2021-40", "2021-40", "2021-40", "2021-40"],
            "week_monday": ["2021-10-04"] * 4,
            "cohort": ["dose0", "dose1", "dose0", "dose1"],
            "age_bin": ["40-49", "40-49", "70-120", "70-120"],
            "population_at_risk": [100, 50, 80, 20],
            "cases": [10, 5, 8, 2],
            "deaths_all": [1, 0, 4, 1],
            "deaths_covid": [1, 0, 3, 1],
            "deaths_non_covid": [0, 0, 1, 0],
        }
    )
    out = build_age_group_weekly(
        weekly,
        age_groups={"younger": ["40-49"], "older": ["70-120"]},
    ).sort_values("age_group", kind="mergesort")

    older = out[out["age_group"] == "older"].iloc[0]
    younger = out[out["age_group"] == "younger"].iloc[0]
    assert int(younger["population_at_risk"]) == 150
    assert int(older["deaths_covid"]) == 4
    assert younger["mortality_rate_covid"] == 1 / 150
    assert older["mortality_rate_non_covid"] == 1 / 100


def test_build_old_young_difference_weekly_computes_differences() -> None:
    group_weekly = pd.DataFrame(
        {
            "iso_week": ["2021-40", "2021-40"],
            "week_monday": ["2021-10-04", "2021-10-04"],
            "age_group": ["younger", "older"],
            "population_at_risk": [150, 100],
            "mortality_rate_covid": [0.01, 0.03],
            "mortality_rate_non_covid": [0.02, 0.05],
        }
    )
    out = build_old_young_difference_weekly(
        group_weekly,
        older_group="older",
        younger_group="younger",
    )
    row = out.iloc[0]
    assert row["mortality_rate_covid_diff_old_minus_young"] == pytest.approx(0.02)
    assert row["mortality_rate_non_covid_diff_old_minus_young"] == pytest.approx(0.03)


def test_build_coverage_dilution_summary_scales_with_coverage() -> None:
    weekly = pd.DataFrame(
        {
            "iso_week": ["2021-41", "2021-41"],
            "week_monday": ["2021-10-11", "2021-10-11"],
            "cohort": ["dose0", "dose0"],
            "age_bin": ["40-49", "70-120"],
            "population_at_risk": [100, 100],
            "mortality_rate_covid": [0.01, 0.02],
        }
    )
    coverage = pd.DataFrame(
        {
            "iso_week": ["2021-41", "2021-41"],
            "age_bin": ["40-49", "70-120"],
            "coverage_ge1": [0.2, 0.8],
        }
    )
    out = build_coverage_dilution_summary(
        weekly,
        coverage,
        age_groups={"younger": ["40-49"], "older": ["70-120"]},
        ve_assumptions=[0.5],
        reference_cohort="dose0",
        wave_start="2021-41",
        wave_end="2021-41",
    )
    older = out[(out["age_group"] == "older") & (out["iso_week"] == "wave_mean")].iloc[0]
    younger = out[(out["age_group"] == "younger") & (out["iso_week"] == "wave_mean")].iloc[0]
    assert older["expected_pop_reduction_ve_0.5"] > younger["expected_pop_reduction_ve_0.5"]


def test_build_multi_split_falsification_summary_returns_rows() -> None:
    weekly = pd.DataFrame(
        {
            "iso_week": ["2021-40", "2021-41", "2021-40", "2021-41"] * 2,
            "week_monday": ["2021-10-04", "2021-10-11", "2021-10-04", "2021-10-11"] * 2,
            "cohort": ["dose0"] * 8,
            "age_bin": ["40-49", "40-49", "70-120", "70-120", "50-59", "50-59", "60-69", "60-69"],
            "population_at_risk": [100, 100, 100, 100, 100, 100, 100, 100],
            "cases": [1, 1, 1, 1, 1, 1, 1, 1],
            "deaths_all": [1, 1, 3, 4, 1, 1, 2, 2],
            "deaths_covid": [0, 0, 1, 2, 0, 0, 1, 1],
            "deaths_non_covid": [1, 1, 2, 2, 1, 1, 1, 1],
            "mortality_rate_covid": [0.0, 0.0, 0.01, 0.02, 0.0, 0.0, 0.01, 0.01],
        }
    )
    coverage = pd.DataFrame(
        {
            "iso_week": ["2021-40", "2021-41", "2021-40", "2021-41", "2021-40", "2021-41", "2021-40", "2021-41"],
            "age_bin": ["40-49", "40-49", "70-120", "70-120", "50-59", "50-59", "60-69", "60-69"],
            "coverage_ge1": [0.2, 0.2, 0.8, 0.8, 0.4, 0.4, 0.6, 0.6],
        }
    )
    out = build_multi_split_falsification_summary(
        weekly,
        coverage_weekly=coverage,
        split_definitions=[
            {"name": "70plus_vs_under70", "younger": ["40-49", "50-59", "60-69"], "older": ["70-120"]},
            {"name": "60plus_vs_under60", "younger": ["40-49", "50-59"], "older": ["60-69", "70-120"]},
        ],
        break_iso_week="2021-41",
        placebo_start="2021-40",
        placebo_end="2021-41",
        ve_assumptions=[0.5],
        reference_cohort="dose0",
        wave_start="2021-41",
        wave_end="2021-41",
    )
    assert set(out["split_name"]) == {"70plus_vs_under70", "60plus_vs_under60"}
    assert (out["outcome"] == "coverage_dilution_wave_mean").any()


def test_negative_control_and_bounds_helpers_return_expected_columns() -> None:
    multi = pd.DataFrame(
        {
            "split_name": ["70plus_vs_under70", "70plus_vs_under70", "70plus_vs_under70", "70plus_vs_under70"],
            "series_kind": ["old_minus_young"] * 4,
            "outcome": [
                "case_rate_diff_old_minus_young",
                "mortality_rate_covid_diff_old_minus_young",
                "mortality_rate_non_covid_diff_old_minus_young",
                "mortality_rate_all_cause_diff_old_minus_young",
            ],
            "level_jump_at_break": [0.01, 0.02, 0.005, 0.025],
            "placebo_rank_slope": [2, 1, 4, 3],
            "placebo_rank_level": [2, 1, 3, 2],
        }
    )
    neg = build_negative_control_rank_summary(multi)
    assert neg.iloc[0]["covid_minus_noncovid_level_jump"] == pytest.approx(0.015)

    wave_ve = pd.DataFrame(
        {
            "cohort": ["dose2", "dose2"],
            "age_bin": ["60-69", "70-120"],
            "cohort_total_person_weeks": [100, 300],
            "ve_covid_death_rate": [0.8, 0.6],
        }
    )
    bounds = build_ve_death_signal_bounds(
        wave_ve,
        multi_split_summary=multi,
        split_definitions=[
            {"name": "70plus_vs_under70", "older": ["70-120"], "younger": ["40-49", "50-59", "60-69"]},
            {"name": "60plus_vs_under60", "older": ["60-69", "70-120"], "younger": ["40-49", "50-59"]},
        ],
        compare_cohort="dose2",
    )
    assert set(bounds["split_name"]) == {"70plus_vs_under70"}
    assert "attenuated_ve_death_signal" in bounds.columns


def test_incidence_severity_decomposition_summary_labels_incidence() -> None:
    multi = pd.DataFrame(
        {
            "split_name": ["70plus_vs_under70"] * 3,
            "series_kind": ["old_minus_young"] * 3,
            "outcome": [
                "case_rate_diff_old_minus_young",
                "cfr_covid_episode_diff_old_minus_young",
                "mortality_rate_covid_diff_old_minus_young",
            ],
            "level_jump_at_break": [-0.005, 0.0004, 0.0002],
            "placebo_rank_slope": [1, 2, 1],
            "placebo_rank_level": [1, 3, 1],
        }
    )
    out = build_incidence_severity_decomposition_summary(multi)
    assert out.iloc[0]["dominant_component"] == "incidence"


def test_quantitative_scenario_bounds_returns_expected_range() -> None:
    bounds = pd.DataFrame(
        {
            "split_name": ["70plus_vs_under70"],
            "older_bins": ["70-120"],
            "older_weighted_wave_ve_covid_death_rate": [0.8],
            "covid_specific_jump_share_after_noncovid_subtraction": [0.4],
            "attenuated_ve_death_signal": [0.32],
        }
    )
    multi = pd.DataFrame(
        {
            "split_name": ["70plus_vs_under70"],
            "series_kind": ["coverage_dilution"],
            "series_name": ["older"],
            "mean_wave_coverage_ge1": [0.8],
        }
    )
    out = build_quantitative_scenario_bounds(bounds, multi)
    row = out.iloc[0]
    assert row["coverage_calibrated_point"] == pytest.approx(0.5)
    assert row["scenario_range_hi"] == pytest.approx(0.5)

"""Population-level falsification checks for ecological old-vs-young comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_VE_ASSUMPTIONS: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)


def _norm_groups(age_groups: Mapping[str, Sequence[str]] | None) -> dict[str, list[str]]:
    if not age_groups:
        return {"younger": ["40-49", "50-59", "60-69"], "older": ["70-120"]}
    out: dict[str, list[str]] = {}
    for name, vals in age_groups.items():
        arr = [str(v) for v in vals]
        if arr:
            out[str(name)] = arr
    if len(out) < 2:
        raise ValueError("falsification age_groups must define at least two non-empty groups")
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(v[ok], weights=w[ok]))


def build_age_group_weekly(
    weekly: pd.DataFrame,
    *,
    age_groups: Mapping[str, Sequence[str]] | None = None,
    coverage_weekly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate weekly metrics across cohorts into broad age groups."""
    groups = _norm_groups(age_groups)
    sub = weekly[weekly["age_bin"] != "all"].copy()
    if sub.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for group_name, bins in groups.items():
        g = sub[sub["age_bin"].isin(bins)].copy()
        if g.empty:
            continue
        agg = (
            g.groupby(["iso_week", "week_monday"], as_index=False)
            .agg(
                population_at_risk=("population_at_risk", "sum"),
                cases=("cases", "sum"),
                deaths_all=("deaths_all", "sum"),
                deaths_covid=("deaths_covid", "sum"),
                deaths_non_covid=("deaths_non_covid", "sum"),
            )
            .assign(age_group=group_name)
        )
        # Rebuild episode-style deaths from row-level CFR × cases so broad-group CFR stays aligned to the pipeline's episode logic.
        if "cfr_covid" in g.columns and "cfr_allcause" in g.columns:
            gtmp = g.copy()
            gtmp["covid_episode_deaths_component"] = (
                pd.to_numeric(gtmp["cfr_covid"], errors="coerce").fillna(0.0)
                * pd.to_numeric(gtmp["cases"], errors="coerce").fillna(0.0)
            )
            gtmp["allcause_episode_deaths_component"] = (
                pd.to_numeric(gtmp["cfr_allcause"], errors="coerce").fillna(0.0)
                * pd.to_numeric(gtmp["cases"], errors="coerce").fillna(0.0)
            )
            epi = gtmp.groupby(["iso_week", "week_monday"], as_index=False).agg(
                covid_episode_deaths_implied=("covid_episode_deaths_component", "sum"),
                allcause_episode_deaths_implied=("allcause_episode_deaths_component", "sum"),
            )
            agg = agg.merge(epi, on=["iso_week", "week_monday"], how="left")
        else:
            agg["covid_episode_deaths_implied"] = np.nan
            agg["allcause_episode_deaths_implied"] = np.nan
        pop = agg["population_at_risk"].astype(float)
        agg["case_rate"] = np.where(pop > 0, agg["cases"] / pop, np.nan)
        agg["mortality_rate_all_cause"] = np.where(pop > 0, agg["deaths_all"] / pop, np.nan)
        agg["mortality_rate_covid"] = np.where(pop > 0, agg["deaths_covid"] / pop, np.nan)
        agg["mortality_rate_non_covid"] = np.where(pop > 0, agg["deaths_non_covid"] / pop, np.nan)
        cases = agg["cases"].astype(float)
        agg["cfr_covid_episode"] = np.where(cases > 0, agg["covid_episode_deaths_implied"] / cases, np.nan)
        agg["cfr_allcause_episode"] = np.where(cases > 0, agg["allcause_episode_deaths_implied"] / cases, np.nan)
        agg = agg.sort_values("iso_week", kind="mergesort")
        agg["cumulative_deaths_covid"] = agg["deaths_covid"].cumsum()
        agg["cumulative_deaths_all"] = agg["deaths_all"].cumsum()
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty or coverage_weekly is None or coverage_weekly.empty:
        return out

    cov = coverage_weekly[coverage_weekly["age_bin"] != "all"].copy()
    if cov.empty:
        return out
    pop_age = (
        sub.groupby(["iso_week", "age_bin"], as_index=False)["population_at_risk"]
        .sum()
        .rename(columns={"population_at_risk": "weekly_population_at_risk"})
    )
    cov = cov.merge(pop_age, on=["iso_week", "age_bin"], how="left")
    cov["weekly_population_at_risk"] = cov["weekly_population_at_risk"].fillna(0)

    cov_rows: list[dict] = []
    for group_name, bins in groups.items():
        csub = cov[cov["age_bin"].isin(bins)]
        if csub.empty:
            continue
        for iso_week, grp in csub.groupby("iso_week", sort=False):
            popw = grp["weekly_population_at_risk"]
            cov_rows.append(
                {
                    "iso_week": iso_week,
                    "age_group": group_name,
                    "coverage_ge1": _weighted_mean(grp["coverage_ge1"], popw),
                    "coverage_ge2": _weighted_mean(grp["coverage_ge2"], popw)
                    if "coverage_ge2" in grp.columns
                    else np.nan,
                    "coverage_ge3": _weighted_mean(grp["coverage_ge3"], popw)
                    if "coverage_ge3" in grp.columns
                    else np.nan,
                }
            )
    cov_grp = pd.DataFrame(cov_rows)
    if cov_grp.empty:
        return out
    return out.merge(cov_grp, on=["iso_week", "age_group"], how="left")


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    x = x[ok]
    y = y[ok]
    w = w[ok]
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    denom = np.sum(w * (x - xbar) ** 2)
    if denom <= 0:
        return ybar, float("nan")
    slope = np.sum(w * (x - xbar) * (y - ybar)) / denom
    intercept = ybar - slope * xbar
    return float(intercept), float(slope)


def _segment_stats(
    sub: pd.DataFrame,
    *,
    outcome_col: str,
    break_idx: int,
) -> dict[str, float]:
    arr = sub.sort_values("iso_week", kind="mergesort").reset_index(drop=True)
    x = np.arange(len(arr), dtype=float)
    y = pd.to_numeric(arr[outcome_col], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(arr["population_at_risk"], errors="coerce").to_numpy(dtype=float)

    pre_mask = x < break_idx
    post_mask = x >= break_idx
    pre_i, pre_s = _weighted_linear_fit(x[pre_mask], y[pre_mask], w[pre_mask])
    post_i, post_s = _weighted_linear_fit(x[post_mask], y[post_mask], w[post_mask])
    if np.isfinite(pre_i) and np.isfinite(pre_s):
        pred_at_break = pre_i + pre_s * break_idx
    else:
        pred_at_break = float("nan")
    if np.isfinite(post_i):
        post_at_break = post_i + post_s * break_idx if np.isfinite(post_s) else post_i
    else:
        post_at_break = float("nan")
    pre_mean = _weighted_mean(pd.Series(y[pre_mask]), pd.Series(w[pre_mask]))
    post_mean = _weighted_mean(pd.Series(y[post_mask]), pd.Series(w[post_mask]))
    return {
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "pre_slope_per_week": pre_s,
        "post_slope_per_week": post_s,
        "slope_change": post_s - pre_s if np.isfinite(pre_s) and np.isfinite(post_s) else np.nan,
        "level_jump_at_break": post_at_break - pred_at_break
        if np.isfinite(post_at_break) and np.isfinite(pred_at_break)
        else np.nan,
        "n_pre": int(pre_mask.sum()),
        "n_post": int(post_mask.sum()),
    }


def build_old_young_difference_weekly(
    group_weekly: pd.DataFrame,
    *,
    older_group: str,
    younger_group: str,
) -> pd.DataFrame:
    """Weekly old-minus-young differences across several outcomes."""
    old = group_weekly[group_weekly["age_group"] == older_group].copy()
    young = group_weekly[group_weekly["age_group"] == younger_group].copy()
    if old.empty or young.empty:
        return pd.DataFrame()
    cols = [
        "cases",
        "deaths_all",
        "deaths_covid",
        "deaths_non_covid",
        "population_at_risk",
        "case_rate",
        "cfr_covid_episode",
        "cfr_allcause_episode",
        "mortality_rate_all_cause",
        "mortality_rate_covid",
        "mortality_rate_non_covid",
        "coverage_ge1",
        "coverage_ge2",
        "coverage_ge3",
    ]
    keep = ["iso_week", "week_monday"] + [c for c in cols if c in old.columns]
    old = old[keep].rename(columns={c: f"{c}_old" for c in keep if c not in {"iso_week", "week_monday"}})
    young = young[keep].rename(columns={c: f"{c}_young" for c in keep if c not in {"iso_week", "week_monday"}})
    out = old.merge(young, on=["iso_week", "week_monday"], how="inner").sort_values("iso_week", kind="mergesort")
    for metric in ("case_rate", "mortality_rate_all_cause", "mortality_rate_covid", "mortality_rate_non_covid", "coverage_ge1", "coverage_ge2", "coverage_ge3"):
        old_col = f"{metric}_old"
        young_col = f"{metric}_young"
        if old_col in out.columns and young_col in out.columns:
            out[f"{metric}_diff_old_minus_young"] = out[old_col] - out[young_col]
    for metric in ("cfr_covid_episode", "cfr_allcause_episode"):
        old_col = f"{metric}_old"
        young_col = f"{metric}_young"
        if old_col in out.columns and young_col in out.columns:
            out[f"{metric}_diff_old_minus_young"] = out[old_col] - out[young_col]
    out["population_at_risk"] = (
        pd.to_numeric(out.get("population_at_risk_old"), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("population_at_risk_young"), errors="coerce").fillna(0)
    )
    return out


def build_breakpoint_tests(
    group_weekly: pd.DataFrame,
    *,
    break_iso_week: str,
    outcomes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-group slope and level tests around a candidate break week."""
    if group_weekly.empty:
        return pd.DataFrame()
    outcomes = list(outcomes or ("mortality_rate_covid", "mortality_rate_non_covid", "mortality_rate_all_cause"))
    wk = sorted(group_weekly["iso_week"].unique())
    if break_iso_week not in wk:
        return pd.DataFrame()
    break_idx = wk.index(break_iso_week)
    rows: list[dict] = []
    for age_group, sub in group_weekly.groupby("age_group", sort=False):
        for outcome_col in outcomes:
            if outcome_col not in sub.columns:
                continue
            stats = _segment_stats(sub, outcome_col=outcome_col, break_idx=break_idx)
            rows.append(
                {
                    "series_kind": "group",
                    "series_name": age_group,
                    "break_iso_week": break_iso_week,
                    "outcome": outcome_col,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def build_difference_breakpoint_tests(
    diff_weekly: pd.DataFrame,
    *,
    break_iso_week: str,
    outcomes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Slope-change tests on old-minus-young weekly differences."""
    if diff_weekly.empty:
        return pd.DataFrame()
    outcomes = list(
        outcomes
        or (
            "case_rate_diff_old_minus_young",
            "cfr_covid_episode_diff_old_minus_young",
            "mortality_rate_covid_diff_old_minus_young",
            "mortality_rate_non_covid_diff_old_minus_young",
            "mortality_rate_all_cause_diff_old_minus_young",
        )
    )
    wk = sorted(diff_weekly["iso_week"].unique())
    if break_iso_week not in wk:
        return pd.DataFrame()
    break_idx = wk.index(break_iso_week)
    rows: list[dict] = []
    for outcome_col in outcomes:
        if outcome_col not in diff_weekly.columns:
            continue
        stats = _segment_stats(diff_weekly, outcome_col=outcome_col, break_idx=break_idx)
        rows.append(
            {
                "series_kind": "old_minus_young",
                "series_name": outcome_col,
                "break_iso_week": break_iso_week,
                "outcome": outcome_col,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def build_placebo_break_scan(
    diff_weekly: pd.DataFrame,
    *,
    outcomes: Sequence[str] | None = None,
    candidate_weeks: Sequence[str] | None = None,
    min_pre_weeks: int = 6,
    min_post_weeks: int = 6,
) -> pd.DataFrame:
    """Scan many candidate break weeks to see whether the chosen break is unusual."""
    if diff_weekly.empty:
        return pd.DataFrame()
    wk = sorted(diff_weekly["iso_week"].unique())
    if candidate_weeks is None:
        candidate_weeks = wk
    outcomes = list(
        outcomes
        or (
            "case_rate_diff_old_minus_young",
            "cfr_covid_episode_diff_old_minus_young",
            "mortality_rate_covid_diff_old_minus_young",
            "mortality_rate_non_covid_diff_old_minus_young",
            "mortality_rate_all_cause_diff_old_minus_young",
        )
    )
    rows: list[dict] = []
    for break_iso_week in candidate_weeks:
        if break_iso_week not in wk:
            continue
        break_idx = wk.index(break_iso_week)
        if break_idx < min_pre_weeks or (len(wk) - break_idx) < min_post_weeks:
            continue
        for outcome_col in outcomes:
            if outcome_col not in diff_weekly.columns:
                continue
            stats = _segment_stats(diff_weekly, outcome_col=outcome_col, break_idx=break_idx)
            rows.append(
                {
                    "break_iso_week": break_iso_week,
                    "outcome": outcome_col,
                    **stats,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ("slope_change", "level_jump_at_break"):
        rank_col = f"{col}_abs_rank"
        out[rank_col] = (
            out.groupby("outcome")[col]
            .transform(lambda s: s.abs().rank(method="dense", ascending=False))
        )
    return out.sort_values(["outcome", "break_iso_week"], kind="mergesort")


def build_coverage_dilution_summary(
    weekly: pd.DataFrame,
    coverage_weekly: pd.DataFrame,
    *,
    age_groups: Mapping[str, Sequence[str]] | None = None,
    ve_assumptions: Sequence[float] | None = None,
    reference_cohort: str = "dose0",
    wave_start: str | None = None,
    wave_end: str | None = None,
) -> pd.DataFrame:
    """Expected population-level reduction under assumed individual VE and observed coverage."""
    groups = _norm_groups(age_groups)
    ve_assumptions = tuple(float(v) for v in (ve_assumptions or DEFAULT_VE_ASSUMPTIONS))
    age_sub = weekly[weekly["age_bin"] != "all"].copy()
    if age_sub.empty or coverage_weekly.empty:
        return pd.DataFrame()

    ref = age_sub[age_sub["cohort"] == reference_cohort][
        ["iso_week", "age_bin", "mortality_rate_covid", "population_at_risk"]
    ].rename(
        columns={
            "mortality_rate_covid": "ref_mortality_rate_covid",
            "population_at_risk": "ref_population_at_risk",
        }
    )
    total_pop = (
        age_sub.groupby(["iso_week", "age_bin"], as_index=False)["population_at_risk"]
        .sum()
        .rename(columns={"population_at_risk": "total_population_at_risk"})
    )
    cov = coverage_weekly[coverage_weekly["age_bin"] != "all"][
        ["iso_week", "age_bin", "coverage_ge1"]
    ].copy()
    merged = ref.merge(cov, on=["iso_week", "age_bin"], how="left").merge(
        total_pop, on=["iso_week", "age_bin"], how="left"
    )
    merged["coverage_ge1"] = merged["coverage_ge1"].fillna(0.0)
    merged["total_population_at_risk"] = merged["total_population_at_risk"].fillna(0.0)

    if wave_start and wave_end:
        wave_weeks = {
            row
            for row in sorted(merged["iso_week"].dropna().unique())
            if str(wave_start) <= str(row) <= str(wave_end)
        }
    else:
        wave_weeks = set(merged["iso_week"].dropna().unique())

    rows: list[dict] = []
    for group_name, bins in groups.items():
        g = merged[merged["age_bin"].isin(bins)].copy()
        if g.empty:
            continue
        for iso_week, grp in g.groupby("iso_week", sort=False):
            weights = grp["total_population_at_risk"]
            observed_ref = _weighted_mean(grp["ref_mortality_rate_covid"], weights)
            coverage = _weighted_mean(grp["coverage_ge1"], weights)
            base = {
                "age_group": group_name,
                "iso_week": iso_week,
                "coverage_ge1_weighted": coverage,
                "observed_reference_rate": observed_ref,
                "in_wave": bool(iso_week in wave_weeks),
            }
            for ve in ve_assumptions:
                expected_bin = grp["ref_mortality_rate_covid"] * (1.0 - grp["coverage_ge1"] * ve)
                expected_rate = _weighted_mean(expected_bin, weights)
                rr = expected_rate / observed_ref if pd.notna(observed_ref) and observed_ref > 0 else np.nan
                base[f"expected_rate_ve_{ve:.1f}"] = expected_rate
                base[f"expected_pop_reduction_ve_{ve:.1f}"] = 1.0 - rr if pd.notna(rr) else np.nan
            rows.append(base)
    weekly_out = pd.DataFrame(rows)
    if weekly_out.empty:
        return weekly_out

    summary_rows: list[dict] = []
    for group_name, grp in weekly_out[weekly_out["in_wave"]].groupby("age_group", sort=False):
        row = {"age_group": group_name}
        row["mean_wave_coverage_ge1"] = float(grp["coverage_ge1_weighted"].mean())
        for ve in ve_assumptions:
            col = f"expected_pop_reduction_ve_{ve:.1f}"
            row[col] = float(grp[col].mean()) if col in grp.columns else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return weekly_out
    summary["iso_week"] = "wave_mean"
    cols = ["iso_week", "age_group", "mean_wave_coverage_ge1"] + [
        f"expected_pop_reduction_ve_{ve:.1f}" for ve in ve_assumptions
    ]
    return pd.concat([weekly_out, summary[cols]], ignore_index=True, sort=False)


def build_multi_split_falsification_summary(
    weekly: pd.DataFrame,
    *,
    coverage_weekly: pd.DataFrame | None,
    split_definitions: Sequence[Mapping[str, object]],
    break_iso_week: str,
    placebo_start: str,
    placebo_end: str,
    ve_assumptions: Sequence[float] | None = None,
    reference_cohort: str = "dose0",
    wave_start: str | None = None,
    wave_end: str | None = None,
) -> pd.DataFrame:
    """Compare several broad age splits on the same breakpoint and dilution checks."""
    rows: list[dict] = []
    for split in split_definitions:
        split_name = str(split.get("name", "")).strip()
        younger = split.get("younger")
        older = split.get("older")
        if not split_name or not younger or not older:
            continue
        age_groups = {
            "younger": [str(x) for x in younger],
            "older": [str(x) for x in older],
        }
        group_weekly = build_age_group_weekly(
            weekly,
            age_groups=age_groups,
            coverage_weekly=coverage_weekly,
        )
        if group_weekly.empty:
            continue
        diff_weekly = build_old_young_difference_weekly(
            group_weekly,
            older_group="older",
            younger_group="younger",
        )
        diff_break = build_difference_breakpoint_tests(
            diff_weekly,
            break_iso_week=break_iso_week,
        )
        placebo_weeks = [
            w
            for w in sorted(diff_weekly["iso_week"].unique())
            if placebo_start <= str(w) <= placebo_end
        ]
        placebo = build_placebo_break_scan(
            diff_weekly,
            candidate_weeks=placebo_weeks,
        )
        cov = build_coverage_dilution_summary(
            weekly,
            coverage_weekly if coverage_weekly is not None else pd.DataFrame(),
            age_groups=age_groups,
            ve_assumptions=ve_assumptions,
            reference_cohort=reference_cohort,
            wave_start=wave_start,
            wave_end=wave_end,
        )

        for outcome in (
            "case_rate_diff_old_minus_young",
            "cfr_covid_episode_diff_old_minus_young",
            "mortality_rate_covid_diff_old_minus_young",
            "mortality_rate_non_covid_diff_old_minus_young",
            "mortality_rate_all_cause_diff_old_minus_young",
        ):
            dsub = diff_break[diff_break["outcome"] == outcome]
            if dsub.empty:
                continue
            row = dsub.iloc[0].to_dict()
            row["split_name"] = split_name
            if not placebo.empty and "outcome" in placebo.columns:
                psub = placebo[placebo["outcome"] == outcome]
                prow = psub[psub["break_iso_week"] == placebo_end]
                if not prow.empty:
                    row["placebo_rank_slope"] = prow.iloc[0].get("slope_change_abs_rank")
                    row["placebo_rank_level"] = prow.iloc[0].get("level_jump_at_break_abs_rank")
            rows.append(row)

        if not cov.empty:
            for _, crow in cov[cov["iso_week"] == "wave_mean"].iterrows():
                rows.append(
                    {
                        "split_name": split_name,
                        "series_kind": "coverage_dilution",
                        "series_name": crow.get("age_group"),
                        "break_iso_week": break_iso_week,
                        "outcome": "coverage_dilution_wave_mean",
                        "pre_mean": np.nan,
                        "post_mean": np.nan,
                        "pre_slope_per_week": np.nan,
                        "post_slope_per_week": np.nan,
                        "slope_change": np.nan,
                        "level_jump_at_break": np.nan,
                        "placebo_rank_slope": np.nan,
                        "placebo_rank_level": np.nan,
                        **{
                            k: crow.get(k)
                            for k in crow.index
                            if str(k).startswith("expected_pop_reduction_ve_")
                            or str(k) in {"mean_wave_coverage_ge1", "age_group"}
                        },
                    }
                )
    return pd.DataFrame(rows)


def build_negative_control_rank_summary(
    multi_split_summary: pd.DataFrame,
) -> pd.DataFrame:
    """One row per split with COVID, case-rate, and non-COVID breakpoint contrasts."""
    if multi_split_summary.empty:
        return pd.DataFrame()
    sub = multi_split_summary[
        multi_split_summary["series_kind"] == "old_minus_young"
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for split_name, grp in sub.groupby("split_name", sort=False):
        row: dict[str, object] = {"split_name": split_name}
        for outcome in (
            "case_rate_diff_old_minus_young",
            "mortality_rate_covid_diff_old_minus_young",
            "mortality_rate_non_covid_diff_old_minus_young",
            "mortality_rate_all_cause_diff_old_minus_young",
        ):
            s = grp[grp["outcome"] == outcome]
            if s.empty:
                continue
            r = s.iloc[0]
            prefix = outcome.replace("_diff_old_minus_young", "")
            for col in (
                "pre_mean",
                "post_mean",
                "slope_change",
                "level_jump_at_break",
                "placebo_rank_slope",
                "placebo_rank_level",
            ):
                row[f"{prefix}_{col}"] = r.get(col)
        covid_jump = pd.to_numeric(pd.Series([row.get("mortality_rate_covid_level_jump_at_break")]), errors="coerce").iloc[0]
        noncov_jump = pd.to_numeric(pd.Series([row.get("mortality_rate_non_covid_level_jump_at_break")]), errors="coerce").iloc[0]
        case_jump = pd.to_numeric(pd.Series([row.get("case_rate_level_jump_at_break")]), errors="coerce").iloc[0]
        row["covid_minus_noncovid_level_jump"] = (
            covid_jump - noncov_jump if pd.notna(covid_jump) and pd.notna(noncov_jump) else np.nan
        )
        row["covid_to_noncovid_level_jump_ratio"] = (
            covid_jump / noncov_jump if pd.notna(covid_jump) and pd.notna(noncov_jump) and noncov_jump != 0 else np.nan
        )
        row["covid_minus_case_level_jump"] = (
            covid_jump - case_jump if pd.notna(covid_jump) and pd.notna(case_jump) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ve_death_signal_bounds(
    wave_ve_summary: pd.DataFrame,
    *,
    multi_split_summary: pd.DataFrame,
    split_definitions: Sequence[Mapping[str, object]],
    compare_cohort: str = "dose2",
) -> pd.DataFrame:
    """
    Heuristic VE-death signal summary combining age-stratified cohort VE with ecological attenuation.

    This is not a formal causal bound. It is a compact consistency table:
    raw cohort VE in older ages, the share of the ecological old-young jump that looks COVID-specific
    after subtracting the non-COVID jump, and the implied attenuated VE signal from that combination.
    """
    if wave_ve_summary.empty or multi_split_summary.empty:
        return pd.DataFrame()
    neg = build_negative_control_rank_summary(multi_split_summary)
    if neg.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for split in split_definitions:
        split_name = str(split.get("name", "")).strip()
        older = [str(x) for x in split.get("older", [])]
        if not split_name or not older:
            continue
        sub = wave_ve_summary[
            (wave_ve_summary["cohort"] == compare_cohort) & (wave_ve_summary["age_bin"].isin(older))
        ].copy()
        nsub = neg[neg["split_name"] == split_name]
        if sub.empty or nsub.empty:
            continue
        nr = nsub.iloc[0]
        w = pd.to_numeric(sub["cohort_total_person_weeks"], errors="coerce").fillna(0.0)
        ve = pd.to_numeric(sub["ve_covid_death_rate"], errors="coerce")
        ok = w.gt(0) & ve.notna()
        weighted_ve = float(np.average(ve[ok], weights=w[ok])) if ok.any() else np.nan
        min_ve = float(ve[ok].min()) if ok.any() else np.nan
        max_ve = float(ve[ok].max()) if ok.any() else np.nan
        covid_jump = pd.to_numeric(pd.Series([nr.get("mortality_rate_covid_level_jump_at_break")]), errors="coerce").iloc[0]
        noncov_jump = pd.to_numeric(pd.Series([nr.get("mortality_rate_non_covid_level_jump_at_break")]), errors="coerce").iloc[0]
        covid_specific_share = (
            max(float(covid_jump - noncov_jump), 0.0) / float(covid_jump)
            if pd.notna(covid_jump) and pd.notna(noncov_jump) and covid_jump > 0
            else np.nan
        )
        attenuated_signal = (
            weighted_ve * covid_specific_share
            if pd.notna(weighted_ve) and pd.notna(covid_specific_share)
            else np.nan
        )
        rows.append(
            {
                "split_name": split_name,
                "older_bins": ",".join(older),
                "older_weighted_wave_ve_covid_death_rate": weighted_ve,
                "older_min_wave_ve_covid_death_rate": min_ve,
                "older_max_wave_ve_covid_death_rate": max_ve,
                "covid_level_jump_at_break": covid_jump,
                "noncovid_level_jump_at_break": noncov_jump,
                "covid_specific_jump_share_after_noncovid_subtraction": covid_specific_share,
                "attenuated_ve_death_signal": attenuated_signal,
            }
        )
    return pd.DataFrame(rows)


def build_incidence_severity_decomposition_summary(
    multi_split_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side case-rate, CFR, and death-rate break summary for each split."""
    if multi_split_summary.empty:
        return pd.DataFrame()
    sub = multi_split_summary[
        multi_split_summary["series_kind"] == "old_minus_young"
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    mapping = {
        "case_rate_diff_old_minus_young": "case_rate",
        "cfr_covid_episode_diff_old_minus_young": "cfr_covid_episode",
        "mortality_rate_covid_diff_old_minus_young": "mortality_rate_covid",
    }
    for split_name, grp in sub.groupby("split_name", sort=False):
        row: dict[str, object] = {"split_name": split_name}
        for outcome, prefix in mapping.items():
            s = grp[grp["outcome"] == outcome]
            if s.empty:
                continue
            r = s.iloc[0]
            for col in (
                "pre_mean",
                "post_mean",
                "slope_change",
                "level_jump_at_break",
                "placebo_rank_slope",
                "placebo_rank_level",
            ):
                row[f"{prefix}_{col}"] = r.get(col)
        case_jump = pd.to_numeric(pd.Series([row.get("case_rate_level_jump_at_break")]), errors="coerce").iloc[0]
        cfr_jump = pd.to_numeric(pd.Series([row.get("cfr_covid_episode_level_jump_at_break")]), errors="coerce").iloc[0]
        death_jump = pd.to_numeric(pd.Series([row.get("mortality_rate_covid_level_jump_at_break")]), errors="coerce").iloc[0]
        row["case_jump_abs"] = abs(case_jump) if pd.notna(case_jump) else np.nan
        row["cfr_jump_abs"] = abs(cfr_jump) if pd.notna(cfr_jump) else np.nan
        row["death_jump_abs"] = abs(death_jump) if pd.notna(death_jump) else np.nan
        if pd.notna(row["case_jump_abs"]) and pd.notna(row["cfr_jump_abs"]):
            if row["case_jump_abs"] > 3 * row["cfr_jump_abs"]:
                row["dominant_component"] = "incidence"
            elif row["cfr_jump_abs"] > 3 * row["case_jump_abs"]:
                row["dominant_component"] = "severity"
            else:
                row["dominant_component"] = "mixed"
        else:
            row["dominant_component"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def build_quantitative_scenario_bounds(
    ve_death_bounds: pd.DataFrame,
    multi_split_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transparent scenario-based VE-death summary.

    Columns are intentionally interpretable rather than "fully identified":
    - ``strict_null_floor``: conservative floor compatible with the ecological design
    - ``negative_control_adjusted_point``: attenuated signal after subtracting non-COVID jump share
    - ``coverage_calibrated_point``: COVID-specific jump share divided by older-group mean coverage
    - ``cohort_ceiling``: older-group weighted cohort VE on COVID death rate
    """
    if ve_death_bounds.empty or multi_split_summary.empty:
        return pd.DataFrame()
    cov = multi_split_summary[
        (multi_split_summary["series_kind"] == "coverage_dilution")
        & (multi_split_summary["series_name"] == "older")
    ].copy()
    if cov.empty:
        return pd.DataFrame()
    cols = [
        "split_name",
        "age_group",
        "mean_wave_coverage_ge1",
    ]
    cov = cov[[c for c in cols if c in cov.columns]].rename(columns={"age_group": "coverage_group"})
    merged = ve_death_bounds.merge(cov, on="split_name", how="left")
    if merged.empty:
        return merged
    coverage = pd.to_numeric(merged.get("mean_wave_coverage_ge1"), errors="coerce")
    share = pd.to_numeric(
        merged.get("covid_specific_jump_share_after_noncovid_subtraction"),
        errors="coerce",
    )
    ceiling = pd.to_numeric(merged.get("older_weighted_wave_ve_covid_death_rate"), errors="coerce")
    merged["strict_null_floor"] = 0.0
    merged["negative_control_adjusted_point"] = pd.to_numeric(
        merged.get("attenuated_ve_death_signal"),
        errors="coerce",
    )
    merged["coverage_calibrated_point"] = np.where(
        pd.notna(coverage) & (coverage > 0) & pd.notna(share),
        share / coverage,
        np.nan,
    )
    merged["cohort_ceiling"] = ceiling
    merged["scenario_range_lo"] = merged["strict_null_floor"]
    merged["scenario_range_hi"] = np.fmin(
        pd.to_numeric(merged["coverage_calibrated_point"], errors="coerce"),
        pd.to_numeric(merged["cohort_ceiling"], errors="coerce"),
    )
    return merged[
        [
            "split_name",
            "older_bins",
            "mean_wave_coverage_ge1",
            "strict_null_floor",
            "negative_control_adjusted_point",
            "coverage_calibrated_point",
            "cohort_ceiling",
            "scenario_range_lo",
            "scenario_range_hi",
        ]
    ].copy()

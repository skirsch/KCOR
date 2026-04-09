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
        pop = agg["population_at_risk"].astype(float)
        agg["case_rate"] = np.where(pop > 0, agg["cases"] / pop, np.nan)
        agg["mortality_rate_all_cause"] = np.where(pop > 0, agg["deaths_all"] / pop, np.nan)
        agg["mortality_rate_covid"] = np.where(pop > 0, agg["deaths_covid"] / pop, np.nan)
        agg["mortality_rate_non_covid"] = np.where(pop > 0, agg["deaths_non_covid"] / pop, np.nan)
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

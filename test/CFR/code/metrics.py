"""Weekly aggregates, rates, wave/baseline summaries, cohort ratios."""

from __future__ import annotations

import multiprocessing as mp
import sys
from datetime import date

import numpy as np
import pandas as pd

from cohort_builder import cohort_mask, iter_followup_mondays, monday_to_iso_week

# Set by build_weekly_metrics before fork Pool; children inherit via copy-on-write (Linux/WSL/macOS fork).
_METRICS_POOL_CTX: dict | None = None


def _metrics_fork_pool_available() -> bool:
    if sys.platform == "win32":
        return False
    try:
        mp.get_context("fork")
        return True
    except ValueError:
        return False


def parallel_stratum_pool_available() -> bool:
    """True if ``build_weekly_metrics(..., metrics_workers>1)`` can use a fork pool (Linux/WSL; not Windows)."""
    return _metrics_fork_pool_available()


def _pool_worker_stratum(task_key: tuple[str, str]) -> list[dict]:
    ctx = _METRICS_POOL_CTX
    if ctx is None:
        raise RuntimeError("metrics pool: missing _METRICS_POOL_CTX")
    return _compute_weekly_stratum_rows(
        ctx["df"],
        task_key[0],
        task_key[1],
        weeks=ctx["weeks"],
        wmap=ctx["wmap"],
        iso_labels=ctx["iso_labels"],
        cohort_masks=ctx.get("cohort_masks"),
    )


def _week_index_map(weeks: list[date]) -> dict[date, int]:
    return {w: i for i, w in enumerate(weeks)}


def _last_alive_week_index(death_monday: date | float, weeks: list[date], wmap: dict[date, int]) -> int:
    """Largest week index j such that person is alive at start of weeks[j]."""
    n = len(weeks)
    if death_monday is None or (isinstance(death_monday, float) and np.isnan(death_monday)):
        return n - 1
    if not isinstance(death_monday, date):
        return n - 1
    if death_monday < weeks[0]:
        return -1
    j = wmap.get(death_monday)
    if j is not None:
        return j
    for k in range(n - 1, -1, -1):
        if death_monday >= weeks[k]:
            return k
    return -1


def _pop_counts_vectorized(sub: pd.DataFrame, weeks: list[date], wmap: dict[date, int]) -> np.ndarray:
    """Population at start of each week for cohort-age subset."""
    n_weeks = len(weeks)
    if len(sub) == 0:
        return np.zeros(n_weeks, dtype=np.int64)
    last_idx = np.empty(len(sub), dtype=np.int32)
    dm = sub["death_monday_allcause"]
    for i in range(len(sub)):
        d = dm.iloc[i]
        if pd.isna(d):
            last_idx[i] = n_weeks - 1
        else:
            last_idx[i] = _last_alive_week_index(d, weeks, wmap)
    arr = np.zeros(n_weeks + 1, dtype=np.int64)
    for k in last_idx:
        if k < 0:
            continue
        arr[0] += 1
        if k + 1 < len(arr):
            arr[k + 1] -= 1
    return np.cumsum(arr[:-1])


def _iso_week_label_to_date(iso: str) -> date | None:
    ts = pd.to_datetime(iso + "-1", format="%G-%V-%u", errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def _hist_event(sub: pd.DataFrame, col: str, weeks: list[date], wmap: dict[date, int]) -> np.ndarray:
    n_weeks = len(weeks)
    h = np.zeros(n_weeks, dtype=np.int64)
    if col not in sub.columns or len(sub) == 0:
        return h
    s = sub[col].dropna()
    for d in s:
        j = wmap.get(d)
        if j is not None:
            h[j] += 1
    return h


def _episode_death_numerators(
    sub: pd.DataFrame,
    weeks: list[date],
    wmap: dict[date, int],
    death_col: str,
) -> np.ndarray:
    """
    One terminal event counted once per person (ID), assigned to the infection **episode week**
    of the **last** positive test on or before that death date (tie-break: higher ``Infection`` index).

    For ``covid_death_monday``, the death date comes from the administrative COVID death field
    (``Date_COVID_death`` / Umrti track in source data)—no lag window; the person either has a
    COVID death date or not.

    Denominator: infection episodes per week (histogram of ``infection_monday``, one count per row).
    """
    n_weeks = len(weeks)
    h = np.zeros(n_weeks, dtype=np.int64)
    if len(sub) == 0 or death_col not in sub.columns or "ID" not in sub.columns:
        return h
    dead = sub.loc[sub[death_col].notna(), ["ID", death_col]].drop_duplicates(subset=["ID"], keep="first")
    if dead.empty:
        return h
    mask_inf = sub["infection_monday"].notna()
    inf = sub.loc[mask_inf, ["ID", "infection_monday"]].copy()
    if inf.empty:
        return h
    if "Infection" in sub.columns:
        inf["_inv"] = (
            pd.to_numeric(sub.loc[mask_inf, "Infection"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
    else:
        inf["_inv"] = 0
    m = inf.merge(dead, on="ID", how="inner")
    m = m[m["infection_monday"] <= m[death_col]]
    if m.empty:
        return h
    m = m.sort_values(["ID", "infection_monday", "_inv"], ascending=[True, False, False])
    attrib = m.groupby("ID", sort=False).head(1)
    for inf_mon in attrib["infection_monday"]:
        j = wmap.get(inf_mon)
        if j is not None:
            h[j] += 1
    return h


def _add_implied_ve_columns(
    weekly: pd.DataFrame,
    *,
    reference_cohort: str,
) -> pd.DataFrame:
    """
    Non-reference rows: ve_implied_cfr_covid = 1 - cfr_covid / ref; same for all-cause.
    Reference rows get NaN. Adds cfr_covid_ref / cfr_allcause_ref for audit.
    """
    ref_df = weekly[weekly["cohort"] == reference_cohort][
        ["iso_week", "age_bin", "cfr_covid", "cfr_allcause"]
    ].drop_duplicates(subset=["iso_week", "age_bin"])
    ref_df = ref_df.rename(
        columns={"cfr_covid": "cfr_covid_ref", "cfr_allcause": "cfr_allcause_ref"}
    )
    out = weekly.merge(ref_df, on=["iso_week", "age_bin"], how="left")
    not_ref = out["cohort"] != reference_cohort
    c_v, c_u = out["cfr_covid"], out["cfr_covid_ref"]
    out["ve_implied_cfr_covid"] = np.where(
        not_ref & (c_u > 0) & pd.notna(c_v) & pd.notna(c_u),
        1.0 - (c_v / c_u),
        np.nan,
    )
    c_v2, c_u2 = out["cfr_allcause"], out["cfr_allcause_ref"]
    out["ve_implied_cfr_allcause"] = np.where(
        not_ref & (c_u2 > 0) & pd.notna(c_v2) & pd.notna(c_u2),
        1.0 - (c_v2 / c_u2),
        np.nan,
    )
    return out


def build_implied_ve_long_summary(
    weekly: pd.DataFrame,
    *,
    baseline_weeks: set[date],
    wave_weeks: set[date],
    reference_cohort: str,
) -> pd.DataFrame:
    """Long-format mean implied VE in baseline vs wave (COVID and all-cause episode CFR)."""
    rows = []
    for cohort in sorted(weekly["cohort"].unique()):
        if cohort == reference_cohort:
            continue
        for age_key in sorted(weekly["age_bin"].unique()):
            wsub = weekly[(weekly["cohort"] == cohort) & (weekly["age_bin"] == age_key)]
            base_m = wsub["iso_week"].map(lambda s: _iso_week_label_to_date(s) in baseline_weeks)
            wave_m = wsub["iso_week"].map(lambda s: _iso_week_label_to_date(s) in wave_weeks)
            base = wsub[base_m.fillna(False)]
            wav = wsub[wave_m.fillna(False)]

            def ms(s: pd.Series, col: str) -> float:
                v = s[col].dropna().values
                return float(np.nanmean(v)) if len(v) else np.nan

            rows.append(
                {
                    "cohort": cohort,
                    "age_bin": age_key,
                    "track": "covid_cfr",
                    "mean_ve_implied_baseline": ms(base, "ve_implied_cfr_covid"),
                    "mean_ve_implied_wave": ms(wav, "ve_implied_cfr_covid"),
                }
            )
            rows.append(
                {
                    "cohort": cohort,
                    "age_bin": age_key,
                    "track": "allcause_cfr",
                    "mean_ve_implied_baseline": ms(base, "ve_implied_cfr_allcause"),
                    "mean_ve_implied_wave": ms(wav, "ve_implied_cfr_allcause"),
                }
            )
    return pd.DataFrame(rows)


def _slice_for_stratum(
    df: pd.DataFrame,
    cohort: str,
    age_key: str,
    *,
    cohort_masks: dict[str, pd.Series] | None,
) -> pd.DataFrame:
    if cohort_masks is not None:
        m = cohort_masks[cohort]
    else:
        m = cohort_mask(df, cohort)
    if age_key == "all":
        return df[m].copy()
    return df[m & (df["age_bin"] == age_key)].copy()


def _compute_weekly_stratum_rows(
    df: pd.DataFrame,
    cohort: str,
    age_key: str,
    *,
    weeks: list[date],
    wmap: dict[date, int],
    iso_labels: list[str],
    cohort_masks: dict[str, pd.Series] | None,
) -> list[dict]:
    """One (cohort × age_bin) slice: episode CFR (no infection→death lag window)."""
    sub = _slice_for_stratum(df, cohort, age_key, cohort_masks=cohort_masks)
    pop = _pop_counts_vectorized(sub, weeks, wmap)
    cases = _hist_event(sub, "infection_monday", weeks, wmap)
    deaths_all = _hist_event(sub, "death_monday_allcause", weeks, wmap)
    deaths_covid = _hist_event(sub, "covid_death_monday", weeks, wmap)
    deaths_non_covid = np.maximum(deaths_all - deaths_covid, 0)

    covid_ep = _episode_death_numerators(sub, weeks, wmap, "covid_death_monday")
    all_ep = _episode_death_numerators(sub, weeks, wmap, "death_monday_allcause")

    rows: list[dict] = []
    for t, w in enumerate(weeks):
        iso = iso_labels[t]
        p = max(int(pop[t]), 0)
        c = int(cases[t])
        da = int(deaths_all[t])
        dc = int(deaths_covid[t])
        dnc = int(deaths_non_covid[t])
        case_rate = c / p if p > 0 else np.nan
        mort_rate = da / p if p > 0 else np.nan
        covid_mort_rate = dc / p if p > 0 else np.nan
        cfr_covid = float(covid_ep[t]) / c if c > 0 else np.nan
        cfr_all = float(all_ep[t]) / c if c > 0 else np.nan
        rows.append(
            {
                "iso_week": iso,
                "week_monday": w.isoformat(),
                "cohort": cohort,
                "age_bin": age_key,
                "population_at_risk": p,
                "cases": c,
                "deaths_all": da,
                "deaths_covid": dc,
                "deaths_non_covid": dnc,
                "case_rate": case_rate,
                "mortality_rate_all_cause": mort_rate,
                "mortality_rate_covid": covid_mort_rate,
                "cfr_covid": cfr_covid,
                "cfr_allcause": cfr_all,
                "decomp_mort_implied": case_rate * cfr_all if c > 0 else np.nan,
            }
        )
    return rows


def build_weekly_metrics(
    df: pd.DataFrame,
    *,
    followup_start: str,
    followup_end: str,
    cohorts: list[str],
    age_bins_config: list[list[int]],
    baseline_start: str,
    baseline_end: str,
    wave_start: str,
    wave_end: str,
    cohort_masks: dict[str, pd.Series] | None = None,
    reference_cohort: str = "dose0",
    metrics_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = iter_followup_mondays(followup_start, followup_end)
    wmap = _week_index_map(weeks)
    iso_labels = [monday_to_iso_week(w) for w in weeks]

    baseline_weeks = set(iter_followup_mondays(baseline_start, baseline_end))
    wave_weeks = set(iter_followup_mondays(wave_start, wave_end))

    age_labels = [f"{lo}-{hi}" for lo, hi in age_bins_config]
    age_labels_all = ["all"] + age_labels

    task_keys: list[tuple[str, str]] = [
        (cohort, age_key) for cohort in cohorts for age_key in age_labels_all
    ]

    global _METRICS_POOL_CTX
    use_pool = (
        metrics_workers > 1
        and len(task_keys) > 1
        and _metrics_fork_pool_available()
    )
    if metrics_workers > 1 and not _metrics_fork_pool_available():
        pass  # caller may log; fall back to sequential

    if use_pool:
        _METRICS_POOL_CTX = {
            "df": df,
            "weeks": weeks,
            "wmap": wmap,
            "iso_labels": iso_labels,
            "cohort_masks": cohort_masks,
        }
        try:
            n_proc = min(int(metrics_workers), len(task_keys))
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_proc) as pool:
                chunks = pool.map(_pool_worker_stratum, task_keys, chunksize=1)
            rows = [r for block in chunks for r in block]
        finally:
            _METRICS_POOL_CTX = None
    else:
        rows = []
        for cohort, age_key in task_keys:
            rows.extend(
                _compute_weekly_stratum_rows(
                    df,
                    cohort,
                    age_key,
                    weeks=weeks,
                    wmap=wmap,
                    iso_labels=iso_labels,
                    cohort_masks=cohort_masks,
                )
            )

    weekly = pd.DataFrame(rows)

    vs_ref = f"_vs_{reference_cohort}"
    ratio_rows = []
    if reference_cohort in cohorts:
        ref = weekly[weekly["cohort"] == reference_cohort].set_index(["iso_week", "age_bin"])
        for cohort in cohorts:
            if cohort == reference_cohort:
                continue
            cur = weekly[weekly["cohort"] == cohort]
            for _, r in cur.iterrows():
                key = (r["iso_week"], r["age_bin"])
                if key not in ref.index:
                    continue
                b = ref.loc[key]
                if isinstance(b, pd.DataFrame):
                    b = b.iloc[0]
                ratio_rows.append(
                    {
                        "iso_week": r["iso_week"],
                        "cohort": cohort,
                        "age_bin": r["age_bin"],
                        f"case_rate_ratio{vs_ref}": (
                            r["case_rate"] / b["case_rate"]
                            if pd.notna(r["case_rate"])
                            and pd.notna(b["case_rate"])
                            and b["case_rate"] > 0
                            else np.nan
                        ),
                        f"mortality_ratio{vs_ref}": (
                            r["mortality_rate_all_cause"] / b["mortality_rate_all_cause"]
                            if pd.notna(r["mortality_rate_all_cause"])
                            and pd.notna(b["mortality_rate_all_cause"])
                            and b["mortality_rate_all_cause"] > 0
                            else np.nan
                        ),
                        f"cfr_covid_ratio{vs_ref}": (
                            r["cfr_covid"] / b["cfr_covid"]
                            if pd.notna(r["cfr_covid"])
                            and pd.notna(b["cfr_covid"])
                            and b["cfr_covid"] > 0
                            else np.nan
                        ),
                    }
                )
    ratios = pd.DataFrame(ratio_rows)
    if len(ratios):
        weekly = weekly.merge(
            ratios,
            on=["iso_week", "cohort", "age_bin"],
            how="left",
        )

    weekly = _add_implied_ve_columns(weekly, reference_cohort=reference_cohort)

    def mean_safe(series: pd.Series) -> float:
        v = series.dropna().values
        return float(np.nanmean(v)) if len(v) else np.nan

    summary_rows = []
    for cohort in cohorts:
        for age_key in age_labels_all:
            wsub = weekly[(weekly["cohort"] == cohort) & (weekly["age_bin"] == age_key)]
            base_mask = wsub["iso_week"].map(lambda s: _iso_week_label_to_date(s) in baseline_weeks)
            wave_mask = wsub["iso_week"].map(lambda s: _iso_week_label_to_date(s) in wave_weeks)
            base = wsub[base_mask.fillna(False)]
            wav = wsub[wave_mask.fillna(False)]
            sl = _slice_for_stratum(df, cohort, age_key, cohort_masks=cohort_masks)
            enroll_n = len(sl)
            tot_cases = int(sl["infection_monday"].notna().sum())
            tot_deaths = int(sl["death_monday_allcause"].notna().sum())

            br = mean_safe(base["case_rate"])
            wr = mean_safe(wav["case_rate"])
            bm = mean_safe(base["mortality_rate_all_cause"])
            wm = mean_safe(wav["mortality_rate_all_cause"])
            row_sum = {
                "cohort": cohort,
                "age_bin": age_key,
                "reference_cohort_for_ratios_ve": reference_cohort,
                "enrollment_n": enroll_n,
                "total_cases_followup": tot_cases,
                "total_deaths_followup": tot_deaths,
                "mean_case_rate_baseline": br,
                "mean_case_rate_wave": wr,
                "wave_over_baseline_case_rate": wr / br if br and br > 0 else np.nan,
                "mean_mortality_baseline": bm,
                "mean_mortality_wave": wm,
                "wave_over_baseline_mortality": wm / bm if bm and bm > 0 else np.nan,
                "mean_cfr_covid_wave": mean_safe(wav["cfr_covid"]),
                "mean_cfr_covid_baseline": mean_safe(base["cfr_covid"]),
            }
            if cohort != reference_cohort:
                row_sum["mean_ve_implied_cfr_covid_baseline"] = mean_safe(base["ve_implied_cfr_covid"])
                row_sum["mean_ve_implied_cfr_covid_wave"] = mean_safe(wav["ve_implied_cfr_covid"])
                row_sum["mean_ve_implied_cfr_allcause_baseline"] = mean_safe(
                    base["ve_implied_cfr_allcause"]
                )
                row_sum["mean_ve_implied_cfr_allcause_wave"] = mean_safe(wav["ve_implied_cfr_allcause"])
            else:
                row_sum["mean_ve_implied_cfr_covid_baseline"] = np.nan
                row_sum["mean_ve_implied_cfr_covid_wave"] = np.nan
                row_sum["mean_ve_implied_cfr_allcause_baseline"] = np.nan
                row_sum["mean_ve_implied_cfr_allcause_wave"] = np.nan
            summary_rows.append(row_sum)
    cohort_summary = pd.DataFrame(summary_rows)

    return weekly, cohort_summary

"""Weekly aggregates, rates, wave/baseline summaries, cohort ratios."""

from __future__ import annotations

import multiprocessing as mp
import sys
from collections.abc import Callable
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


def _sub_one_row_per_person(sub: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ``ID`` for person-week denominators and death-week numerators.

    With ``single_infection_only: false``, the frame still has one row per infection episode; those
    extra rows must not multiply ``population_at_risk`` or duplicate the same death date. When
    ``Infection`` is present, keep the lowest episode index per ID (stable ``mergesort``).
    """
    if sub.empty or "ID" not in sub.columns:
        return sub
    s = sub
    if "Infection" in s.columns:
        inv = pd.to_numeric(s["Infection"], errors="coerce")
        s = s.assign(
            _inv_ord=inv.fillna(10**9).astype(np.int64),
        )
        s = s.sort_values(["ID", "_inv_ord"], kind="mergesort").drop(columns=["_inv_ord"])
    return s.drop_duplicates(subset=["ID"], keep="first")


def _pop_counts_vectorized(sub: pd.DataFrame, weeks: list[date], wmap: dict[date, int]) -> np.ndarray:
    """Population at start of each week; ``sub`` must be one row per person (see ``_sub_one_row_per_person``)."""
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
    sub_p = _sub_one_row_per_person(sub)
    pop = _pop_counts_vectorized(sub_p, weeks, wmap)
    cases = _hist_event(sub, "infection_monday", weeks, wmap)
    deaths_all = _hist_event(sub_p, "death_monday_allcause", weeks, wmap)
    deaths_covid = _hist_event(sub_p, "covid_death_monday", weeks, wmap)
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


# --- Period-integrated summaries (sum person-weeks and events; VE vs reference) -----------------

VE_STRONGLY_NEGATIVE_THRESHOLD = -1.0
# CFR ratios are unstable at low case counts; do not flag strongly negative CFR-VE as suspicious then.
MIN_REF_CASES_FOR_CFR_VE_STRONG_WARN = 80
MIN_COHORT_CASES_FOR_CFR_VE_STRONG_WARN = 25


def iso_weeks_in_period(period_start_iso: str, period_end_iso: str) -> set[str]:
    """ISO week labels (YYYY-WW) inclusive from config strings."""
    return {monday_to_iso_week(d) for d in iter_followup_mondays(period_start_iso, period_end_iso)}


def expected_weeks_in_period(period_start_iso: str, period_end_iso: str) -> int:
    return len(list(iter_followup_mondays(period_start_iso, period_end_iso)))


def build_period_aggregate_summary(
    weekly: pd.DataFrame,
    *,
    period_start: str,
    period_end: str,
    period_name: str,
    rate_suffix: str,
) -> pd.DataFrame:
    """
    Sum ``population_at_risk``, cases, and deaths over all ISO weeks in ``period_start``–``period_end``.

    Rate columns are named with ``rate_suffix`` (e.g. ``_wave`` → ``case_rate_wave``).
    ``total_rows`` = number of weekly rows summed per stratum (equals ``weeks_in_period`` when
    exactly one row per ISO week). ``expected_weeks_in_period`` is the calendar week count for the
    configured window (sanity vs ``weeks_in_period``).

    **Estimands:** ``case_rate*`` is infections per person-week; ``cfr_*`` is deaths per **case**
    (severity conditional on infection); ``*_death_rate*`` is deaths per person-week (population
    burden). These are not interchangeable when comparing strata or periods.
    """
    iso_set = iso_weeks_in_period(period_start, period_end)
    exp_w = expected_weeks_in_period(period_start, period_end)
    sub = weekly[weekly["iso_week"].isin(iso_set)]
    if sub.empty:
        return pd.DataFrame()

    agg_kw: dict = {
        "weeks_in_period": ("iso_week", "nunique"),
        "total_rows": ("iso_week", "count"),
        "total_person_weeks": ("population_at_risk", "sum"),
        "total_cases": ("cases", "sum"),
        "total_covid_deaths": ("deaths_covid", "sum"),
        "total_allcause_deaths": ("deaths_all", "sum"),
    }
    if "deaths_non_covid" in sub.columns:
        agg_kw["total_noncovid_deaths"] = ("deaths_non_covid", "sum")

    out = sub.groupby(["cohort", "age_bin"], sort=False, as_index=False).agg(**agg_kw)
    out["expected_weeks_in_period"] = exp_w
    out["period_start"] = period_start
    out["period_end"] = period_end
    out["period_name"] = period_name

    tpw = out["total_person_weeks"].to_numpy(dtype=float)
    tc = out["total_cases"].to_numpy(dtype=float)
    tcd = out["total_covid_deaths"].to_numpy(dtype=float)
    tad = out["total_allcause_deaths"].to_numpy(dtype=float)

    def _div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.where(den > 0, num / den, np.nan)

    sfx = rate_suffix if rate_suffix.startswith("_") else f"_{rate_suffix}"
    out[f"case_rate{sfx}"] = _div(tc, tpw)
    out[f"covid_death_rate{sfx}"] = _div(tcd, tpw)
    out[f"allcause_death_rate{sfx}"] = _div(tad, tpw)
    out[f"cfr_covid{sfx}"] = _div(tcd, tc)
    out[f"cfr_allcause{sfx}"] = _div(tad, tc)
    if "total_noncovid_deaths" in out.columns:
        tnc = out["total_noncovid_deaths"].to_numpy(dtype=float)
        out[f"noncovid_death_rate{sfx}"] = _div(tnc, tpw)

    return out


def merge_period_unique_people(summary: pd.DataFrame, df_model: pd.DataFrame) -> pd.DataFrame:
    """Add ``total_unique_people`` per (cohort, age_bin) from row-level ``df_model``."""
    if summary.empty:
        return summary
    out = summary.copy()
    if df_model.empty or "ID" not in df_model.columns:
        out["total_unique_people"] = np.nan
        return out
    rows: list[dict] = []
    for _, crow in out[["cohort", "age_bin"]].drop_duplicates().iterrows():
        cohort, ab = crow["cohort"], crow["age_bin"]
        try:
            m = cohort_mask(df_model, str(cohort))
        except (KeyError, ValueError):
            rows.append(
                {"cohort": cohort, "age_bin": ab, "total_unique_people": np.nan}
            )
            continue
        s = df_model.loc[m]
        if str(ab) != "all":
            s = s.loc[s["age_bin"] == ab]
        n = int(s["ID"].nunique()) if len(s) else 0
        rows.append({"cohort": cohort, "age_bin": ab, "total_unique_people": n})
    uc = pd.DataFrame(rows)
    if "total_unique_people" in out.columns:
        out = out.drop(columns=["total_unique_people"])
    return out.merge(uc, on=["cohort", "age_bin"], how="left")


def build_period_ve_summary(
    period_summary: pd.DataFrame,
    *,
    reference_cohort: str,
    rate_suffix: str,
) -> pd.DataFrame:
    """
    RR, VE (= 1 − RR), and absolute rate differences vs ``reference_cohort`` within each ``age_bin``.

    **Interpretation (do not conflate estimands):** ``ve_case_rate``, ``ve_cfr_covid``, and
    ``ve_covid_death_rate`` are **not expected to match**, because:

    - **Case-rate VE** (``ve_case_rate``) measures **infection incidence reduction** (per person-week).
    - **CFR VE** (``ve_cfr_covid``, ``ve_cfr_allcause``) measures **severity conditional on
      infection** (deaths per case).
    - **COVID death-rate VE** (``ve_covid_death_rate``) **combines** incidence and severity
      (deaths per person-week).

    CFR-type rates answer severity given a positive test; all-cause death rate per person-week
    answers population mortality burden.
    """
    if period_summary.empty:
        return pd.DataFrame()

    sfx = rate_suffix if rate_suffix.startswith("_") else f"_{rate_suffix}"
    rate_keys = [
        (f"case_rate{sfx}", "case_rate"),
        (f"covid_death_rate{sfx}", "covid_death_rate"),
        (f"allcause_death_rate{sfx}", "allcause_death_rate"),
        (f"cfr_covid{sfx}", "cfr_covid"),
        (f"cfr_allcause{sfx}", "cfr_allcause"),
    ]
    nc_col = f"noncovid_death_rate{sfx}"
    if nc_col in period_summary.columns:
        rate_keys.append((nc_col, "noncovid_death_rate"))

    ref = period_summary[period_summary["cohort"] == reference_cohort].copy()
    if ref.empty:
        return pd.DataFrame()
    ref_idx = ref.set_index("age_bin")

    out_rows: list[dict] = []
    for _, row in period_summary.iterrows():
        co = row["cohort"]
        if co == reference_cohort:
            continue
        ab = row["age_bin"]
        if ab not in ref_idx.index:
            continue
        r0 = ref_idx.loc[ab]
        if isinstance(r0, pd.DataFrame):
            r0 = r0.iloc[0]

        def _i(x: object) -> int:
            try:
                if pd.isna(x):
                    return 0
            except (TypeError, ValueError):
                pass
            return int(x)

        rec: dict = {
            "cohort": co,
            "age_bin": ab,
            "reference_cohort": reference_cohort,
            "ref_total_person_weeks": _i(r0["total_person_weeks"]),
            "ref_total_cases": _i(r0["total_cases"]),
            "ref_total_covid_deaths": _i(r0["total_covid_deaths"]),
            "ref_total_allcause_deaths": _i(r0["total_allcause_deaths"]),
            "cohort_total_person_weeks": _i(row["total_person_weeks"]),
            "cohort_total_cases": _i(row["total_cases"]),
            "cohort_total_covid_deaths": _i(row["total_covid_deaths"]),
            "cohort_total_allcause_deaths": _i(row["total_allcause_deaths"]),
        }
        if "total_noncovid_deaths" in row.index and "total_noncovid_deaths" in r0.index:
            rec["ref_total_noncovid_deaths"] = _i(r0["total_noncovid_deaths"])
            rec["cohort_total_noncovid_deaths"] = _i(row["total_noncovid_deaths"])

        for col, short in rate_keys:
            v, v0 = row.get(col), r0.get(col)
            try:
                fv = float(v) if pd.notna(v) else np.nan
                fv0 = float(v0) if pd.notna(v0) else np.nan
            except (TypeError, ValueError):
                fv = fv0 = np.nan
            rr = fv / fv0 if pd.notna(fv) and pd.notna(fv0) and fv0 > 0 else np.nan
            ve = 1.0 - rr if pd.notna(rr) else np.nan
            diff = fv - fv0 if pd.notna(fv) and pd.notna(fv0) else np.nan
            rec[f"rr_{short}"] = rr
            rec[f"ve_{short}"] = ve
            rec[f"diff_{short}"] = diff

        out_rows.append(rec)

    return pd.DataFrame(out_rows)


def warn_period_summary_sanity(
    period_summary: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    reference_cohort: str,
    period_iso_weeks: set[str],
    rate_suffix: str,
    period_label: str,
    log: Callable[[str], None],
) -> None:
    """
    Non-fatal checks: missing ref, bad denominators, pathological RR/VE, integrated vs mean rate.

    * **RR:** warn only if **RR < 0** (impossible for non-negative rates). **RR = 0** is allowed
      (cohort rate zero vs positive reference).
    * **Strongly negative VE** (below ``VE_STRONGLY_NEGATIVE_THRESHOLD``): for **CFR** VEs only,
      skip the warning unless reference and comparator have enough cases (stable CFR); sparse
      strata often produce extreme values without indicating a pipeline bug.
    """
    if period_summary.empty:
        log(f"[{period_label}] period summary sanity: empty table")
        return

    sfx = rate_suffix if rate_suffix.startswith("_") else f"_{rate_suffix}"
    ref = period_summary[period_summary["cohort"] == reference_cohort]
    for ab in period_summary["age_bin"].unique():
        if ab not in ref["age_bin"].values:
            log(f"[{period_label}] sanity WARNING: missing reference cohort {reference_cohort!r} age_bin={ab!r}")

    for _, row in period_summary.iterrows():
        tpw = row["total_person_weeks"]
        if tpw == 0 and (row["total_cases"] > 0 or row["total_covid_deaths"] > 0 or row["total_allcause_deaths"] > 0):
            log(
                f"[{period_label}] sanity WARNING: {row['cohort']} {row['age_bin']}: "
                "total_person_weeks=0 but cases or deaths > 0"
            )
        wk = row["weeks_in_period"]
        exp = row["expected_weeks_in_period"]
        if pd.notna(wk) and pd.notna(exp) and int(wk) != int(exp):
            log(
                f"[{period_label}] sanity WARNING: {row['cohort']} {row['age_bin']}: "
                f"weeks_in_period={int(wk)} != expected_weeks_in_period={int(exp)} (truncation?)"
            )

    ve_summary = build_period_ve_summary(
        period_summary, reference_cohort=reference_cohort, rate_suffix=rate_suffix
    )
    for _, vr in ve_summary.iterrows():
        ref_cases = int(vr.get("ref_total_cases", 0) or 0)
        co_cases = int(vr.get("cohort_total_cases", 0) or 0)
        for c in vr.index:
            if c.startswith("rr_"):
                val = vr[c]
                # RR == 0 is valid (zero cohort rate vs positive reference), not pathological.
                if pd.notna(val) and float(val) < 0:
                    log(
                        f"[{period_label}] sanity WARNING: {vr['cohort']} {vr['age_bin']} {c}={float(val):.4f} (<0)"
                    )
                continue
            if not c.startswith("ve_"):
                continue
            val = vr[c]
            if not pd.notna(val):
                continue
            val = float(val)
            if val > 1.0:
                log(
                    f"[{period_label}] sanity WARNING: {vr['cohort']} {vr['age_bin']} {c}={val:.4f} (>1)"
                )
            elif val < VE_STRONGLY_NEGATIVE_THRESHOLD:
                if c in ("ve_cfr_covid", "ve_cfr_allcause"):
                    if (
                        ref_cases < MIN_REF_CASES_FOR_CFR_VE_STRONG_WARN
                        or co_cases < MIN_COHORT_CASES_FOR_CFR_VE_STRONG_WARN
                    ):
                        continue
                log(
                    f"[{period_label}] sanity WARNING: {vr['cohort']} {vr['age_bin']} {c}={val:.4f} "
                    f"(strongly negative < {VE_STRONGLY_NEGATIVE_THRESHOLD})"
                )

    wsub = weekly[weekly["iso_week"].isin(period_iso_weeks)]
    for (co, ab), grp in wsub.groupby(["cohort", "age_bin"], sort=False):
        if grp.empty:
            continue
        mean_m = float(grp["mortality_rate_all_cause"].mean())
        ps = period_summary[(period_summary["cohort"] == co) & (period_summary["age_bin"] == ab)]
        if ps.empty:
            continue
        tpw = float(ps["total_person_weeks"].iloc[0])
        tad = float(ps["total_allcause_deaths"].iloc[0])
        integ = tad / tpw if tpw > 0 else np.nan
        if pd.notna(mean_m) and pd.notna(integ) and mean_m > 0:
            rel = abs(mean_m - integ) / mean_m
            if rel > 0.01:
                log(
                    f"[{period_label}] sanity NOTE: {co} {ab}: mean weekly mortality_rate_all_cause "
                    f"({mean_m:.6f}) vs total_allcause_deaths/total_person_weeks ({integ:.6f}) "
                    f"rel_diff={rel:.4f} (Jensen / varying denominators; not necessarily an error)"
                )


def log_period_aggregate_for_console(
    period_summary: pd.DataFrame,
    log: Callable[[str], None],
    *,
    title: str,
    rate_suffix: str,
    wave_ve_summary: pd.DataFrame | None = None,
    reference_cohort: str = "dose0",
    compare_cohort: str = "dose2",
    include_age_strata: bool = True,
) -> None:
    """
    Print integrated totals and rates for ``age_bin=='all'``; optional VE line for compare vs ref.

    If ``include_age_strata`` and ``wave_ve_summary`` are set, logs one compact VE line per
    ``age_bin`` other than ``all`` for ``compare_cohort`` vs ``reference_cohort``.
    """
    if period_summary.empty:
        return
    sfx = rate_suffix if rate_suffix.startswith("_") else f"_{rate_suffix}"
    sub = period_summary[period_summary["age_bin"] == "all"].sort_values("cohort")
    log(f"Period integrated summary — {title} (age_bin=all)")

    def _fmt8(x: object) -> str:
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "nan"
            if pd.isna(x):
                return "nan"
        except (TypeError, ValueError):
            return "nan"
        return f"{float(x):.8f}"

    for _, r in sub.iterrows():
        log(
            f"  {r['cohort']}: total_cases={int(r['total_cases']):,} "
            f"total_covid_deaths={int(r['total_covid_deaths']):,} "
            f"total_allcause_deaths={int(r['total_allcause_deaths']):,} "
            f"total_person_weeks={int(r['total_person_weeks']):,} "
            f"cfr_covid{sfx}={_fmt8(r.get(f'cfr_covid{sfx}'))} "
            f"covid_death_rate{sfx}={_fmt8(r.get(f'covid_death_rate{sfx}'))} "
            f"allcause_death_rate{sfx}={_fmt8(r.get(f'allcause_death_rate{sfx}'))}"
        )

    if wave_ve_summary is None or wave_ve_summary.empty:
        return
    wv = wave_ve_summary[
        (wave_ve_summary["age_bin"] == "all") & (wave_ve_summary["cohort"] == compare_cohort)
    ]
    if wv.empty:
        return
    w = wv.iloc[0]

    def _fmt6(k: str) -> str:
        v = w.get(k, np.nan)
        return f"{float(v):.6f}" if pd.notna(v) else "nan"

    log(
        f"  implied VE vs {reference_cohort} ({compare_cohort}, age_bin=all): "
        f"ve_case_rate={_fmt6('ve_case_rate')} "
        f"ve_cfr_covid={_fmt6('ve_cfr_covid')} "
        f"ve_covid_death_rate={_fmt6('ve_covid_death_rate')} "
        f"ve_allcause_death_rate={_fmt6('ve_allcause_death_rate')} "
        f"ve_cfr_allcause={_fmt6('ve_cfr_allcause')}"
    )

    if not include_age_strata:
        return

    def _fmt6_row(row: pd.Series, k: str) -> str:
        v = row.get(k, np.nan)
        return f"{float(v):.6f}" if pd.notna(v) else "nan"

    ages = sorted(
        wave_ve_summary["age_bin"].unique(),
        key=lambda x: (str(x) == "all", str(x)),
    )
    for ab in ages:
        if ab == "all":
            continue
        wrows = wave_ve_summary[
            (wave_ve_summary["age_bin"] == ab)
            & (wave_ve_summary["cohort"] == compare_cohort)
        ]
        if wrows.empty:
            continue
        rw = wrows.iloc[0]
        log(
            f"  implied VE vs {reference_cohort} ({compare_cohort}, age_bin={ab}): "
            f"ve_case_rate={_fmt6_row(rw, 've_case_rate')} "
            f"ve_cfr_covid={_fmt6_row(rw, 've_cfr_covid')} "
            f"ve_covid_death_rate={_fmt6_row(rw, 've_covid_death_rate')} "
            f"ve_allcause_death_rate={_fmt6_row(rw, 've_allcause_death_rate')} "
            f"ve_cfr_allcause={_fmt6_row(rw, 've_cfr_allcause')}"
        )

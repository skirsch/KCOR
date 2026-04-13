#!/usr/bin/env python3
"""
Czech CFR / infection / mortality pipeline entry point.

Expects the national CSV at data/Czech/records.csv (53 columns, ISO-week dates).
That file is not committed; use --smoke or data/Czech/records_100k.csv for quick tests.
Runtime is roughly linear in row count (on the order of minutes for full population on a laptop).

Kaplan–Meier outputs need ``lifelines`` in the project venv: from repo root run
``.venv/bin/python test/CFR/run_cfr_analysis.py`` (Linux/WSL) or
``.venv\\Scripts\\python.exe test/CFR/run_cfr_analysis.py`` (Windows), or ``test/CFR/run_with_venv.sh``.

Optional landmark all-cause KM from a fixed ISO week (``extensions.km_landmark_survival``) writes
per-age ``km_landmark_<week>_first_mfg_age_<bin>.png`` and ``..._dose_censor_age_<bin>.png``, plus long
CSVs ``km_landmark_<week>_first_mfg_by_age_bin.csv`` and ``..._dose_censor_by_age_bin.csv``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CFR_ROOT = Path(__file__).resolve().parent
CODE_DIR = CFR_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from analysis import (  # noqa: E402
    build_infected_cohort_age_composition_table,
    build_km_post_infection_age_bin_table,
    build_km_post_infection_table,
    run_time_since_dose2_analysis,
    stability_check_quiet_period,
)
from km_landmark import (  # noqa: E402
    DOSE_COHORTS,
    FIRST_MFG_COHORTS,
    build_km_landmark_dose_nextdose_censor_table,
    build_km_landmark_first_mfg_table,
    resolve_landmark_age_bins,
)
from cohort_builder import (  # noqa: E402
    PRIMARY_ENROLLMENT_COHORTS,
    add_prior_infection_before_enrollment_flag,
    build_enrollment_table,
    cohort_mask,
    iter_followup_mondays,
    monday_to_iso_week,
)
from coverage import (  # noqa: E402
    build_vaccine_coverage_summary,
    build_weekly_vaccine_coverage,
    log_vaccine_coverage_console,
)
from falsification import (  # noqa: E402
    build_age_group_weekly,
    build_breakpoint_tests,
    build_coverage_dilution_summary,
    build_difference_breakpoint_tests,
    build_incidence_severity_decomposition_summary,
    build_multi_split_falsification_summary,
    build_negative_control_rank_summary,
    build_old_young_difference_weekly,
    build_placebo_break_scan,
    build_quantitative_scenario_bounds,
    build_ve_death_signal_bounds,
)
from load_data import load_czech_records, validate_loaded_schema  # noqa: E402
from metrics import (  # noqa: E402
    build_implied_ve_long_summary,
    build_period_aggregate_summary,
    build_period_ve_summary,
    build_weekly_metrics,
    enrolled_covid_death_histogram_by_week,
    iso_weeks_in_period,
    log_period_aggregate_for_console,
    merge_period_unique_people,
    parallel_stratum_pool_available,
    warn_period_summary_sanity,
)
from qa_summary import log_qa_summary, write_debug_birth_cohort_weekly_csv  # noqa: E402
from plots import (  # noqa: E402
    plot_case_rate,
    plot_cfr,
    plot_covid_mortality_per_million,
    plot_cumulative_cases_deaths,
    plot_debug_weekly_covid_deaths,
    plot_debug_weekly_covid_deaths_enrolled_total,
    plot_debug_weekly_population_at_risk,
    plot_decomposition_compare,
    plot_expected_vs_observed,
    plot_expected_vs_observed_cumulative_only,
    plot_km_post_infection,
    plot_km_survival_curves,
    plot_mortality_rate,
    plot_old_young_difference,
    plot_placebo_break_scan,
    plot_vaccine_coverage_all,
    plot_vaccine_coverage_by_age,
    plot_wave_ve_summary,
)
from simulate_expected import compute_expected_vs_observed  # noqa: E402


def _ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[cfr {_ts()}] {msg}", flush=True)


def _resolve(p: str, root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Czech CFR / case / mortality analysis")
    ap.add_argument(
        "--config",
        default=str(CFR_ROOT / "config" / "czech.yaml"),
        help="YAML config path",
    )
    ap.add_argument("--input", default=None, help="Override dataset CSV path")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Use data/Czech/records_100k.csv (ignores dataset.path unless --input is set)",
    )
    ap.add_argument(
        "--metrics-workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel (cohort×age) strata in build_weekly_metrics via fork (Linux/WSL). "
        "0 = all CPUs. Default: CFR_METRICS_WORKERS env or cfr.metrics_workers in YAML or 1.",
    )
    args = ap.parse_args()
    t0 = time.monotonic()

    cfg_path = Path(args.config).resolve()
    _log(f"loading config {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.input:
        data_path = _resolve(str(args.input), REPO_ROOT)
    elif args.smoke:
        data_path = _resolve("data/Czech/records_100k.csv", REPO_ROOT)
        _log("smoke mode: using data/Czech/records_100k.csv")
    else:
        data_path = _resolve(str(cfg["dataset"]["path"]), REPO_ROOT)

    out_dir = _resolve(cfg.get("output_dir", "test/CFR/out"), REPO_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"output directory {out_dir}")

    exp_cfg = cfg.get("expected_deaths", {})
    ref_cohort = str(exp_cfg.get("reference_cohort", "dose0"))
    vax_cohort = str(exp_cfg.get("vaccinated_cohort", "dose2"))
    _log(
        f"comparison cohorts: reference={ref_cohort}, vaccinated={vax_cohort} "
        f"(expected-vs-observed + ratios + implied VE)"
    )

    repo_code = REPO_ROOT / "code"
    filt = cfg.get("filters", {})
    cfr_cfg = cfg.get("cfr") or {}
    if "restrict_to_pfizer_moderna" in filt:
        restrict_pm = bool(filt["restrict_to_pfizer_moderna"])
    elif "non_mrna_only" in filt:
        restrict_pm = bool(filt["non_mrna_only"])
    else:
        restrict_pm = True
    # Default: keep every infection row (Infection 1, 2, 3, …). Set single_infection_only: true to drop >1.
    single_inf = bool(filt.get("single_infection_only", False))
    if cfg.get("reinfection_sensitivity") or filt.get("reinfection_sensitivity", False):
        single_inf = False

    _log(f"repo_root={REPO_ROOT}")
    _log(f"reading {data_path}")
    if restrict_pm:
        _log(
            "restrict_to_pfizer_moderna=True "
            "(unvaccinated + Pfizer/Moderna on doses 1–4 only; drops AZ/J&J/Novavax/other)"
        )
    else:
        _log("restrict_to_pfizer_moderna=False (all manufacturer codes kept)")
    _log(f"single_infection_only={single_inf}")

    _log("load_czech_records …")
    df = load_czech_records(
        data_path,
        restrict_to_pfizer_moderna=restrict_pm,
        single_infection_only=single_inf,
        repo_code_dir=repo_code,
    )
    validate_loaded_schema(df, stage="load")
    _log(f"rows after load: {len(df):,}")

    _log("reading cohort / follow-up settings from config …")
    cohort_cfg = cfg["cohort"]
    enrollment_week = str(cohort_cfg["enrollment_week"])
    followup_start = str(cohort_cfg["followup_start"])
    followup_end = str(cohort_cfg["followup_end"])
    cohorts = list(cfg["cohorts"])
    _extra_cohorts = [c for c in cohorts if c not in PRIMARY_ENROLLMENT_COHORTS]
    if _extra_cohorts:
        _log(
            f"note: enrollment-era design uses {sorted(PRIMARY_ENROLLMENT_COHORTS)} only; "
            f"ignoring cfg cohorts {_extra_cohorts}"
        )
    cohorts = [c for c in cohorts if c in PRIMARY_ENROLLMENT_COHORTS]
    if not cohorts:
        raise ValueError(
            "cfg['cohorts'] must include at least one of dose0, dose1, dose2 "
            f"(got {cfg['cohorts']!r})"
        )
    age_bins_config = [list(x) for x in cfg["age_bins"]]
    age_labels = [f"{lo}-{hi}" for lo, hi in age_bins_config]

    def _resolve_metrics_workers() -> int:
        if args.metrics_workers is not None:
            w = int(args.metrics_workers)
        elif os.environ.get("CFR_METRICS_WORKERS", "").strip() != "":
            w = int(os.environ["CFR_METRICS_WORKERS"])
        else:
            w = int(cfr_cfg.get("metrics_workers", 1))
        if w == 0:
            return max(1, (os.cpu_count() or 1))
        return max(1, w)

    metrics_workers = _resolve_metrics_workers()
    baseline = cfg["baseline"]
    wave = cfg["wave"]

    _log(
        f"build_enrollment_table (enrollment_week={enrollment_week}; "
        "vectorized date parse like KCOR_CMR) …"
    )
    df = build_enrollment_table(
        df,
        enrollment_week=enrollment_week,
        age_bins=age_bins_config,
        progress_log=_log,
    )
    df = add_prior_infection_before_enrollment_flag(df, progress_log=_log)
    validate_loaded_schema(df, stage="enrollment")

    n_total = len(df)
    excluded_age = int(df["age_bin"].isna().sum())
    _log(f"excluded missing age_bin (overall): {excluded_age} / {n_total}")

    for c in cohorts:
        m = cohort_mask(df, c)
        n_flag = int(m.sum())
        n_miss_age = int((m & df["age_bin"].isna()).sum())
        n_age = int((m & df["age_bin"].notna()).sum())
        _log(
            f"cohort {c}: N={n_flag} at enrollment; "
            f"excluded_no_age_bin={n_miss_age}; N_with_age_bin={n_age}"
        )

    df_model = df[df["age_bin"].notna()].copy()
    if df_model["covid_death_monday"].notna().sum() == 0:
        _log(
            "warning: no parseable COVID death dates (Date_COVID_death); "
            "COVID CFR / expected-deaths COVID track may be sparse or NaN"
        )

    cov_cfg = cfg.get("coverage") or {}
    if cov_cfg.get("enabled", True):
        _log("build_weekly_vaccine_coverage …")
        cov_df = build_weekly_vaccine_coverage(
            df_model,
            followup_start=followup_start,
            followup_end=followup_end,
            age_bins_config=age_bins_config,
        )
        if cov_df.empty:
            _log("vaccine coverage: empty (no follow-up weeks or no rows) — skip CSV/plots")
        else:
            cov_df.to_csv(out_dir / "vaccine_coverage_weekly.csv", index=False)
            cov_sum = build_vaccine_coverage_summary(
                cov_df, wave=wave, baseline=baseline
            )
            cov_sum.to_csv(out_dir / "vaccine_coverage_summary.csv", index=False)
            plot_dpi = int(cfg.get("plots", {}).get("dpi", 120))
            if cov_cfg.get("save_plots", True):
                plot_vaccine_coverage_by_age(
                    cov_df, out_dir / "vaccine_coverage_by_age.png", dpi=plot_dpi
                )
                plot_vaccine_coverage_all(
                    cov_df, out_dir / "vaccine_coverage_all.png", dpi=plot_dpi
                )
            log_vaccine_coverage_console(
                cov_df,
                _log,
                wave_start=str(wave["start"]),
                baseline_start=str(baseline["start"]),
            )
    else:
        cov_df = pd.DataFrame()

    if metrics_workers > 1:
        if parallel_stratum_pool_available():
            _log(
                f"build_weekly_metrics (follow-up {followup_start}–{followup_end}, "
                f"reference_cohort={ref_cohort}, metrics_workers={metrics_workers}, fork pool) …"
            )
        else:
            _log(
                f"build_weekly_metrics: metrics_workers={metrics_workers} ignored (no fork on this OS); "
                "sequential strata …"
            )
    else:
        _log(
            f"build_weekly_metrics (follow-up {followup_start}–{followup_end}, "
            f"reference_cohort={ref_cohort}) …"
        )
    weekly, cohort_summary = build_weekly_metrics(
        df_model,
        followup_start=followup_start,
        followup_end=followup_end,
        cohorts=cohorts,
        age_bins_config=age_bins_config,
        baseline_start=str(baseline["start"]),
        baseline_end=str(baseline["end"]),
        wave_start=str(wave["start"]),
        wave_end=str(wave["end"]),
        reference_cohort=ref_cohort,
        metrics_workers=metrics_workers,
    )

    assert (weekly["population_at_risk"] >= 0).all(), "negative population"
    assert (weekly["cases"] >= 0).all()
    assert (weekly["deaths_all"] >= 0).all()
    zpop = weekly.loc[weekly["population_at_risk"] == 0, "iso_week"].nunique()
    if zpop:
        _log(f"note: {zpop} distinct iso_weeks have some strata with population_at_risk==0")
    _log(
        f"weekly population_at_risk min={weekly['population_at_risk'].min()} "
        f"max={weekly['population_at_risk'].max()}"
    )

    _qs_full = cfg.get("qa_summary")
    _qs_dict = _qs_full if isinstance(_qs_full, dict) else {}
    wave_cfg = cfg.get("wave") or {}
    qa_period_start = str(
        _qs_dict.get("period_start", wave_cfg.get("start", followup_start))
    )
    qa_period_end = str(_qs_dict.get("period_end", wave_cfg.get("end", followup_end)))
    qa_cfg = _qs_full if isinstance(_qs_full, dict) else None
    if args.smoke and qa_cfg and isinstance(qa_cfg.get("spot_check"), dict):
        # Subsample will not match full-population LPZ expected counts
        qa_cfg = {
            **qa_cfg,
            "spot_check": {k: v for k, v in qa_cfg["spot_check"].items() if k != "expected"},
        }
    log_qa_summary(
        df_model,
        weekly,
        cohort_followup_start=followup_start,
        cohort_followup_end=followup_end,
        qa_period_start=qa_period_start,
        qa_period_end=qa_period_end,
        log=_log,
        qa_cfg=qa_cfg,
        df_enrollment_all=df,
    )

    _dbg_csv = _qs_dict.get("debug_enrollment_weekly_csv")
    _dbg_on = False
    _dbg_lo = _dbg_hi = 1930
    _dbg_fn = "debug_193x_enrollment_weekly.csv"
    if _dbg_csv is True:
        _dbg_on = True
        _dbg_lo, _dbg_hi = 1930, 1939
    elif isinstance(_dbg_csv, dict) and _dbg_csv.get("enabled"):
        _dbg_on = True
        try:
            _dbg_lo = int(_dbg_csv["birth_year_min"])
            _dbg_hi = int(_dbg_csv["birth_year_max"])
        except (KeyError, TypeError, ValueError):
            _dbg_on = False
            _log("debug_enrollment_weekly_csv: enabled but invalid birth_year_min/max — skip")
        _dbg_fn = str(_dbg_csv.get("filename", _dbg_fn))
    if _dbg_on:
        write_debug_birth_cohort_weekly_csv(
            df_model,
            followup_start=followup_start,
            followup_end=followup_end,
            birth_year_min=_dbg_lo,
            birth_year_max=_dbg_hi,
            cohorts=cohorts,
            out_path=out_dir / _dbg_fn,
            log=_log,
        )

    baseline_week_dates = set(
        iter_followup_mondays(str(baseline["start"]), str(baseline["end"]))
    )
    wave_week_dates = set(iter_followup_mondays(str(wave["start"]), str(wave["end"])))
    _log("build_implied_ve_long_summary …")
    implied_ve_long = build_implied_ve_long_summary(
        weekly,
        baseline_weeks=baseline_week_dates,
        wave_weeks=wave_week_dates,
        reference_cohort=ref_cohort,
    )

    wave_iso_set = iso_weeks_in_period(str(wave["start"]), str(wave["end"]))
    baseline_iso_set = iso_weeks_in_period(str(baseline["start"]), str(baseline["end"]))

    _log("period-integrated wave / baseline summaries …")
    wave_summary = build_period_aggregate_summary(
        weekly,
        period_start=str(wave["start"]),
        period_end=str(wave["end"]),
        period_name="wave",
        rate_suffix="_wave",
    )
    wave_summary = merge_period_unique_people(wave_summary, df_model)
    baseline_summary = build_period_aggregate_summary(
        weekly,
        period_start=str(baseline["start"]),
        period_end=str(baseline["end"]),
        period_name="baseline",
        rate_suffix="_baseline",
    )
    baseline_summary = merge_period_unique_people(baseline_summary, df_model)

    wave_ve_summary = build_period_ve_summary(
        wave_summary, reference_cohort=ref_cohort, rate_suffix="_wave"
    )
    baseline_ve_summary = build_period_ve_summary(
        baseline_summary, reference_cohort=ref_cohort, rate_suffix="_baseline"
    )

    warn_period_summary_sanity(
        wave_summary,
        weekly,
        reference_cohort=ref_cohort,
        period_iso_weeks=wave_iso_set,
        rate_suffix="_wave",
        period_label="wave",
        log=_log,
    )
    warn_period_summary_sanity(
        baseline_summary,
        weekly,
        reference_cohort=ref_cohort,
        period_iso_weeks=baseline_iso_set,
        rate_suffix="_baseline",
        period_label="baseline",
        log=_log,
    )

    _log_period_ve_by_age = bool(cfr_cfg.get("log_period_summary_ve_by_age_bin", True))
    log_period_aggregate_for_console(
        wave_summary,
        _log,
        title=f"{wave['start']} to {wave['end']} (wave)",
        rate_suffix="_wave",
        wave_ve_summary=wave_ve_summary,
        reference_cohort=ref_cohort,
        compare_cohort=vax_cohort,
        include_age_strata=_log_period_ve_by_age,
    )
    log_period_aggregate_for_console(
        baseline_summary,
        _log,
        title=f"{baseline['start']} to {baseline['end']} (baseline)",
        rate_suffix="_baseline",
        wave_ve_summary=baseline_ve_summary,
        reference_cohort=ref_cohort,
        compare_cohort=vax_cohort,
        include_age_strata=_log_period_ve_by_age,
    )

    _log("writing CSVs …")
    weekly.to_csv(out_dir / "weekly_metrics.csv", index=False)
    cohort_summary.to_csv(out_dir / "cohort_summary.csv", index=False)
    implied_ve_long.to_csv(out_dir / "implied_ve_summary.csv", index=False)
    wave_summary.to_csv(out_dir / "wave_summary.csv", index=False)
    baseline_summary.to_csv(out_dir / "baseline_summary.csv", index=False)
    wave_ve_summary.to_csv(out_dir / "wave_ve_summary.csv", index=False)
    baseline_ve_summary.to_csv(out_dir / "baseline_ve_summary.csv", index=False)

    age_only = weekly[weekly["age_bin"] != "all"].copy()
    age_only.to_csv(out_dir / "age_stratified_metrics.csv", index=False)

    fals_cfg = cfg.get("falsification") or {}
    older_group = str(fals_cfg.get("older_group_name", "older"))
    younger_group = str(fals_cfg.get("younger_group_name", "younger"))
    fals_age_groups = fals_cfg.get(
        "age_groups",
        {
            younger_group: ["40-49", "50-59", "60-69"],
            older_group: ["70-120"],
        },
    )
    break_iso_week = str(fals_cfg.get("break_iso_week", wave["start"]))
    placebo_start = str(fals_cfg.get("placebo_start", baseline["start"]))
    placebo_end = str(fals_cfg.get("placebo_end", baseline["end"]))
    ve_assumptions = fals_cfg.get("ve_assumptions", [0.3, 0.5, 0.7, 0.9])
    split_definitions = fals_cfg.get(
        "split_definitions",
        [
            {"name": "70plus_vs_under70", "older": ["70-120"], "younger": ["40-49", "50-59", "60-69"]},
            {"name": "60plus_vs_under60", "older": ["60-69", "70-120"], "younger": ["40-49", "50-59"]},
            {"name": "50plus_vs_under50", "older": ["50-59", "60-69", "70-120"], "younger": ["40-49"]},
        ],
    )
    _log(
        f"falsification checks (break={break_iso_week}, placebo={placebo_start}–{placebo_end}, "
        f"groups={list(fals_age_groups.keys())}) …"
    )
    fals_group_weekly = build_age_group_weekly(
        weekly,
        age_groups=fals_age_groups,
        coverage_weekly=cov_df,
    )
    fals_diff_weekly = build_old_young_difference_weekly(
        fals_group_weekly,
        older_group=older_group,
        younger_group=younger_group,
    )
    fals_break_tests = build_breakpoint_tests(
        fals_group_weekly,
        break_iso_week=break_iso_week,
    )
    fals_diff_tests = build_difference_breakpoint_tests(
        fals_diff_weekly,
        break_iso_week=break_iso_week,
    )
    placebo_weeks = [
        w
        for w in sorted(fals_diff_weekly["iso_week"].unique())
        if placebo_start <= str(w) <= placebo_end
    ] if len(fals_diff_weekly) else []
    fals_placebo = build_placebo_break_scan(
        fals_diff_weekly,
        candidate_weeks=placebo_weeks,
    )
    fals_cov = build_coverage_dilution_summary(
        weekly,
        cov_df,
        age_groups=fals_age_groups,
        ve_assumptions=ve_assumptions,
        reference_cohort=ref_cohort,
        wave_start=str(wave["start"]),
        wave_end=str(wave["end"]),
    )
    fals_multi = build_multi_split_falsification_summary(
        weekly,
        coverage_weekly=cov_df,
        split_definitions=split_definitions,
        break_iso_week=break_iso_week,
        placebo_start=placebo_start,
        placebo_end=placebo_end,
        ve_assumptions=ve_assumptions,
        reference_cohort=ref_cohort,
        wave_start=str(wave["start"]),
        wave_end=str(wave["end"]),
    )
    fals_group_weekly.to_csv(out_dir / "falsification_group_weekly.csv", index=False)
    fals_diff_weekly.to_csv(out_dir / "falsification_old_young_diff_weekly.csv", index=False)
    fals_break_tests.to_csv(out_dir / "falsification_break_tests.csv", index=False)
    fals_diff_tests.to_csv(out_dir / "falsification_diff_break_tests.csv", index=False)
    fals_placebo.to_csv(out_dir / "falsification_placebo_scan.csv", index=False)
    fals_cov.to_csv(out_dir / "falsification_coverage_dilution.csv", index=False)
    fals_multi.to_csv(out_dir / "falsification_multi_split_summary.csv", index=False)
    fals_neg = build_negative_control_rank_summary(fals_multi)
    fals_neg.to_csv(out_dir / "falsification_negative_control_summary.csv", index=False)
    fals_decomp = build_incidence_severity_decomposition_summary(fals_multi)
    fals_decomp.to_csv(out_dir / "falsification_incidence_severity_summary.csv", index=False)
    fals_bounds = build_ve_death_signal_bounds(
        wave_ve_summary,
        multi_split_summary=fals_multi,
        split_definitions=split_definitions,
        compare_cohort=vax_cohort,
    )
    fals_bounds.to_csv(out_dir / "falsification_ve_death_bounds.csv", index=False)
    fals_quant = build_quantitative_scenario_bounds(fals_bounds, fals_multi)
    fals_quant.to_csv(out_dir / "falsification_quantitative_scenario_bounds.csv", index=False)

    _log("falsification checks (infection-naive at enrollment only) …")
    naive_mask = ~df_model["prior_infection_before_enrollment"].fillna(False)
    naive_rows = int(naive_mask.sum())
    naive_people = int(df_model.loc[naive_mask, "ID"].nunique()) if "ID" in df_model.columns else naive_rows
    _log(f"  infection-naive rows={naive_rows:,}; infection-naive unique IDs={naive_people:,}")
    naive_cohort_masks = {
        c: cohort_mask(df_model, c) & naive_mask
        for c in cohorts
    }
    weekly_naive, _ = build_weekly_metrics(
        df_model,
        followup_start=followup_start,
        followup_end=followup_end,
        cohorts=cohorts,
        age_bins_config=age_bins_config,
        baseline_start=str(baseline["start"]),
        baseline_end=str(baseline["end"]),
        wave_start=str(wave["start"]),
        wave_end=str(wave["end"]),
        cohort_masks=naive_cohort_masks,
        reference_cohort=ref_cohort,
        metrics_workers=metrics_workers,
    )
    fals_group_weekly_naive = build_age_group_weekly(
        weekly_naive,
        age_groups=fals_age_groups,
        coverage_weekly=cov_df,
    )
    fals_diff_weekly_naive = build_old_young_difference_weekly(
        fals_group_weekly_naive,
        older_group=older_group,
        younger_group=younger_group,
    )
    fals_break_tests_naive = build_breakpoint_tests(
        fals_group_weekly_naive,
        break_iso_week=break_iso_week,
    )
    fals_diff_tests_naive = build_difference_breakpoint_tests(
        fals_diff_weekly_naive,
        break_iso_week=break_iso_week,
    )
    placebo_weeks_naive = [
        w
        for w in sorted(fals_diff_weekly_naive["iso_week"].unique())
        if placebo_start <= str(w) <= placebo_end
    ] if len(fals_diff_weekly_naive) else []
    fals_placebo_naive = build_placebo_break_scan(
        fals_diff_weekly_naive,
        candidate_weeks=placebo_weeks_naive,
    )
    fals_cov_naive = build_coverage_dilution_summary(
        weekly_naive,
        cov_df,
        age_groups=fals_age_groups,
        ve_assumptions=ve_assumptions,
        reference_cohort=ref_cohort,
        wave_start=str(wave["start"]),
        wave_end=str(wave["end"]),
    )
    fals_multi_naive = build_multi_split_falsification_summary(
        weekly_naive,
        coverage_weekly=cov_df,
        split_definitions=split_definitions,
        break_iso_week=break_iso_week,
        placebo_start=placebo_start,
        placebo_end=placebo_end,
        ve_assumptions=ve_assumptions,
        reference_cohort=ref_cohort,
        wave_start=str(wave["start"]),
        wave_end=str(wave["end"]),
    )
    weekly_naive.to_csv(out_dir / "weekly_metrics_naive_at_enrollment.csv", index=False)
    fals_group_weekly_naive.to_csv(out_dir / "falsification_naive_group_weekly.csv", index=False)
    fals_diff_weekly_naive.to_csv(out_dir / "falsification_naive_old_young_diff_weekly.csv", index=False)
    fals_break_tests_naive.to_csv(out_dir / "falsification_naive_break_tests.csv", index=False)
    fals_diff_tests_naive.to_csv(out_dir / "falsification_naive_diff_break_tests.csv", index=False)
    fals_placebo_naive.to_csv(out_dir / "falsification_naive_placebo_scan.csv", index=False)
    fals_cov_naive.to_csv(out_dir / "falsification_naive_coverage_dilution.csv", index=False)
    fals_multi_naive.to_csv(out_dir / "falsification_naive_multi_split_summary.csv", index=False)
    fals_neg_naive = build_negative_control_rank_summary(fals_multi_naive)
    fals_neg_naive.to_csv(out_dir / "falsification_naive_negative_control_summary.csv", index=False)
    fals_decomp_naive = build_incidence_severity_decomposition_summary(fals_multi_naive)
    fals_decomp_naive.to_csv(out_dir / "falsification_naive_incidence_severity_summary.csv", index=False)
    fals_bounds_naive = build_ve_death_signal_bounds(
        wave_ve_summary,
        multi_split_summary=fals_multi_naive,
        split_definitions=split_definitions,
        compare_cohort=vax_cohort,
    )
    fals_bounds_naive.to_csv(out_dir / "falsification_naive_ve_death_bounds.csv", index=False)
    fals_quant_naive = build_quantitative_scenario_bounds(fals_bounds_naive, fals_multi_naive)
    fals_quant_naive.to_csv(out_dir / "falsification_naive_quantitative_scenario_bounds.csv", index=False)

    ve = float(exp_cfg.get("ve_death", 0.9))
    cfr_ref_mode = str(exp_cfg.get("cfr_reference", "same_week"))
    use_covid = bool(exp_cfg.get("use_covid_cfr", True))
    cfr_column = str(exp_cfg.get("cfr_column", "cfr_covid" if use_covid else "cfr_allcause"))
    if cfr_column not in weekly.columns:
        cfr_column = "cfr_covid" if "cfr_covid" in weekly.columns else "cfr_allcause"

    base_iso = {
        monday_to_iso_week(d)
        for d in iter_followup_mondays(str(baseline["start"]), str(baseline["end"]))
    }

    _log(
        f"compute_expected_vs_observed (VE_death={ve}, cfr_reference={cfr_ref_mode}, "
        f"{ref_cohort}→{vax_cohort}) …"
    )
    evo = compute_expected_vs_observed(
        weekly,
        ve_death=ve,
        cfr_reference=cfr_ref_mode,
        cfr_column=cfr_column,
        vaccinated_cohort=vax_cohort,
        reference_cohort=ref_cohort,
        age_bin="all",
        baseline_iso_weeks=base_iso if cfr_ref_mode == "baseline_mean" else None,
    )
    evo.to_csv(out_dir / "expected_vs_observed.csv", index=False)

    dpi = int(cfg.get("plots", {}).get("dpi", 120))
    ws, we = str(wave["start"]), str(wave["end"])

    _log("plotting main time series …")
    plot_case_rate(weekly, out_dir / "case_rate.png", age_bin="all", wave_start=ws, wave_end=we, dpi=dpi)
    cfr_plot_col = "cfr_covid" if use_covid else "cfr_allcause"
    if cfr_plot_col not in weekly.columns:
        cfr_plot_col = "cfr_covid"
    plot_cfr(
        weekly,
        out_dir / "cfr.png",
        age_bin="all",
        cfr_col=cfr_plot_col,
        wave_start=ws,
        wave_end=we,
        dpi=dpi,
    )
    plot_mortality_rate(weekly, out_dir / "mortality_rate.png", age_bin="all", wave_start=ws, wave_end=we, dpi=dpi)
    plot_cfg = cfg.get("plots") or {}
    covid_mort_cohorts = plot_cfg.get("covid_mortality_weekly_cohorts")
    if covid_mort_cohorts is None:
        covid_mort_cohorts = [c for c in cohorts if c in PRIMARY_ENROLLMENT_COHORTS]
    else:
        covid_mort_cohorts = [str(c) for c in covid_mort_cohorts]
    if plot_cfg.get("save_covid_mortality_per_million_weekly", True):
        _log("plot COVID deaths per million at risk (weekly, dose0–dose2) …")
        plot_covid_mortality_per_million(
            weekly,
            out_dir / "covid_deaths_per_million_weekly.png",
            age_bin="all",
            wave_start=ws,
            wave_end=we,
            cohorts=covid_mort_cohorts,
            dpi=dpi,
        )

    if cfg.get("plots", {}).get("save_wave_ve_plot", True):
        _log("plot wave_ve_summary (decomposition) …")
        plot_wave_ve_summary(
            wave_ve_summary,
            out_dir / "wave_ve_summary.png",
            compare_cohort=vax_cohort,
            reference_cohort=ref_cohort,
            dpi=dpi,
        )

    # Same cohorts as main TS plots (config order), not only ref/vax — e.g. dose1 on decomposition.
    present = set(weekly["cohort"].unique())
    decomp_cohorts = [c for c in cohorts if c in present]
    if len(decomp_cohorts) < 2:
        decomp_cohorts = sorted(present)
    _log(f"plot decomposition_compare cohorts={decomp_cohorts} …")
    plot_decomposition_compare(
        weekly,
        out_dir / "decomposition.png",
        cohorts=decomp_cohorts,
        age_bin="all",
        wave_start=ws,
        wave_end=we,
        dpi=dpi,
    )

    _log("plot expected_vs_observed (weekly + cumulative panel) …")
    plot_expected_vs_observed(evo, out_dir / "expected_vs_observed.png", ve=ve, wave_start=ws, wave_end=we, dpi=dpi)
    _log("plot expected_vs_observed_cumulative_only …")
    plot_expected_vs_observed_cumulative_only(
        evo, out_dir / "expected_vs_observed_cumulative.png", ve=ve, wave_start=ws, wave_end=we, dpi=dpi
    )

    plot_cumulative_cases_deaths(
        weekly, out_dir / "cumulative_cases_deaths.png", age_bin="all", wave_start=ws, wave_end=we, dpi=dpi
    )
    plot_old_young_difference(
        fals_diff_weekly,
        out_dir / "falsification_old_young_diff.png",
        break_iso_week=break_iso_week,
        dpi=dpi,
    )
    plot_old_young_difference(
        fals_diff_weekly_naive,
        out_dir / "falsification_naive_old_young_diff.png",
        break_iso_week=break_iso_week,
        dpi=dpi,
    )
    plot_placebo_break_scan(
        fals_placebo,
        out_dir / "falsification_placebo_scan.png",
        actual_break_iso_week=break_iso_week,
        dpi=dpi,
    )

    if cfg.get("plots", {}).get("save_age_stratified_ts"):
        _log("age-stratified plots (case rate, CFR, mortality) …")
        for ab in sorted(weekly["age_bin"].unique()):
            if ab == "all":
                continue
            plot_case_rate(
                weekly,
                out_dir / f"case_rate_age_{ab}.png",
                age_bin=ab,
                wave_start=ws,
                wave_end=we,
                dpi=dpi,
            )
            plot_cfr(
                weekly,
                out_dir / f"cfr_age_{ab}.png",
                age_bin=ab,
                cfr_col=cfr_plot_col,
                wave_start=ws,
                wave_end=we,
                dpi=dpi,
            )
            plot_mortality_rate(
                weekly,
                out_dir / f"mortality_rate_age_{ab}.png",
                age_bin=ab,
                wave_start=ws,
                wave_end=we,
                dpi=dpi,
            )
            if plot_cfg.get("save_covid_mortality_per_million_weekly", True):
                plot_covid_mortality_per_million(
                    weekly,
                    out_dir / f"covid_deaths_per_million_weekly_age_{ab}.png",
                    age_bin=ab,
                    wave_start=ws,
                    wave_end=we,
                    cohorts=covid_mort_cohorts,
                    dpi=dpi,
                )

    if plot_cfg.get("save_debug_weekly_deaths_pop", True):
        dbg_ab = str(plot_cfg.get("debug_weekly_denominator_age_bin", "70-120"))
        if dbg_ab in set(weekly["age_bin"].unique()):
            _log(f"debug plots: weekly COVID deaths + pop at risk (age_bin={dbg_ab}) …")
            dbg_slug = dbg_ab.replace("/", "-")
            plot_debug_weekly_covid_deaths(
                weekly,
                out_dir / f"debug_weekly_covid_deaths_age_{dbg_slug}.png",
                age_bin=dbg_ab,
                wave_start=ws,
                wave_end=we,
                cohorts=covid_mort_cohorts,
                dpi=dpi,
            )
            plot_debug_weekly_population_at_risk(
                weekly,
                out_dir / f"debug_weekly_population_at_risk_age_{dbg_slug}.png",
                age_bin=dbg_ab,
                wave_start=ws,
                wave_end=we,
                cohorts=covid_mort_cohorts,
                dpi=dpi,
            )
            hist_tbl, n_ppl, n_cv_dt = enrolled_covid_death_histogram_by_week(
                df_model,
                age_bin_label=dbg_ab,
                cohorts=covid_mort_cohorts,
                followup_start=followup_start,
                followup_end=followup_end,
            )
            hist_tbl.to_csv(
                out_dir / f"debug_enrolled_covid_deaths_weekly_total_age_{dbg_slug}.csv",
                index=False,
            )
            in_win = int(hist_tbl["deaths_covid_enrolled_total"].sum())
            _log(
                f"debug {dbg_ab}: enrolled persons (union {covid_mort_cohorts})={n_ppl:,}; "
                f"with parsed COVID-death week={n_cv_dt:,}; "
                f"COVID deaths bucketed into follow-up [{followup_start}–{followup_end}]={in_win:,}. "
                "These are analytic-cohort counts (same file / dose strata), not national 70+ totals."
            )
            plot_debug_weekly_covid_deaths_enrolled_total(
                weekly,
                out_dir / f"debug_weekly_covid_deaths_ENROLLED_TOTAL_age_{dbg_slug}.png",
                age_bin=dbg_ab,
                wave_start=ws,
                wave_end=we,
                cohorts=covid_mort_cohorts,
                dpi=dpi,
            )

    _log("quiet-period stability checks (reference + vaccinated cohorts, all + age bins) …")
    stab_cohorts = []
    for c in (ref_cohort, vax_cohort):
        if c in set(weekly["cohort"].unique()):
            stab_cohorts.append(c)
    for c in stab_cohorts:
        stability_check_quiet_period(
            weekly,
            baseline_start=str(baseline["start"]),
            baseline_end=str(baseline["end"]),
            cohort=c,
            age_bin="all",
        )
    for c in stab_cohorts:
        for ab in age_labels:
            stability_check_quiet_period(
                weekly,
                baseline_start=str(baseline["start"]),
                baseline_end=str(baseline["end"]),
                cohort=c,
                age_bin=ab,
            )

    ext = cfg.get("extensions", {})
    if ext.get("km_post_infection"):
        _log("Kaplan–Meier post-infection …")
        km_tbl, km_reason = build_km_post_infection_table(df_model, followup_end=followup_end, cohorts=cohorts)
        if len(km_tbl):
            km_tbl.to_csv(out_dir / "km_post_infection.csv", index=False)
            plot_km_post_infection(km_tbl, out_dir / "km_post_infection.png", age_bin="all", dpi=dpi)
            km_age_tbl, km_age_reason = build_km_post_infection_age_bin_table(
                df_model,
                followup_end=followup_end,
                cohorts=cohorts,
                age_bins=age_labels,
            )
            if len(km_age_tbl):
                km_age_tbl.to_csv(out_dir / "km_post_infection_by_age_bin.csv", index=False)
                for ab in age_labels:
                    plot_km_post_infection(
                        km_age_tbl,
                        out_dir / f"km_post_infection_age_{ab}.png",
                        age_bin=ab,
                        dpi=dpi,
                    )
            elif km_age_reason:
                _log(f"age-banded Kaplan-Meier skipped: {km_age_reason}")
            infected_comp = build_infected_cohort_age_composition_table(df_model, cohorts=cohorts)
            if len(infected_comp):
                infected_comp.to_csv(out_dir / "infected_cohort_age_composition.csv", index=False)
            _log("KM tables/plots and infected cohort composition written")
        else:
            _log(f"Kaplan–Meier skipped: {km_reason}")

    km_land_cfg = ext.get("km_landmark_survival")
    if isinstance(km_land_cfg, dict) and km_land_cfg.get("enabled"):
        lm_week = str(km_land_cfg.get("landmark_iso_week") or enrollment_week)
        lm_slug = lm_week.replace("-", "_")
        lm_ages = resolve_landmark_age_bins(km_land_cfg, age_labels)
        _bad = [x for x in lm_ages if x not in set(age_labels)]
        if _bad:
            raise ValueError(
                f"km_landmark_survival age_bins not in cfg age_bins labels: {_bad!r} "
                f"(allowed: {age_labels!r})"
            )
        _log(
            "Kaplan–Meier landmark (first-dose manufacturer + dose with next-dose censor) "
            f"for age_bin in [{', '.join(lm_ages)}] …"
        )

        mfg_frames: list[pd.DataFrame] = []
        dose_frames: list[pd.DataFrame] = []
        for lm_age in lm_ages:
            age_slug = lm_age.replace("-", "_")

            mfg_tbl, mfg_reason = build_km_landmark_first_mfg_table(
                df_model,
                landmark_iso_week=lm_week,
                followup_end=followup_end,
                age_bin=lm_age,
            )
            if len(mfg_tbl):
                mfg_frames.append(mfg_tbl)
                mfg_path = out_dir / f"km_landmark_{lm_slug}_first_mfg_age_{age_slug}.png"
                plot_km_survival_curves(
                    mfg_tbl,
                    mfg_path,
                    title=(
                        f"Landmark Kaplan–Meier (all-cause), first-dose manufacturer at {lm_week}\n"
                        f"age_bin={lm_age} (alive at landmark; censor at follow-up end)"
                    ),
                    xlabel="weeks since landmark (ISO week Monday)",
                    dpi=dpi,
                    cohort_order=FIRST_MFG_COHORTS,
                )
            elif mfg_reason:
                _log(f"landmark KM (first mfg, age_bin={lm_age}) skipped: {mfg_reason}")

            dose_tbl, dose_reason = build_km_landmark_dose_nextdose_censor_table(
                df_model,
                landmark_iso_week=lm_week,
                followup_end=followup_end,
                age_bin=lm_age,
            )
            if len(dose_tbl):
                dose_frames.append(dose_tbl)
                dose_path = out_dir / f"km_landmark_{lm_slug}_dose_censor_age_{age_slug}.png"
                plot_km_survival_curves(
                    dose_tbl,
                    dose_path,
                    title=(
                        f"Landmark Kaplan–Meier (all-cause), dose at {lm_week}\n"
                        f"age_bin={lm_age}; censor at next dose or follow-up end"
                    ),
                    xlabel="weeks since landmark (ISO week Monday)",
                    dpi=dpi,
                    cohort_order=DOSE_COHORTS,
                )
            elif dose_reason:
                _log(f"landmark KM (dose censor, age_bin={lm_age}) skipped: {dose_reason}")

        if mfg_frames:
            pd.concat(mfg_frames, ignore_index=True).to_csv(
                out_dir / f"km_landmark_{lm_slug}_first_mfg_by_age_bin.csv",
                index=False,
            )
        if dose_frames:
            pd.concat(dose_frames, ignore_index=True).to_csv(
                out_dir / f"km_landmark_{lm_slug}_dose_censor_by_age_bin.csv",
                index=False,
            )
        _log("landmark KM plots + by_age_bin CSVs done")

    if ext.get("time_since_dose2_plots"):
        _log("time-since-dose2 weekly slice …")
        bins = ext.get("time_since_dose2_weeks_bins", [[0, 4], [5, 12], [13, 999]])
        weekly_ts = run_time_since_dose2_analysis(
            df_model,
            bins=[list(b) for b in bins],
            followup_start=followup_start,
            followup_end=followup_end,
            age_bins_config=age_bins_config,
            baseline_start=str(baseline["start"]),
            baseline_end=str(baseline["end"]),
            wave_start=str(wave["start"]),
            wave_end=str(wave["end"]),
            reference_cohort=ref_cohort,
            metrics_workers=metrics_workers,
        )
        weekly_ts.to_csv(out_dir / "weekly_metrics_time_since_dose2.csv", index=False)
        plot_case_rate(
            weekly_ts,
            out_dir / "case_rate_time_since_dose2.png",
            age_bin="all",
            wave_start=ws,
            wave_end=we,
            dpi=dpi,
        )

    _log(f"done; outputs under {out_dir}")
    elapsed = time.monotonic() - t0
    _log(f"elapsed {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()

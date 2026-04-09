#!/usr/bin/env python3
"""
Czech CFR / infection / mortality pipeline entry point.

Expects the national CSV at data/Czech/records.csv (53 columns, ISO-week dates).
That file is not committed; use --smoke or data/Czech/records_100k.csv for quick tests.
Runtime is roughly linear in row count (on the order of minutes for full population on a laptop).
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
    build_km_post_infection_table,
    run_time_since_dose2_analysis,
    stability_check_quiet_period,
)
from cohort_builder import (  # noqa: E402
    PRIMARY_ENROLLMENT_COHORTS,
    build_enrollment_table,
    cohort_mask,
    iter_followup_mondays,
    monday_to_iso_week,
)
from load_data import load_czech_records, validate_loaded_schema  # noqa: E402
from metrics import (  # noqa: E402
    build_implied_ve_long_summary,
    build_weekly_metrics,
    parallel_stratum_pool_available,
)
from qa_summary import log_qa_summary  # noqa: E402
from plots import (  # noqa: E402
    plot_case_rate,
    plot_cfr,
    plot_cumulative_cases_deaths,
    plot_decomposition_compare,
    plot_expected_vs_observed,
    plot_expected_vs_observed_cumulative_only,
    plot_km_post_infection,
    plot_mortality_rate,
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

    _log("writing CSVs …")
    weekly.to_csv(out_dir / "weekly_metrics.csv", index=False)
    cohort_summary.to_csv(out_dir / "cohort_summary.csv", index=False)
    implied_ve_long.to_csv(out_dir / "implied_ve_summary.csv", index=False)

    age_only = weekly[weekly["age_bin"] != "all"].copy()
    age_only.to_csv(out_dir / "age_stratified_metrics.csv", index=False)

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
            plot_km_post_infection(km_tbl, out_dir / "km_post_infection.png", dpi=dpi)
            _log("KM table and plot written")
        else:
            _log(f"Kaplan–Meier skipped: {km_reason}")

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

#!/usr/bin/env python3
"""
Explain why CFR ``covid_deaths_per_million_weekly`` plots differ from ``audit_covid_deaths_iso_week.py``.

The per-million curves use ``build_weekly_metrics`` numerators/denominators: **only people alive at
enrollment** in **dose0 | dose1 | dose2**, stratified by **age_bin at enrollment**. The audit script
counts **everyone in the CSV** with Umrti week in range (no enrollment gate).

This script loads the same CSV + YAML as the CFR pipeline (optional mRNA filter off), runs
enrollment, then for one ISO week and age_bin prints:

  - file-wide unique IDs (age_bin slice) with COVID death that week
  - enrolled-cohort unique IDs (dose0∪dose1∪dose2) with COVID death that week
  - per stratum ``deaths_covid`` and ``population_at_risk`` from the same logic as weekly_metrics
  - implied per-million = deaths_covid / population_at_risk * 1e6

Example:
  python test/CFR/diagnose_covid_per_million_gap.py --iso-week 2021-48 --age-bin 70-120 \\
      --csv data/Czech/records.csv --config test/CFR/config/czech.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CFR_ROOT = Path(__file__).resolve().parent
CODE_DIR = CFR_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from cohort_builder import (  # noqa: E402
    build_enrollment_table,
    cohort_mask,
    iter_followup_mondays,
    monday_to_iso_week,
)
from load_data import load_czech_records  # noqa: E402
from metrics import (  # noqa: E402
    _compute_weekly_stratum_rows,
    _hist_event,
    _sub_one_row_per_person,
    _week_index_map,
    enrolled_covid_death_histogram_by_week,
)


def _resolve(p: str, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare audit-style COVID death counts vs CFR weekly strata.")
    ap.add_argument("--iso-week", type=str, required=True)
    ap.add_argument("--age-bin", type=str, default="70-120")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=CFR_ROOT / "config" / "czech.yaml")
    ap.add_argument("--no-mrna-filter", action="store_true", help="Keep all manufacturers (default for fair audit match)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write report UTF-8 text here",
    )
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    csv_path = args.csv or _resolve(cfg["dataset"]["path"], REPO_ROOT)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    cohort_cfg = cfg["cohort"]
    enrollment_week = str(cohort_cfg["enrollment_week"])
    followup_start = str(cohort_cfg["followup_start"])
    followup_end = str(cohort_cfg["followup_end"])
    cohorts = [c for c in cfg["cohorts"] if c in ("dose0", "dose1", "dose2")]
    age_bins_config = [list(x) for x in cfg["age_bins"]]

    restrict_pm = not args.no_mrna_filter
    repo_code = REPO_ROOT / "code"
    _log_lines: list[str] = []

    def ln(s: str = "") -> None:
        print(s)
        _log_lines.append(s)

    ln(f"Generated (UTC): {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    ln(f"csv: {csv_path}")
    ln(f"config: {cfg_path}")
    ln(f"restrict_to_pfizer_moderna: {restrict_pm}")
    ln(f"target ISO week: {args.iso_week}")
    ln(f"age_bin (enrollment): {args.age_bin}")
    ln("")

    ln("Loading CSV …")
    df = load_czech_records(
        csv_path,
        restrict_to_pfizer_moderna=restrict_pm,
        single_infection_only=False,
        repo_code_dir=repo_code if restrict_pm else None,
    )
    if not restrict_pm and df.shape[1] < 50:
        raise SystemExit("Expected Czech NR columns after load")

    ln(f"rows loaded: {len(df):,}")
    ln("build_enrollment_table …")
    df = build_enrollment_table(
        df,
        enrollment_week=enrollment_week,
        age_bins=age_bins_config,
        progress_log=None,
    )
    df_model = df[df["age_bin"].notna()].copy()
    ln(f"rows with age_bin: {len(df_model):,}")
    ln("")

    weeks = iter_followup_mondays(followup_start, followup_end)
    wmap = _week_index_map(weeks)
    iso_labels = [monday_to_iso_week(w) for w in weeks]
    target = str(args.iso_week).strip()
    if target not in iso_labels:
        raise SystemExit(f"ISO week {target} not in follow-up {followup_start}–{followup_end}")
    wi = iso_labels.index(target)

    ab = args.age_bin

    # --- File-wide: everyone with this age_bin (still in analytic file), COVID death in target week
    sub_file = df_model[df_model["age_bin"] == ab].copy()
    subp_file = _sub_one_row_per_person(sub_file)
    h_file = _hist_event(subp_file, "covid_death_monday", weeks, wmap)
    ln(f"FILE-WIDE (all rows, age_bin={ab}): unique people ≈ histogram sum = {int(h_file.sum()):,}")
    ln(f"  COVID deaths in ISO week {target} only: {int(h_file[wi]):,}")
    ln("")

    # --- Enrolled dose0|dose1|dose2 only (same as weekly_metrics strata)
    cm = cohort_mask(df_model, cohorts[0])
    for c in cohorts[1:]:
        cm = cm | cohort_mask(df_model, c)
    sub_enr = df_model.loc[cm & (df_model["age_bin"] == ab)].copy()
    subp_enr = _sub_one_row_per_person(sub_enr)
    h_enr = _hist_event(subp_enr, "covid_death_monday", weeks, wmap)
    ln(f"ENROLLED (dose0∪dose1∪dose2 at enrollment, age_bin={ab}):")
    ln(f"  unique people in stratum (one row per ID): {len(subp_enr):,}")
    ln(f"  COVID deaths in ISO week {target}: {int(h_enr[wi]):,}")
    ln("")

    ln("WEEKLY_METRICS-STYLE (must match plot numerators/denominators for that week):")
    total_d = 0
    total_p = 0
    for cohort in cohorts:
        rows = _compute_weekly_stratum_rows(
            df_model,
            cohort,
            ab,
            weeks=weeks,
            wmap=wmap,
            iso_labels=iso_labels,
            cohort_masks=None,
        )
        by_iso = {r["iso_week"]: r for r in rows}
        r = by_iso[target]
        d = int(r["deaths_covid"])
        p = int(r["population_at_risk"])
        total_d += d
        total_p += p
        per_m = (d / p * 1.0e6) if p > 0 else float("nan")
        ln(
            f"  {cohort}: deaths_covid={d:,}  population_at_risk={p:,}  "
            f"deaths/million={per_m:.4f}"
        )
    ln(f"  SUM cohorts: deaths_covid={total_d:,}  population_at_risk={total_p:,}")
    if total_p > 0:
        ln(f"  pooled deaths/million (sum d / sum p, descriptive only): {total_d / total_p * 1e6:.4f}")
    ln("")

    hist_tbl, n_ppl, n_cv = enrolled_covid_death_histogram_by_week(
        df_model,
        age_bin_label=ab,
        cohorts=cohorts,
        followup_start=followup_start,
        followup_end=followup_end,
    )
    row = hist_tbl.loc[hist_tbl["iso_week"] == target].iloc[0]
    ln(f"enrolled_covid_death_histogram_by_week (cross-check): week {target} total = {int(row['deaths_covid_enrolled_total']):,}")
    ln(f"  (n_people union strata={n_ppl:,}, with any parsed COVID-death week={n_cv:,})")
    ln("")

    ln("WHY this differs from audit_covid_deaths_iso_week.py (703 for 2021-48, age≥70):")
    ln("  - Audit uses age ≥ 70 via 2021 − YearOfBirth (not necessarily same set as enrollment age_bin 70-120).")
    ln("  - Audit includes people NOT in dose0|dose1|dose2 at enrollment (e.g. not alive at enrollment week).")
    ln("  - Per-million plots further SPLIT enrolled people by dose — spikes use tiny denominators per cohort.")
    ln("")

    if int(h_enr[wi]) != total_d:
        ln(f"WARNING: enrolled histogram week ({int(h_enr[wi])}) != sum weekly rows ({total_d}) — investigate.")
    if int(row["deaths_covid_enrolled_total"]) != total_d:
        ln(f"WARNING: histogram_by_week ({int(row['deaths_covid_enrolled_total'])}) != sum strata ({total_d}).")

    report = "\n".join(_log_lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

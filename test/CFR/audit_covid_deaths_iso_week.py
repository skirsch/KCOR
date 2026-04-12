#!/usr/bin/env python3
"""
Count unique people aged ≥min_age with a COVID death (Umrti / Date_COVID_death) in a given ISO week.

Uses the same 53-column Czech NR layout as the CFR pipeline: Czech column 8 ``Umrti`` maps to
``Date_COVID_death`` (ISO week strings like ``2021-48``). **No enrollment restriction**, **no
mRNA manufacturer filter** — this is the direct file audit the CFR weekly plots do *not* show.

Why CFR debug plots can be much smaller:
  - They count only people **alive at enrollment** in **dose0|dose1|dose2** strata, then bucket by
    ``covid_death_monday`` into follow-up weeks. National-style totals require summing everyone in
    the extract with a COVID-death week, as this script does.

Awk note: ``deaths.csv`` is quoted CSV; naive ``awk -F, '$8==...'`` breaks on commas inside fields.
Use this script (or ``csvkit``) instead.

Examples (from repo root, WSL or Python):
  python test/CFR/audit_covid_deaths_iso_week.py --csv data/deaths.csv --iso-week 2021-48
  python test/CFR/audit_covid_deaths_iso_week.py --csv data/Czech/records.csv --iso-week 2021-48 --min-age 70

Optional refinement (Pfizer/Moderna-only, same rule as CFR YAML):
  python test/CFR/audit_covid_deaths_iso_week.py --csv data/deaths.csv --iso-week 2021-48 --mrna-only \\
      --repo-code ../../code
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CFR_CODE = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CFR_CODE))

from cohort_builder import monday_to_iso_week  # noqa: E402
from load_data import (  # noqa: E402
    CZECH_TO_ENGLISH_COLUMNS,
    EXPECTED_COL_COUNT,
    read_csv_flex,
)


def _covid_death_iso_week_series(date_covid_death: pd.Series) -> pd.Series:
    """Match cohort_builder Date_COVID_death parsing → ISO week label ``YYYY-WW``."""
    x = date_covid_death.fillna("").astype(str).str.strip()
    x = x.replace({"nan": "", "NaT": "", "<NA>": "", "None": ""})
    bad = x.str.match(r"^\d$", na=False)
    ts = pd.to_datetime(x + "-1", format="%G-%V-%u", errors="coerce")
    ts = ts.mask(bad, pd.NaT)

    def _label(t) -> object:
        if pd.isna(t):
            return pd.NA
        return monday_to_iso_week(t.date())

    return ts.map(_label)


def _birth_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit COVID (Umrti) deaths by ISO week and age (raw NR file).")
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to 53-col Czech CSV (default: data/deaths.csv then data/Czech/records.csv under repo root)",
    )
    ap.add_argument("--iso-week", type=str, required=True, help="Target ISO week label, e.g. 2021-48")
    ap.add_argument("--min-age", type=int, default=70, help="Minimum completed age (default 70)")
    ap.add_argument(
        "--age-reference-year",
        type=int,
        default=2021,
        help="Completed age = reference_year - birth_year (default 2021 for late-2021 waves)",
    )
    ap.add_argument(
        "--mrna-only",
        action="store_true",
        help="Drop rows with non–Pfizer/Moderna codes on doses 1–4 (requires --repo-code)",
    )
    ap.add_argument(
        "--repo-code",
        type=Path,
        default=None,
        help="Repo ``code/`` directory (required for --mrna-only); default: <repo>/code",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write this report (UTF-8) to the given path",
    )
    ap.add_argument(
        "--save-reference",
        action="store_true",
        help="Write report next to the CSV: audit_covid_deaths_iso_week_<ISO>_reference.txt",
    )
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        for candidate in (REPO_ROOT / "data" / "deaths.csv", REPO_ROOT / "data" / "Czech" / "records.csv"):
            if candidate.is_file():
                csv_path = candidate
                break
        if csv_path is None:
            raise SystemExit(
                "No --csv given and neither data/deaths.csv nor data/Czech/records.csv exists. "
                "Pass --csv explicitly."
            )

    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"File not found: {csv_path}")

    df = read_csv_flex(csv_path)
    if df.shape[1] != EXPECTED_COL_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_COL_COUNT} columns (Czech NR export), got {df.shape[1]} in {csv_path}"
        )
    df = df.copy()
    df.columns = CZECH_TO_ENGLISH_COLUMNS

    if args.mrna_only:
        repo_code = args.repo_code or (REPO_ROOT / "code")
        from load_data import load_czech_records  # noqa: E402

        df = load_czech_records(
            csv_path,
            restrict_to_pfizer_moderna=True,
            single_infection_only=False,
            repo_code_dir=repo_code.resolve(),
        )

    iso = _covid_death_iso_week_series(df["Date_COVID_death"])
    target = str(args.iso_week).strip()
    in_week = iso == target

    birth = _birth_year(df["YearOfBirth"])
    age_completed = args.age_reference_year - birth
    age_ok = age_completed >= args.min_age

    m = in_week & age_ok
    n_hit_rows = int(m.sum())
    n_unique_id = int(df.loc[m, "ID"].nunique()) if n_hit_rows else 0

    # People with COVID death in target week but unknown birth year
    m_week_unknown_birth = in_week & birth.isna()
    n_week_unknown_birth = int(df.loc[m_week_unknown_birth, "ID"].nunique())

    m_all = in_week
    n_all_id = int(df.loc[m_all, "ID"].nunique())

    lines = [
        f"Generated (UTC): {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        f"file: {csv_path}",
        f"rows in file: {len(df):,}",
        f"mrna_only filter: {args.mrna_only}",
        f"target ISO week (Umrti→Date_COVID_death): {target}",
        f"age: completed age >= {args.min_age} using reference year {args.age_reference_year} − YearOfBirth",
        "---",
        f"rows with COVID-death week == {target} and age_ok: {n_hit_rows:,}",
        f"unique ID in that set: {n_unique_id:,}",
    ]
    if n_week_unknown_birth:
        lines.append(
            f"note: {n_week_unknown_birth:,} unique ID have COVID week {target} but missing/invalid birth year "
            "(excluded from age filter above)"
        )
    lines.extend(
        [
            "---",
            f"unique ID with COVID week == {target} (all ages, same file): {n_all_id:,}",
        ]
    )
    report = "\n".join(lines) + "\n"
    print(report, end="")

    if args.output is not None:
        out_path = args.output
    elif args.save_reference:
        slug = target.replace("/", "-")
        out_path = csv_path.parent / f"audit_covid_deaths_iso_week_{slug}_reference.txt"
    else:
        out_path = None
    if out_path is not None:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

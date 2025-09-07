#!/usr/bin/env python3
"""
Kaplan–Meier survival analysis for KCOR_CMR.xlsx

- Reads `data/Czech/KCOR_CMR.xlsx`
- Filters to a single enrollment sheet (e.g., "2021_24")
- Filters YearOfBirth within an inclusive range, e.g., (1940, 2000)
- Aggregates across sexes
- Groups doses as specified, e.g., [0] vs [1,2]
- Computes weekly Kaplan–Meier survival S(t)
  - Equalizes initial population by setting both groups' N0 to the vaccinated
    group's population at enrollment, to produce comparable curves when
    cohorts are death-matched but have different population sizes.
  - Uses weekly discrete hazard: h_t = deaths_t / risk_t
    S_t = S_{t-1} * (1 - h_t)
  - risk_t is survivors entering the week; with discrete weekly steps we
    approximate risk_t as current survivors.
  - Deaths and Alive are taken from sheet columns and aggregated over sexes.

Outputs:
- PNG plot under `validation/kaplan_meier/out/` with both survival curves.

Usage:
    python validation/kaplan_meier/KM.py --sheet 2021_24 --birth-years 1940 2000 \
        --groups "0" "1,2" --out validation/kaplan_meier/out/KM_2021_24_1940_2000.png

Dependencies:
    pip install pandas numpy matplotlib openpyxl
"""
import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


DATA_PATH = Path("data/Czech/KCOR_CMR.xlsx")


def parse_groups(group_args: List[str]) -> List[Tuple[int, ...]]:
    """Parse dose group strings like ["0", "1,2"] -> [(0,), (1,2)]."""
    groups: List[Tuple[int, ...]] = []
    for g in group_args:
        parts = [p.strip() for p in g.split(",") if p.strip() != ""]
        tup = tuple(sorted(set(int(p) for p in parts)))
        if len(tup) == 0:
            raise ValueError(f"Invalid group spec: '{g}'")
        groups.append(tup)
    return groups


def parse_plot_range(plot_spec: str) -> List[Tuple[int, ...]]:
    """Parse plot range 'start,end[,step]' into individual dose groups.

    Returns list of singletons: [(d,), (d+step,), ...] covering the inclusive range.
    If step omitted, defaults to 1.
    """
    if not plot_spec:
        return [(0,), (1, 2)]
    parts = [p.strip() for p in plot_spec.split(',') if p.strip() != '']
    if len(parts) < 2:
        raise ValueError("--plot requires at least 'start,end' (e.g., 0,2 or 0,2,1)")
    start = int(parts[0])
    end = int(parts[1])
    step = int(parts[2]) if len(parts) >= 3 else 1
    if step == 0:
        raise ValueError("step must be non-zero")
    if (end - start) * step < 0:
        raise ValueError("step sign inconsistent with start/end")
    # Build inclusive range
    seq = list(range(start, end + (1 if step > 0 else -1), step))
    if not seq:
        raise ValueError("empty plot range")
    return [(d,) for d in seq]


def load_sheet(sheet: str, data_path: Path) -> pd.DataFrame:
    """Load a single enrollment sheet from the workbook.

    Expects columns: ISOweekDied, DateDied, YearOfBirth, Sex, Dose, Alive, Dead
    """
    xls = pd.ExcelFile(data_path)
    if sheet not in xls.sheet_names:
        raise ValueError(f"Sheet '{sheet}' not found in {data_path}")
    df = pd.read_excel(data_path, sheet_name=sheet)
    # Normalize types
    df["DateDied"] = pd.to_datetime(df["DateDied"])  # may contain date-like strings
    # Drop absurd birth year outliers, keep -1 as unknown
    df = df[df["YearOfBirth"] <= 2020]
    return df


def filter_enrollment_start(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Filter to rows on/after the enrollment start date derived from sheet name YYYY_WW.
    Mirrors logic in KCOR.py.
    """
    if "_" not in sheet:
        return df
    year_str, week_str = sheet.split("_")
    enrollment_year = int(year_str)
    enrollment_week = int(week_str)
    jan1 = datetime(enrollment_year, 1, 1)
    days_to_monday = (7 - jan1.weekday()) % 7
    if days_to_monday == 0 and jan1.weekday() != 0:
        days_to_monday = 7
    first_monday = jan1 + timedelta(days=days_to_monday)
    enrollment_date = first_monday + timedelta(weeks=enrollment_week - 1)
    return df[df["DateDied"] >= enrollment_date]


def get_enrollment_date(sheet: str) -> pd.Timestamp:
    """Compute the enrollment start date (Monday of the ISO week encoded in sheet)."""
    if "_" not in sheet:
        raise ValueError(f"Cannot derive enrollment date from sheet name: {sheet}")
    year_str, week_str = sheet.split("_")
    enrollment_year = int(year_str)
    enrollment_week = int(week_str)
    jan1 = datetime(enrollment_year, 1, 1)
    days_to_monday = (7 - jan1.weekday()) % 7
    if days_to_monday == 0 and jan1.weekday() != 0:
        days_to_monday = 7
    first_monday = jan1 + timedelta(days=days_to_monday)
    return pd.to_datetime(first_monday + timedelta(weeks=enrollment_week - 1))


def filter_birth_years(df: pd.DataFrame, birth_year_range: Tuple[int, int]) -> pd.DataFrame:
    a, b = birth_year_range
    return df[(df["YearOfBirth"] >= a) & (df["YearOfBirth"] <= b)]


def aggregate_across_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across sexes for each (YearOfBirth, Dose, DateDied)."""
    grouped = (
        df.groupby(["YearOfBirth", "Dose", "DateDied"])  # order matters for cumulative ops later
        .agg({
            "ISOweekDied": "first",
            "Alive": "sum",
            "Dead": "sum",
        })
        .reset_index()
        .sort_values(["YearOfBirth", "Dose", "DateDied"])
        .reset_index(drop=True)
    )
    return grouped


def build_groups(df: pd.DataFrame, groups: List[Tuple[int, ...]]) -> Dict[Tuple[int, ...], pd.DataFrame]:
    """Sum Alive/Dead over specified dose groups per (YearOfBirth, DateDied)."""
    out: Dict[Tuple[int, ...], pd.DataFrame] = {}
    for grp in groups:
        gdf = df[df["Dose"].isin(grp)]
        if gdf.empty:
            # Keep empty to fail fast later with a clear message
            out[grp] = gdf
            continue
        summed = (
            gdf.groupby(["YearOfBirth", "DateDied"]).agg({
                "ISOweekDied": "first",
                "Alive": "sum",
                "Dead": "sum",
            }).reset_index().sort_values(["YearOfBirth", "DateDied"]).reset_index(drop=True)
        )
        out[grp] = summed
    return out


def to_week_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer week index t starting from 0 per (YearOfBirth)."""
    df = df.sort_values(["YearOfBirth", "DateDied"]).copy()
    df["t"] = df.groupby(["YearOfBirth"]).cumcount().astype(int)
    return df


def compute_km_by_age(summed: pd.DataFrame) -> pd.DataFrame:
    """Compute Kaplan–Meier survival per YearOfBirth with weekly steps.

    Assumes the first row per age is enrollment week. Uses standard KM product-limit:
      S_t = Π_{k<=t} (1 - d_k / n_k)
    where d_k = deaths in week k, n_k = at-risk entering week k (approximated by current survivors).
    """
    out_rows: List[Dict] = []
    for yob, g in summed.groupby("YearOfBirth", sort=False):
        g = g.sort_values("DateDied").reset_index(drop=True)
        # Enrollment population for this age = Alive in first week
        if g.empty:
            continue
        n0 = float(g.loc[0, "Alive"]) if "Alive" in g.columns else np.nan
        if not np.isfinite(n0) or n0 <= 0:
            # If missing, approximate from first nonzero alive
            nz = g[g["Alive"] > 0]
            n0 = float(nz["Alive"].iloc[0]) if not nz.empty else np.nan
        S = 1.0
        n_at_risk = n0
        for i, row in g.iterrows():
            d = float(row.get("Dead", 0.0))
            if n_at_risk and n_at_risk > 0:
                h = max(0.0, min(1.0, d / n_at_risk))
            else:
                h = 0.0
            S = S * (1.0 - h)
            n_at_risk = max(0.0, n_at_risk - d)
            out_rows.append({
                "YearOfBirth": yob,
                "DateDied": row["DateDied"],
                "t": i,
                "Alive0": n0,
                "Deaths": d,
                "S": S,
            })
    return pd.DataFrame(out_rows)


def equalize_population_ref_to(group_km: pd.DataFrame, ref_km: pd.DataFrame) -> pd.DataFrame:
    """Rescale survival so that the initial population equals the reference N0.

    Given S(t) built from per-age N0, we form survivors(t) = N0_age * S_age(t) and then
    sum across ages; to equalize, we set total N0 across ages equal to the reference
    group's total N0_ref, and rescale the survivors series proportionally.
    """
    # Total initial population across ages
    def total_N0(df: pd.DataFrame) -> float:
        return (
            df.sort_values(["YearOfBirth", "t"])  # ensure t=0 exists per age
            .groupby("YearOfBirth").first()["Alive0"].fillna(0.0).sum()
        )

    N0_ref = total_N0(ref_km)
    N0_grp = total_N0(group_km)
    scale = (N0_ref / N0_grp) if (N0_grp and N0_grp > 0) else 1.0

    # Convert to cohort-level survivors by summing per-age survivors
    def to_survivors(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["survivors_age"] = df["Alive0"].fillna(0.0) * df["S"].fillna(1.0)
        # Align by calendar date (weekly bins)
        cohort = (
            df.groupby("DateDied")["survivors_age"].sum().reset_index().sort_values("DateDied")
        )
        cohort["t"] = np.arange(len(cohort), dtype=int)
        return cohort

    grp_surv = to_survivors(group_km)
    grp_surv["survivors"] = grp_surv["survivors_age"] * scale
    grp_surv = grp_surv.drop(columns=["survivors_age"])  # keep clean
    return grp_surv


def plot_survival(curves: Dict[Tuple[int, ...], pd.DataFrame], title: str, out_path: Path) -> Path:
    plt.figure(figsize=(10, 6))
    for grp, df in curves.items():
        if df.empty:
            continue
        # Convert to survival fraction relative to equalized N0 (first value)
        if "survivors" in df.columns:
            n0 = float(df["survivors"].iloc[0]) if len(df) else 1.0
            y = df["survivors"].astype(float) / (n0 if n0 > 0 else 1.0)
        else:
            # If not equalized, assume S is available
            y = df["S"].astype(float)
        label = "+".join(str(d) for d in grp)
        plt.plot(df["t"], y, label=f"Dose {label}")
    plt.xlabel("Weeks since enrollment")
    plt.ylabel("Survival S(t)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    abs_out = out_path.resolve()
    # Write to a temporary file then replace atomically to ensure overwrite
    from tempfile import NamedTemporaryFile
    tmp_dir = abs_out.parent
    ts_tmp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    tmp_path = tmp_dir / f".{abs_out.stem}_{ts_tmp}{abs_out.suffix}"
    try:
        plt.savefig(tmp_path, dpi=150)
        try:
            import os
            os.replace(str(tmp_path), str(abs_out))
            final_path = abs_out
        except Exception as e:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fallback = abs_out.with_name(f"{abs_out.stem}_{ts}{abs_out.suffix}")
            print(f"[WARN] Could not replace {abs_out} ({e}); writing to {fallback}")
            os.replace(str(tmp_path), str(fallback))
            final_path = fallback
    except Exception as e:
        print(f"[ERROR] Failed to save figure: {e}")
        final_path = abs_out
    finally:
        plt.close()
    return final_path


def main():
    p = argparse.ArgumentParser(description="Kaplan–Meier survival from KCOR_CMR.xlsx")
    p.add_argument("--sheet", required=True, help="Enrollment sheet name, e.g., 2021_24")
    p.add_argument("--birth-years", nargs=2, type=int, metavar=("START", "END"), required=True,
                   help="Inclusive birth year range, e.g., 1940 2000")
    p.add_argument("--groups", nargs=2, required=False,
                   help="Two dose groups, e.g., '0' '1,2' means [0] vs [1,2] (deprecated; use --plot)")
    p.add_argument("--plot", required=False,
                   help="Dose range 'start,end[,step]'; builds groups: start vs combine(start+step..end). Default 0,2,1")
    p.add_argument("--out", default=None, help="Output PNG path")
    p.add_argument("--input", default=None, help="Path to KCOR_CMR.xlsx (defaults to repo-root data path)")
    p.add_argument("--equal-population", type=int, choices=[0,1], default=1,
                   help="1 = equalize initial populations between cohorts (default), 0 = use source populations")
    args = p.parse_args()

    sheet = args.sheet
    birth_years = (args.birth_years[0], args.birth_years[1])
    if args.groups:
        group_specs = parse_groups(args.groups)
    else:
        group_specs = parse_plot_range(args.plot or "0,2,1")

    # Resolve input path; default to repo-root data path
    if args.input:
        data_path = Path(args.input)
    else:
        # Repo root assumed to be three levels up from this file: validation/kaplan_meier/code/KM.py
        repo_root = Path(__file__).resolve().parents[3]
        data_path = repo_root / "data/Czech/KCOR_CMR.xlsx"

    df = load_sheet(sheet, data_path)
    df = filter_enrollment_start(df, sheet)
    df = filter_birth_years(df, birth_years)
    df = aggregate_across_sex(df)

    grouped = build_groups(df, group_specs)
    
    # Print deaths on enrollment date and on 2024-04-01 for each cohort (summed across ages)
    enrollment_dt = get_enrollment_date(sheet)
    final_dt = pd.to_datetime("2024-04-01")
    print("Week            Cohort #   Deaths   CumDeaths")
    for label_grp in group_specs:
        label = "+".join(str(d) for d in label_grp)
        gdf = grouped[label_grp]
        # Ensure DateDied dtype is datetime64[ns]
        gdf["DateDied"] = pd.to_datetime(gdf["DateDied"]) 
        cohort_by_date = gdf.groupby("DateDied")["Dead"].sum().sort_index()
        cohort_cum = cohort_by_date.cumsum()
        d_enroll = int(cohort_by_date.loc[cohort_by_date.index == enrollment_dt].sum())
        d_final  = int(cohort_by_date.loc[cohort_by_date.index == final_dt].sum())
        c_enroll = int(cohort_cum.loc[cohort_cum.index == enrollment_dt].sum())
        c_final  = int(cohort_cum.loc[cohort_cum.index == final_dt].sum())
        print(f"{enrollment_dt.date()}    {label:>7}   {d_enroll:>6}   {c_enroll:>9}")
        print(f"{final_dt.date()}    {label:>7}   {d_final:>6}   {c_final:>9}")
    # Compute KM per group, per age
    km_by_group: Dict[Tuple[int, ...], pd.DataFrame] = {}
    for grp, gdf in grouped.items():
        if gdf.empty:
            raise ValueError(f"No data available for group {grp} in sheet {sheet} and birth years {birth_years}")
        gdf = to_week_index(gdf)
        km_by_group[grp] = compute_km_by_age(gdf)

    # Build curves per parameter
    curves: Dict[Tuple[int, ...], pd.DataFrame] = {}
    if args.equal_population == 1 and len(group_specs) > 0:
        # Choose reference group: prefer highest nonzero dose in selection; else first
        nonzero_groups = [g for g in group_specs if any(d > 0 for d in g)]
        if nonzero_groups:
            ref_group = max(nonzero_groups, key=lambda g: max(g))
        else:
            ref_group = group_specs[0]
        ref_km = km_by_group[ref_group]
        # Equalize each group's initial population to the reference group's N0
        for g in group_specs:
            if g == ref_group:
                curves[g] = equalize_population_ref_to(ref_km, ref_km)
            else:
                curves[g] = equalize_population_ref_to(km_by_group[g], ref_km)
    else:
        # Use source populations (no equalization)
        def to_cohort_survivors(df_km: pd.DataFrame) -> pd.DataFrame:
            df_km = df_km.copy()
            df_km["survivors_age"] = df_km["Alive0"].fillna(0.0) * df_km["S"].fillna(1.0)
            cohort = df_km.groupby("DateDied")["survivors_age"].sum().reset_index().sort_values("DateDied")
            cohort["t"] = np.arange(len(cohort), dtype=int)
            cohort.rename(columns={"survivors_age": "survivors"}, inplace=True)
            return cohort
        for g in group_specs:
            curves[g] = to_cohort_survivors(km_by_group[g])

    title = f"Kaplan–Meier: {sheet}, YoB {birth_years[0]}-{birth_years[1]}"
    out_path = Path(args.out) if args.out else Path(f"validation/kaplan_meier/out/KM_{sheet}_{birth_years[0]}_{birth_years[1]}.png")
    abs_out = plot_survival(curves, title, out_path)
    # Verify file exists and report absolute path
    if abs_out.exists():
        print(f"[Done] Wrote plot to {abs_out}")
    else:
        print(f"[WARN] Save reported success but file not found at {abs_out}")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
ICD-10 Population Structural Shift Analysis

Analyze population-level structural shifts in ICD-10 cause of death patterns over time,
comparing pre-vaccine (2010-2019) vs post-vaccine (2021-2023) eras to detect structural breaks
and distinguish selection bias from true biological shifts.

USAGE:
    python icd_population_shift.py [input_csv] [output_dir]
    
    Defaults:
    - input_csv: ../data/Czech2/data.csv
    - output_dir: ../data/Czech2/

OUTPUTS:
    - icd_population_shift.csv: ICD-level pre vs post comparison (all ages)
    - icd_system_yearly.csv: Organ system distributions by year
    - icd_system_shift.csv: Organ system pre vs post shifts (all ages)
    - icd_population_shift_age65_89.csv: ICD-level comparison (age 65-89)
    - icd_system_shift_age65_89.csv: Organ system shifts (age 65-89)

DEPENDENCIES:
    pip install pandas
"""

import sys
import pandas as pd
from pathlib import Path


def compute_icd_shifts(death_df, output_path, suffix=""):
    """
    Compute ICD distribution shifts between pre-vaccine and post-vaccine eras.
    
    Args:
        death_df: DataFrame with death records containing 'year' and 'icd' columns
        output_path: Path object for output directory
        suffix: Optional suffix for output filename (e.g., "_age65_89")
    """
    print(f"\n[ICD Shifts{suffix}] Computing ICD distribution shifts...")
    
    # Compute ICD distributions per year
    year_icd_counts = (
        death_df.groupby(["year", "icd"])
        .size()
        .rename("count")
        .reset_index()
    )
    
    # Convert raw counts to % of deaths for each ICD within each year
    year_totals = year_icd_counts.groupby("year")["count"].transform("sum")
    year_icd_counts["pct"] = year_icd_counts["count"] / year_totals
    
    print(f"  Total ICD-year combinations: {len(year_icd_counts):,}")
    
    # Define eras
    pre = year_icd_counts[year_icd_counts["year"].between(2010, 2019)]
    covid_pre_vax = year_icd_counts[year_icd_counts["year"] == 2020]
    post = year_icd_counts[year_icd_counts["year"].between(2021, 2023)]
    
    print(f"  Pre-vaccine era (2010-2019): {len(pre):,} records")
    print(f"  COVID pre-vax (2020): {len(covid_pre_vax):,} records")
    print(f"  Post-vaccine era (2021-2023): {len(post):,} records")
    
    # Compute average ICD shares before vs after rollout
    pre_mean = pre.groupby("icd")["pct"].mean()
    post_mean = post.groupby("icd")["pct"].mean()
    
    shift = pd.DataFrame({
        "pre_pct": pre_mean,
        "post_pct": post_mean
    }).fillna(0)
    
    shift["diff"] = shift["post_pct"] - shift["pre_pct"]
    shift.sort_values("diff", ascending=False, inplace=True)
    
    output_file = output_path / f"icd_population_shift{suffix}.csv"
    shift.to_csv(output_file)
    print(f"  Saved: {output_file}")
    
    return shift


def compute_system_shifts(death_df, output_path, suffix=""):
    """
    Compute organ system distribution shifts between pre-vaccine and post-vaccine eras.
    
    Args:
        death_df: DataFrame with death records containing 'year' and 'system' columns
        output_path: Path object for output directory
        suffix: Optional suffix for output filename (e.g., "_age65_89")
    """
    print(f"\n[System Shifts{suffix}] Computing organ system distribution shifts...")
    
    # Compute system shares per year
    sys_year = (
        death_df.groupby(["year", "system"])
        .size()
        .rename("count")
        .reset_index()
    )
    
    # Convert to %
    sys_totals = sys_year.groupby("year")["count"].transform("sum")
    sys_year["pct"] = sys_year["count"] / sys_totals
    
    # Save yearly system distributions (only for all-ages, not age-restricted)
    if suffix == "":
        yearly_file = output_path / "icd_system_yearly.csv"
        sys_year.to_csv(yearly_file, index=False)
        print(f"  Saved: {yearly_file}")
    
    # Compute pre vs post shifts
    sys_pre = sys_year[sys_year["year"].between(2010, 2019)]
    sys_post = sys_year[sys_year["year"].between(2021, 2023)]
    
    sys_pre_mean = sys_pre.groupby("system")["pct"].mean()
    sys_post_mean = sys_post.groupby("system")["pct"].mean()
    
    sys_shift = pd.DataFrame({
        "pre_pct": sys_pre_mean,
        "post_pct": sys_post_mean
    }).fillna(0)
    
    sys_shift["diff"] = sys_shift["post_pct"] - sys_shift["pre_pct"]
    sys_shift.sort_values("diff", ascending=False, inplace=True)
    
    shift_file = output_path / f"icd_system_shift{suffix}.csv"
    sys_shift.to_csv(shift_file)
    print(f"  Saved: {shift_file}")
    
    return sys_shift


def main():
    # Parse command-line arguments
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_dir = "../data/Czech2/"
    else:
        input_file = "../data/Czech2/data.csv"
        output_dir = "../data/Czech2/"
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[Population Shift Analysis] Reading input file: {input_file}")
    print(f"[Population Shift Analysis] Output directory: {output_dir}")
    
    # Step 1: Load and prepare death dataset
    print("\n[Step 1] Loading and preparing death dataset...")
    try:
        df = pd.read_csv(input_file, dtype=str, low_memory=False)
        print(f"  Loaded {len(df):,} total records")
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}")
        sys.exit(1)
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Filter to death records
    death_df = df[df["udalost"].str.lower() == "umrti"].copy()
    print(f"  Found {len(death_df):,} death records")
    
    if len(death_df) == 0:
        print("ERROR: No death records found")
        sys.exit(1)
    
    # Identify ICD column
    icd_cols = [c for c in death_df.columns if "diag" in c]
    if not icd_cols:
        print("ERROR: No ICD diagnosis column found (expected column containing 'diag')")
        sys.exit(1)
    
    icd_col = icd_cols[0]
    print(f"  Using ICD column: {icd_col}")
    
    death_df["icd"] = death_df[icd_col].str.strip()
    death_df = death_df[death_df["icd"].notna() & (death_df["icd"] != "")]
    print(f"  Death records with ICD codes: {len(death_df):,}")
    
    # Convert key fields to numeric
    death_df["year"] = pd.to_numeric(death_df["rok_udalosti"], errors="coerce")
    death_df["age"] = (
        pd.to_numeric(death_df["rok_udalosti"], errors="coerce") -
        pd.to_numeric(death_df["rok_narozeni"], errors="coerce")
    )
    
    # Step 2: Restrict to clean analysis window
    print("\n[Step 2] Restricting to analysis window (2010-2023)...")
    death_df = death_df[death_df["year"].between(2010, 2023)]
    print(f"  Death records in analysis window: {len(death_df):,}")
    
    if len(death_df) == 0:
        print("ERROR: No death records in analysis window")
        sys.exit(1)
    
    # Step 3-4: Compute ICD shifts (all ages)
    shift_all = compute_icd_shifts(death_df, output_path)
    
    # Step 5: Organ-system analysis (all ages)
    print("\n[Step 5] Adding organ-system analysis...")
    lookup_file = output_path / "icd_system_lookup.csv"
    if lookup_file.exists():
        lookup = pd.read_csv(lookup_file)
        death_df_system = death_df.copy()
        death_df_system["icd_prefix"] = death_df_system["icd"].str[:1]
        death_df_system = death_df_system.merge(lookup, on="icd_prefix", how="left")
        death_df_system["system"] = death_df_system["system"].fillna("other")
        
        sys_shift_all = compute_system_shifts(death_df_system, output_path)
    else:
        print(f"  Warning: Lookup file not found: {lookup_file}")
        print("  Skipping organ-system analysis")
        death_df_system = None
    
    # Step 6: Age-restricted population analysis (65-89 years)
    print("\n[Step 6] Age-restricted analysis (65-89 years)...")
    senior = death_df[death_df["age"].between(65, 89)].copy()
    print(f"  Death records in age range 65-89: {len(senior):,}")
    
    if len(senior) > 0:
        # Re-run ICD shifts on age-restricted subset
        shift_senior = compute_icd_shifts(senior, output_path, suffix="_age65_89")
        
        # Re-run system shifts on age-restricted subset
        if death_df_system is not None:
            senior_system = death_df_system[death_df_system["age"].between(65, 89)].copy()
            sys_shift_senior = compute_system_shifts(senior_system, output_path, suffix="_age65_89")
    else:
        print("  Warning: No deaths in age range 65-89")
    
    print("\n[Population Shift Analysis] Analysis complete!")
    print(f"\nOutput files:")
    print(f"  - {output_path / 'icd_population_shift.csv'}")
    print(f"  - {output_path / 'icd_system_yearly.csv'}")
    print(f"  - {output_path / 'icd_system_shift.csv'}")
    print(f"  - {output_path / 'icd_population_shift_age65_89.csv'}")
    print(f"  - {output_path / 'icd_system_shift_age65_89.csv'}")


if __name__ == "__main__":
    main()


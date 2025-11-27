#!/usr/bin/env python3
"""
KCOR Mortality Analysis Pipeline

This script implements person-month survival analysis on Czech event-level data
to estimate vaccination effects on mortality risk using KCOR methodology.

The pipeline:
1. Builds fixed cohorts based on vaccination status at enrollment date
2. Constructs person-month survival data from enrollment forward
3. Computes cohort-specific hazard curves
4. Applies KCOR slope-normalization to remove age/health structure bias
5. Computes adjusted cumulative hazards and KCOR ratios

USAGE:
    python kcor_mortality.py <input.csv> <output_dir> [--enroll-year YEAR] [--enroll-month MONTH] [--max-fu-months MONTHS] [--quiet-min MIN] [--quiet-max MAX]

DEPENDENCIES:
    pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def ym_to_index(year: int, month: int, base_year: int, base_month: int) -> int:
    """
    Convert year-month to time index relative to base year-month.
    
    Args:
        year: Year of event
        month: Month of event (1-12)
        base_year: Base year (enrollment year)
        base_month: Base month (enrollment month)
    
    Returns:
        Time index (0 = enrollment month)
    """
    return (year - base_year) * 12 + (month - base_month)


def label_cohort(dose: int, separate_doses: bool = False) -> str:
    """
    Label cohort based on dose number.
    
    Args:
        dose: Number of doses (0, 1, 2, 3, ...)
        separate_doses: If True, separate doses 3,4,5,6; else group as dose3plus
    
    Returns:
        Cohort label string
    """
    if dose == 0:
        return "dose0_unvaccinated"
    elif dose == 1:
        return "dose1"
    elif dose == 2:
        return "dose2"
    elif separate_doses and dose >= 3:
        return f"dose{dose}"
    else:
        return "dose3plus"


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Load Czech event-level data and preprocess.
    
    Args:
        csv_path: Path to input CSV file
    
    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Convert numeric fields
    numeric_cols = [
        "rok_udalosti", "mesic_udalosti", "rok_narozeni",
        "covid_ocko_poradi_davky", "covid_onemoc_poradi_infekce"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    print(f"Loaded {len(df):,} records")
    return df


def build_baseline_cohorts(
    df: pd.DataFrame,
    enroll_year: int,
    enroll_month: int,
    separate_doses: bool = False
) -> pd.DataFrame:
    """
    Build baseline cohorts based on vaccination status at enrollment.
    
    Args:
        df: Preprocessed event-level DataFrame
        enroll_year: Enrollment year
        enroll_month: Enrollment month (1-12)
        separate_doses: If True, separate doses 3,4,5,6
    
    Returns:
        DataFrame with columns: id_zeny, baseline_dose, cohort
    """
    print(f"Building baseline cohorts for enrollment {enroll_year}-{enroll_month:02d}...")
    
    # Extract vaccination events
    vax_events = df[df["udalost"].str.lower() == "covid ockovani"].copy()
    if len(vax_events) == 0:
        raise ValueError("No vaccination events found. Check 'udalost' column values.")
    
    # Compute time index for vaccination events
    vax_events["t_index"] = vax_events.apply(
        lambda r: ym_to_index(
            int(r["rok_udalosti"]) if pd.notna(r["rok_udalosti"]) else enroll_year,
            int(r["mesic_udalosti"]) if pd.notna(r["mesic_udalosti"]) else enroll_month,
            enroll_year,
            enroll_month
        ),
        axis=1
    )
    
    # Keep only vaccinations on or before enrollment
    vax_before = vax_events[vax_events["t_index"] <= 0].copy()
    
    # Get maximum dose per person before enrollment
    baseline_dose = (
        vax_before.groupby("id_zeny")["covid_ocko_poradi_davky"]
        .max()
        .rename("baseline_dose")
    )
    
    # Create baseline DataFrame for all persons
    persons = df["id_zeny"].dropna().unique()
    baseline = pd.DataFrame({"id_zeny": persons})
    
    # Merge baseline doses
    baseline = baseline.merge(baseline_dose, on="id_zeny", how="left")
    baseline["baseline_dose"] = baseline["baseline_dose"].fillna(0).astype(int)
    
    # Label cohorts
    baseline["cohort"] = baseline["baseline_dose"].apply(
        lambda d: label_cohort(d, separate_doses)
    )
    
    print(f"Baseline cohorts: {baseline['cohort'].value_counts().to_dict()}")
    return baseline


def determine_death_times(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    enroll_year: int,
    enroll_month: int
) -> pd.DataFrame:
    """
    Determine death times for each person and merge into baseline.
    
    Args:
        df: Preprocessed event-level DataFrame
        baseline: Baseline cohort DataFrame
        enroll_year: Enrollment year
        enroll_month: Enrollment month
    
    Returns:
        Baseline DataFrame with death_t column added
    """
    print("Determining death times...")
    
    # Extract death events
    death_events = df[df["udalost"].str.lower() == "umrti"].copy()
    
    if len(death_events) > 0:
        # Compute time index for death events
        death_events["t_index"] = death_events.apply(
            lambda r: ym_to_index(
                int(r["rok_udalosti"]) if pd.notna(r["rok_udalosti"]) else enroll_year,
                int(r["mesic_udalosti"]) if pd.notna(r["mesic_udalosti"]) else enroll_month,
                enroll_year,
                enroll_month
            ),
            axis=1
        )
        
        # Get first death per person
        death_time = (
            death_events.groupby("id_zeny")["t_index"]
            .min()
            .rename("death_t")
        )
        
        # Merge into baseline
        baseline = baseline.merge(death_time, on="id_zeny", how="left")
    else:
        baseline["death_t"] = np.nan
    
    # Exclude people who died before enrollment
    before = len(baseline)
    baseline = baseline[(baseline["death_t"].isna()) | (baseline["death_t"] >= 0)].copy()
    after = len(baseline)
    
    if before != after:
        print(f"Excluded {before - after} persons who died before enrollment")
    
    return baseline


def build_person_month_table(
    baseline: pd.DataFrame,
    max_fu_months: int
) -> pd.DataFrame:
    """
    Build person-month survival table from enrollment onward.
    
    Args:
        baseline: Baseline DataFrame with cohort and death_t
        max_fu_months: Maximum follow-up months
    
    Returns:
        Person-month DataFrame with columns: id_zeny, cohort, t, event
    """
    print(f"Building person-month table (max follow-up: {max_fu_months} months)...")
    
    rows = []
    
    for _, row in baseline.iterrows():
        pid = row["id_zeny"]
        cohort = row["cohort"]
        death_t = row["death_t"]  # may be NaN
        
        # Last month observed for this person
        if pd.isna(death_t):
            last_t = max_fu_months
        else:
            last_t = min(int(death_t), max_fu_months)
        
        # Create rows for each month from t=0 to last_t
        for t in range(0, last_t + 1):
            event = 0
            if not pd.isna(death_t) and int(death_t) == t:
                event = 1
            rows.append((pid, cohort, t, event))
    
    pm = pd.DataFrame(rows, columns=["id_zeny", "cohort", "t", "event"])
    print(f"Created {len(pm):,} person-month records")
    return pm


def compute_hazard_curves(pm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly hazard curves per cohort.
    
    Args:
        pm: Person-month DataFrame
    
    Returns:
        Hazard DataFrame with columns: cohort, t, deaths, at_risk, hazard
    """
    print("Computing hazard curves...")
    
    # Number of events per cohort and month
    events = (
        pm.groupby(["cohort", "t"])["event"]
        .sum()
        .rename("deaths")
        .reset_index()
    )
    
    # Number at risk at start of each month
    risk = (
        pm.groupby(["cohort", "t"])["id_zeny"]
        .nunique()
        .rename("at_risk")
        .reset_index()
    )
    
    # Merge and compute hazards
    haz = events.merge(risk, on=["cohort", "t"], how="right").fillna(0)
    haz["hazard"] = haz["deaths"] / (haz["at_risk"] + 1e-12)  # Avoid division by zero
    
    # Sort by cohort and time
    haz = haz.sort_values(["cohort", "t"]).reset_index(drop=True)
    
    print(f"Computed hazards for {len(haz)} cohort-month combinations")
    return haz


def fit_slopes(
    haz: pd.DataFrame,
    quiet_t_min: int,
    quiet_t_max: int
) -> pd.DataFrame:
    """
    Fit Gompertz slopes per cohort in quiet period.
    
    Args:
        haz: Hazard DataFrame
        quiet_t_min: Start of quiet period (months)
        quiet_t_max: End of quiet period (months)
    
    Returns:
        Slopes DataFrame with columns: cohort, intercept, slope
    """
    print(f"Fitting slopes in quiet period [{quiet_t_min}, {quiet_t_max}]...")
    
    # Compute log hazard (replace 0 with NaN)
    haz["log_hazard"] = np.log(haz["hazard"].replace(0, np.nan))
    
    # Filter to quiet period
    quiet_data = haz[(haz["t"] >= quiet_t_min) & (haz["t"] <= quiet_t_max)].copy()
    
    slopes = []
    
    for cohort, g in quiet_data.groupby("cohort"):
        g = g.dropna(subset=["log_hazard"])
        if len(g) < 2:
            print(f"  Warning: Cohort {cohort} has <2 valid points in quiet period, skipping")
            continue
        
        x = g["t"].values
        y = g["log_hazard"].values
        
        # Simple linear regression: y = a + b*x
        try:
            b, a = np.polyfit(x, y, 1)
            slopes.append((cohort, a, b))
            print(f"  {cohort}: slope={b:.6f}, intercept={a:.6f}")
        except Exception as e:
            print(f"  Warning: Failed to fit slope for {cohort}: {e}")
            continue
    
    slopes_df = pd.DataFrame(slopes, columns=["cohort", "intercept", "slope"])
    
    if len(slopes_df) == 0:
        raise ValueError("No slopes could be fitted. Check quiet period and data.")
    
    return slopes_df


def apply_slope_normalization(
    haz: pd.DataFrame,
    slopes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply slope normalization to hazards.
    
    Args:
        haz: Hazard DataFrame
        slopes_df: Slopes DataFrame
    
    Returns:
        Hazard DataFrame with hazard_adj column added
    """
    print("Applying slope normalization...")
    
    # Merge slopes
    haz = haz.merge(slopes_df[["cohort", "slope"]], on="cohort", how="left")
    
    # Apply normalization: hazard_adj = hazard * exp(-slope * t)
    haz["hazard_adj"] = haz["hazard"] * np.exp(-haz["slope"] * haz["t"])
    
    # Fill NaN for cohorts without slopes (use original hazard)
    haz["hazard_adj"] = haz["hazard_adj"].fillna(haz["hazard"])
    
    return haz


def compute_cumulative_hazards(haz: pd.DataFrame) -> pd.DataFrame:
    """
    Compute adjusted cumulative hazards per cohort.
    
    Args:
        haz: Hazard DataFrame with hazard_adj
    
    Returns:
        Hazard DataFrame with cum_hazard_adj column added
    """
    print("Computing cumulative hazards...")
    
    # Ensure sorted
    haz = haz.sort_values(["cohort", "t"]).reset_index(drop=True)
    
    # Cumulative sum per cohort
    haz["cum_hazard_adj"] = (
        haz.groupby("cohort")["hazard_adj"]
        .cumsum()
    )
    
    return haz


def compute_kcor_ratios(
    haz: pd.DataFrame,
    ref_cohort: str = "dose0_unvaccinated",
    vax_cohorts: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute KCOR ratios (vaccinated vs unvaccinated).
    
    Args:
        haz: Hazard DataFrame with cum_hazard_adj
        ref_cohort: Reference cohort name (unvaccinated)
        vax_cohorts: List of vaccinated cohort names (if None, auto-detect)
    
    Returns:
        KCOR ratios DataFrame
    """
    print(f"Computing KCOR ratios (reference: {ref_cohort})...")
    
    # Auto-detect vaccinated cohorts if not provided
    if vax_cohorts is None:
        all_cohorts = haz["cohort"].unique()
        vax_cohorts = [c for c in all_cohorts if c != ref_cohort]
    
    # Get reference cumulative hazards
    ref = haz[haz["cohort"] == ref_cohort][["t", "cum_hazard_adj"]].rename(
        columns={"cum_hazard_adj": "cum_ref"}
    )
    
    if len(ref) == 0:
        raise ValueError(f"Reference cohort '{ref_cohort}' not found in data")
    
    kcor_rows = []
    
    for vc in vax_cohorts:
        sub = haz[haz["cohort"] == vc][["t", "cum_hazard_adj"]].rename(
            columns={"cum_hazard_adj": "cum_vax"}
        )
        
        if len(sub) == 0:
            print(f"  Warning: Cohort {vc} not found, skipping")
            continue
        
        merged = sub.merge(ref, on="t", how="inner")
        merged["cohort"] = vc
        merged["kcor_ratio"] = merged["cum_vax"] / (merged["cum_ref"] + 1e-12)
        kcor_rows.append(merged)
    
    if len(kcor_rows) == 0:
        raise ValueError("No KCOR ratios could be computed")
    
    kcor = pd.concat(kcor_rows, ignore_index=True)
    return kcor


def plot_kcor_ratios(
    kcor: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot KCOR ratios over time.
    
    Args:
        kcor: KCOR ratios DataFrame
        output_path: Path to save plot
        title: Plot title (optional)
    """
    print(f"Creating KCOR ratio plot: {output_path}")
    
    plt.figure(figsize=(10, 6))
    
    cohorts = kcor["cohort"].unique()
    for cohort in cohorts:
        sub = kcor[kcor["cohort"] == cohort]
        plt.plot(sub["t"], sub["kcor_ratio"], label=cohort, marker='o', markersize=3)
    
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="No effect (KCOR=1)")
    plt.xlabel("Months since enrollment", fontsize=12)
    plt.ylabel("KCOR adjusted cumulative hazard ratio", fontsize=12)
    plt.title(title or "KCOR: Vaccinated vs Unvaccinated (slope-normalized)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def run_kcor_pipeline(
    csv_path: str,
    output_dir: str,
    enroll_year: int = 2021,
    enroll_month: int = 7,
    max_fu_months: int = 24,
    quiet_t_min: int = 3,
    quiet_t_max: int = 10,
    separate_doses: bool = False,
    ref_cohort: str = "dose0_unvaccinated"
) -> Dict[str, pd.DataFrame]:
    """
    Run complete KCOR mortality analysis pipeline.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Output directory for results
        enroll_year: Enrollment year
        enroll_month: Enrollment month (1-12)
        max_fu_months: Maximum follow-up months
        quiet_t_min: Start of quiet period for slope estimation
        quiet_t_max: End of quiet period for slope estimation
        separate_doses: If True, separate doses 3,4,5,6
        ref_cohort: Reference cohort name
    
    Returns:
        Dictionary of result DataFrames
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    results_dir = os.path.join(output_dir, "results")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 60)
    print("KCOR Mortality Analysis Pipeline")
    print("=" * 60)
    print(f"Enrollment: {enroll_year}-{enroll_month:02d}")
    print(f"Follow-up: {max_fu_months} months")
    print(f"Quiet period: [{quiet_t_min}, {quiet_t_max}] months")
    print("=" * 60)
    
    # Step 1: Load and preprocess
    df = load_and_preprocess_data(csv_path)
    
    # Step 2: Build baseline cohorts
    baseline = build_baseline_cohorts(df, enroll_year, enroll_month, separate_doses)
    
    # Step 3: Determine death times
    baseline = determine_death_times(df, baseline, enroll_year, enroll_month)
    
    # Step 4: Build person-month table
    pm = build_person_month_table(baseline, max_fu_months)
    
    # Step 5: Compute hazard curves
    haz = compute_hazard_curves(pm)
    
    # Save raw hazards
    haz_raw_path = os.path.join(raw_dir, "kcor_hazard_raw.csv")
    haz[["cohort", "t", "deaths", "at_risk", "hazard"]].to_csv(haz_raw_path, index=False)
    print(f"Saved raw hazards to {haz_raw_path}")
    
    # Step 6: Fit slopes
    slopes_df = fit_slopes(haz, quiet_t_min, quiet_t_max)
    
    # Save slopes
    slopes_path = os.path.join(raw_dir, "kcor_slopes.csv")
    slopes_df.to_csv(slopes_path, index=False)
    print(f"Saved slopes to {slopes_path}")
    
    # Step 7: Apply slope normalization
    haz = apply_slope_normalization(haz, slopes_df)
    
    # Step 8: Compute cumulative hazards
    haz = compute_cumulative_hazards(haz)
    
    # Save adjusted hazards
    haz_adj_path = os.path.join(raw_dir, "kcor_hazard_adjusted.csv")
    haz.to_csv(haz_adj_path, index=False)
    print(f"Saved adjusted hazards to {haz_adj_path}")
    
    # Step 9: Compute KCOR ratios
    kcor = compute_kcor_ratios(haz, ref_cohort)
    
    # Save KCOR ratios
    kcor_path = os.path.join(results_dir, "kcor_ratios.csv")
    kcor.to_csv(kcor_path, index=False)
    print(f"Saved KCOR ratios to {kcor_path}")
    
    # Step 10: Create plot
    plot_path = os.path.join(plots_dir, "kcor_ratio_plot.png")
    title = f"KCOR: Vaccinated vs Unvaccinated (enrollment: {enroll_year}-{enroll_month:02d})"
    plot_kcor_ratios(kcor, plot_path, title)
    
    # Create summary
    summary = {
        "enrollment": f"{enroll_year}-{enroll_month:02d}",
        "max_fu_months": max_fu_months,
        "quiet_period": f"[{quiet_t_min}, {quiet_t_max}]",
        "cohorts": baseline["cohort"].value_counts().to_dict(),
        "total_persons": len(baseline),
        "total_person_months": len(pm)
    }
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return {
        "haz": haz,
        "slopes": slopes_df,
        "kcor": kcor,
        "baseline": baseline,
        "pm": pm,
        "summary": summary
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KCOR Mortality Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--enroll-year", type=int, default=2021, help="Enrollment year (default: 2021)")
    parser.add_argument("--enroll-month", type=int, default=7, help="Enrollment month 1-12 (default: 7)")
    parser.add_argument("--max-fu-months", type=int, default=24, help="Maximum follow-up months (default: 24)")
    parser.add_argument("--quiet-min", type=int, default=3, help="Quiet period start month (default: 3)")
    parser.add_argument("--quiet-max", type=int, default=10, help="Quiet period end month (default: 10)")
    parser.add_argument("--separate-doses", action="store_true", help="Separate doses 3,4,5,6 instead of grouping")
    parser.add_argument("--ref-cohort", default="dose0_unvaccinated", help="Reference cohort (default: dose0_unvaccinated)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return 1
    
    if not (1 <= args.enroll_month <= 12):
        print(f"Error: Enrollment month must be 1-12, got {args.enroll_month}")
        return 1
    
    if args.quiet_min >= args.quiet_max:
        print(f"Error: Quiet period min ({args.quiet_min}) must be < max ({args.quiet_max})")
        return 1
    
    try:
        results = run_kcor_pipeline(
            csv_path=args.input_csv,
            output_dir=args.output_dir,
            enroll_year=args.enroll_year,
            enroll_month=args.enroll_month,
            max_fu_months=args.max_fu_months,
            quiet_t_min=args.quiet_min,
            quiet_t_max=args.quiet_max,
            separate_doses=args.separate_doses,
            ref_cohort=args.ref_cohort
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


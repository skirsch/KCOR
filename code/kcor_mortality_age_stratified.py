#!/usr/bin/env python3
"""
KCOR Mortality Age-Stratified Analysis

This script extends the KCOR mortality pipeline to perform age-stratified analysis,
running the full pipeline within each age band to reduce residual confounding.

USAGE:
    python kcor_mortality_age_stratified.py <input.csv> <output_dir> [--age-bands BANDS] [--enroll-year YEAR] [--enroll-month MONTH]

DEPENDENCIES:
    pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import the core pipeline
from kcor_mortality import (
    load_and_preprocess_data,
    build_baseline_cohorts,
    determine_death_times,
    build_person_month_table,
    compute_hazard_curves,
    fit_slopes,
    apply_slope_normalization,
    compute_cumulative_hazards,
    compute_kcor_ratios,
    plot_kcor_ratios
)


def default_age_bands() -> List[Tuple[str, int, int]]:
    """
    Return default age bands.
    
    Returns:
        List of (label, min_age, max_age) tuples
    """
    return [
        ("65-74", 65, 74),
        ("75-84", 75, 84),
        ("85+", 85, 200)  # 200 as upper bound
    ]


def compute_age_at_enrollment(
    df: pd.DataFrame,
    enroll_year: int,
    enroll_month: int
) -> pd.DataFrame:
    """
    Compute age at enrollment for each person.
    
    Args:
        df: Preprocessed event-level DataFrame
        enroll_year: Enrollment year
        enroll_month: Enrollment month
    
    Returns:
        DataFrame with id_zeny and age_enroll columns
    """
    # Get birth year per person
    people_age = (
        df.groupby("id_zeny")[["rok_narozeni"]]
        .first()
        .reset_index()
    )
    
    # Compute age at enrollment (approximate: age at enrollment year)
    people_age["age_enroll"] = enroll_year - people_age["rok_narozeni"]
    
    # Handle missing/invalid birth years
    people_age = people_age[people_age["age_enroll"].between(0, 150)].copy()
    
    return people_age[["id_zeny", "age_enroll"]]


def assign_age_band(
    baseline: pd.DataFrame,
    age_df: pd.DataFrame,
    age_bands: List[Tuple[str, int, int]]
) -> pd.DataFrame:
    """
    Assign age band to each person in baseline.
    
    Args:
        baseline: Baseline cohort DataFrame
        age_df: DataFrame with id_zeny and age_enroll
        age_bands: List of (label, min_age, max_age) tuples
    
    Returns:
        Baseline DataFrame with age_band column added
    """
    # Merge age information
    baseline = baseline.merge(age_df, on="id_zeny", how="left")
    
    # Assign age bands
    def get_age_band(age):
        if pd.isna(age):
            return "unknown"
        for label, min_age, max_age in age_bands:
            if min_age <= age <= max_age:
                return label
        return "out_of_range"
    
    baseline["age_band"] = baseline["age_enroll"].apply(get_age_band)
    
    return baseline


def run_age_stratified_analysis(
    csv_path: str,
    output_dir: str,
    enroll_year: int = 2021,
    enroll_month: int = 7,
    max_fu_months: int = 24,
    quiet_t_min: int = 3,
    quiet_t_max: int = 10,
    separate_doses: bool = False,
    age_bands: Optional[List[Tuple[str, int, int]]] = None,
    min_persons_per_band: int = 100
) -> Dict[str, pd.DataFrame]:
    """
    Run KCOR analysis stratified by age bands.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Output directory for results
        enroll_year: Enrollment year
        enroll_month: Enrollment month
        max_fu_months: Maximum follow-up months
        quiet_t_min: Start of quiet period
        quiet_t_max: End of quiet period
        separate_doses: If True, separate doses 3,4,5,6
        age_bands: List of (label, min_age, max_age) tuples (if None, uses defaults)
        min_persons_per_band: Minimum persons required per age band
    
    Returns:
        Dictionary mapping age_band to result dictionaries
    """
    if age_bands is None:
        age_bands = default_age_bands()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("KCOR Mortality Age-Stratified Analysis")
    print("=" * 80)
    print(f"Enrollment: {enroll_year}-{enroll_month:02d}")
    print(f"Age bands: {[b[0] for b in age_bands]}")
    print("=" * 80)
    
    # Step 1: Load and preprocess
    df = load_and_preprocess_data(csv_path)
    
    # Step 2: Compute ages
    print("\nComputing ages at enrollment...")
    age_df = compute_age_at_enrollment(df, enroll_year, enroll_month)
    print(f"Computed ages for {len(age_df):,} persons")
    
    # Step 3: Build baseline cohorts
    baseline = build_baseline_cohorts(df, enroll_year, enroll_month, separate_doses)
    
    # Step 4: Assign age bands
    baseline = assign_age_band(baseline, age_df, age_bands)
    
    # Step 5: Determine death times
    baseline = determine_death_times(df, baseline, enroll_year, enroll_month)
    
    # Check age band sizes
    print("\nAge band sizes:")
    age_band_counts = baseline["age_band"].value_counts()
    for band, count in age_band_counts.items():
        print(f"  {band}: {count:,} persons")
    
    # Run analysis for each age band
    results = {}
    
    for label, min_age, max_age in age_bands:
        print(f"\n{'=' * 80}")
        print(f"Processing age band: {label} ({min_age}-{max_age} years)")
        print(f"{'=' * 80}")
        
        # Filter baseline to this age band
        band_baseline = baseline[baseline["age_band"] == label].copy()
        
        if len(band_baseline) < min_persons_per_band:
            print(f"  Warning: Only {len(band_baseline)} persons in {label}, skipping (minimum: {min_persons_per_band})")
            continue
        
        # Create output subdirectory
        band_output_dir = os.path.join(output_dir, "age_stratified", label)
        os.makedirs(band_output_dir, exist_ok=True)
        raw_dir = os.path.join(band_output_dir, "raw")
        results_dir = os.path.join(band_output_dir, "results")
        plots_dir = os.path.join(band_output_dir, "plots")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            # Build person-month table
            pm = build_person_month_table(band_baseline, max_fu_months)
            
            # Compute hazard curves
            haz = compute_hazard_curves(pm)
            
            # Save raw hazards
            haz_raw_path = os.path.join(raw_dir, "kcor_hazard_raw.csv")
            haz[["cohort", "t", "deaths", "at_risk", "hazard"]].to_csv(haz_raw_path, index=False)
            
            # Fit slopes
            slopes_df = fit_slopes(haz, quiet_t_min, quiet_t_max)
            
            # Save slopes
            slopes_path = os.path.join(raw_dir, "kcor_slopes.csv")
            slopes_df.to_csv(slopes_path, index=False)
            
            # Apply slope normalization
            haz = apply_slope_normalization(haz, slopes_df)
            
            # Compute cumulative hazards
            haz = compute_cumulative_hazards(haz)
            
            # Save adjusted hazards
            haz_adj_path = os.path.join(raw_dir, "kcor_hazard_adjusted.csv")
            haz.to_csv(haz_adj_path, index=False)
            
            # Compute KCOR ratios
            kcor = compute_kcor_ratios(haz)
            
            # Save KCOR ratios
            kcor_path = os.path.join(results_dir, "kcor_ratios.csv")
            kcor.to_csv(kcor_path, index=False)
            
            # Create plot
            plot_path = os.path.join(plots_dir, "kcor_ratio_plot.png")
            title = f"KCOR: {label} (enrollment: {enroll_year}-{enroll_month:02d})"
            plot_kcor_ratios(kcor, plot_path, title)
            
            # Store results
            results[label] = {
                "haz": haz,
                "slopes": slopes_df,
                "kcor": kcor,
                "baseline": band_baseline,
                "pm": pm,
                "n_persons": len(band_baseline)
            }
            
            print(f"  ✓ Completed analysis for {label}")
            
        except Exception as e:
            print(f"  ✗ Failed analysis for {label}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create age-stratified summary
    print(f"\n{'=' * 80}")
    print("Age-Stratified Analysis Summary")
    print(f"{'=' * 80}")
    
    summary_rows = []
    for label, result in results.items():
        kcor_df = result["kcor"]
        for cohort in kcor_df["cohort"].unique():
            cohort_data = kcor_df[kcor_df["cohort"] == cohort]
            if len(cohort_data) > 0:
                final_row = cohort_data.iloc[-1]
                summary_rows.append({
                    "age_band": label,
                    "cohort": cohort,
                    "n_persons": result["n_persons"],
                    "final_t": final_row["t"],
                    "final_kcor": final_row["kcor_ratio"]
                })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "age_stratified", "summary_by_age_band.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved age-stratified summary to {summary_path}")
        print("\n" + summary_df.to_string(index=False))
    
    return results


def compute_age_standardized_kcor(
    results: Dict[str, Dict],
    output_dir: str
) -> pd.DataFrame:
    """
    Compute age-standardized KCOR (similar to ASMR in KCOR.py).
    
    This weights age-specific KCOR values by expected deaths in each age band.
    
    Args:
        results: Dictionary mapping age_band to result dictionaries
        output_dir: Output directory
    
    Returns:
        Age-standardized KCOR DataFrame
    """
    print("\n" + "=" * 80)
    print("Computing Age-Standardized KCOR")
    print("=" * 80)
    
    # Collect cumulative hazards by age band and cohort
    age_band_data = []
    
    for label, result in results.items():
        haz = result["haz"]
        baseline = result["baseline"]
        
        # Compute weights based on person-time and hazard in quiet period
        # Similar to KCOR.py ASMR weighting: w_a ∝ h_a × PT_a(W)
        quiet_haz = haz[
            (haz["t"] >= 3) & (haz["t"] <= 10)
        ].copy()
        
        if len(quiet_haz) > 0:
            # Average hazard in quiet period per cohort
            avg_hazard = quiet_haz.groupby("cohort")["hazard"].mean()
            # Person-time in quiet period per cohort
            person_time = quiet_haz.groupby("cohort")["at_risk"].sum()
            
            # Weight = hazard × person-time
            weights = avg_hazard * person_time
            
            for cohort in haz["cohort"].unique():
                cohort_haz = haz[haz["cohort"] == cohort].sort_values("t")
                if len(cohort_haz) > 0:
                    age_band_data.append({
                        "age_band": label,
                        "cohort": cohort,
                        "weight": weights.get(cohort, 0.0),
                        "cum_hazard_adj": cohort_haz["cum_hazard_adj"].values
                    })
    
    if not age_band_data:
        print("Warning: No data available for age standardization")
        return pd.DataFrame()
    
    # Group by cohort and compute weighted average
    standardized_rows = []
    
    # Get all cohorts
    all_cohorts = set()
    for data in age_band_data:
        all_cohorts.add(data["cohort"])
    
    # Get reference cohort (unvaccinated)
    ref_cohort = "dose0_unvaccinated"
    vax_cohorts = [c for c in all_cohorts if c != ref_cohort]
    
    # For each vaccinated cohort, compute weighted average vs reference
    for vax_cohort in vax_cohorts:
        # Get reference cumulative hazards by age band
        ref_data = {d["age_band"]: d["cum_hazard_adj"] for d in age_band_data if d["cohort"] == ref_cohort}
        vax_data = {d["age_band"]: d["cum_hazard_adj"] for d in age_band_data if d["cohort"] == vax_cohort}
        weights_dict = {d["age_band"]: d["weight"] for d in age_band_data if d["cohort"] == vax_cohort}
        
        # Find common age bands
        common_bands = set(ref_data.keys()) & set(vax_data.keys())
        
        if not common_bands:
            continue
        
        # For each time point, compute weighted average ratio
        max_len = max(len(ref_data[b]) for b in common_bands)
        
        for t in range(max_len):
            weighted_ratios = []
            total_weight = 0
            
            for band in common_bands:
                if t < len(ref_data[band]) and t < len(vax_data[band]):
                    ref_ch = ref_data[band][t]
                    vax_ch = vax_data[band][t]
                    weight = weights_dict.get(band, 0.0)
                    
                    if ref_ch > 0 and weight > 0:
                        ratio = vax_ch / ref_ch
                        weighted_ratios.append(ratio * weight)
                        total_weight += weight
            
            if total_weight > 0:
                standardized_ratio = sum(weighted_ratios) / total_weight
                standardized_rows.append({
                    "cohort": vax_cohort,
                    "t": t,
                    "kcor_ratio_standardized": standardized_ratio
                })
    
    if standardized_rows:
        standardized_df = pd.DataFrame(standardized_rows)
        
        # Save
        standardized_path = os.path.join(output_dir, "age_stratified", "kcor_age_standardized.csv")
        standardized_df.to_csv(standardized_path, index=False)
        print(f"\nSaved age-standardized KCOR to {standardized_path}")
        
        return standardized_df
    
    return pd.DataFrame()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KCOR Mortality Age-Stratified Analysis",
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
    parser.add_argument("--age-bands", nargs="+", help="Age bands as 'label:min:max' (e.g., '65-74:65:74')")
    parser.add_argument("--min-persons", type=int, default=100, help="Minimum persons per age band (default: 100)")
    
    args = parser.parse_args()
    
    # Parse age bands if provided
    age_bands = None
    if args.age_bands:
        age_bands = []
        for band_str in args.age_bands:
            parts = band_str.split(":")
            if len(parts) == 3:
                label, min_age, max_age = parts
                age_bands.append((label, int(min_age), int(max_age)))
            else:
                print(f"Warning: Invalid age band format '{band_str}', using defaults")
                age_bands = None
                break
    
    # Validate arguments
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return 1
    
    try:
        results = run_age_stratified_analysis(
            csv_path=args.input_csv,
            output_dir=args.output_dir,
            enroll_year=args.enroll_year,
            enroll_month=args.enroll_month,
            max_fu_months=args.max_fu_months,
            quiet_t_min=args.quiet_min,
            quiet_t_max=args.quiet_max,
            separate_doses=args.separate_doses,
            age_bands=age_bands,
            min_persons_per_band=args.min_persons
        )
        
        # Compute age-standardized KCOR
        compute_age_standardized_kcor(results, args.output_dir)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


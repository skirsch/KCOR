#!/usr/bin/env python3
"""
Extract cumulative hazard values for Table 3 from KCOR_CMR.xlsx.

This script computes raw cumulative hazards for Dose 0 and Dose 2
by age band, matching Table 2's age band structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Constants matching KCOR.py
DYNAMIC_HVE_SKIP_WEEKS = 2
EPS = 1e-12

def hazard_from_mr_improved(mr: np.ndarray) -> np.ndarray:
    """Improved discrete-time hazard transform for MR measured as D / N_start.
    
    h = -ln((1 - 1.5 MR) / (1 - 0.5 MR))
    """
    mr_clipped = np.clip(mr, 0.0, 0.999)
    num = 1.0 - 1.5 * mr_clipped
    den = 1.0 - 0.5 * mr_clipped
    num = np.clip(num, EPS, None)
    den = np.clip(den, EPS, None)
    return -np.log(num / den)


def get_age_band(yob: int, reference_year: int = 2020) -> str:
    """Map YearOfBirth to age band using reference year 2020."""
    if yob == -2:
        return "All ages (full population)"
    if yob < 0:
        return None  # Skip special values like -1
    
    age = reference_year - yob
    
    if 40 <= age <= 49:
        return "40–49"
    elif 50 <= age <= 59:
        return "50–59"
    elif 60 <= age <= 69:
        return "60–69"
    elif 70 <= age <= 79:
        return "70–79"
    elif 80 <= age <= 89:
        return "80–89"
    elif 90 <= age <= 99:
        return "90–99"
    else:
        return None  # Outside age range


def compute_cumulative_hazards(df: pd.DataFrame, dose: int, age_band: str = None) -> float:
    """Compute cumulative hazard for a specific dose and age band.
    
    Args:
        df: DataFrame filtered to specific dose
        dose: Dose value (0 or 2)
        age_band: Age band string or None for all ages
        
    Returns:
        Final cumulative hazard value at week 2024-16
    """
    # Filter by dose
    df_dose = df[df['Dose'] == dose].copy()
    
    # Filter by age band if specified
    if age_band is not None:
        if age_band == "All ages (full population)":
            # For "all ages", aggregate across all valid YearOfBirth values
            # Exclude special values like -1, -2 (these are metadata, not actual birth years)
            df_dose = df_dose[df_dose['YearOfBirth'] >= 0].copy()
        else:
            # Map age band to YearOfBirth ranges
            df_dose['age_band'] = df_dose['YearOfBirth'].apply(get_age_band)
            df_dose = df_dose[df_dose['age_band'] == age_band]
    
    if df_dose.empty:
        return np.nan
    
    # Filter to time window: from enrollment week 2021_24 through 2024_16
    # Convert ISOweekDied to comparable format for filtering
    def week_to_int(week_str):
        """Convert ISO week string (e.g., '2021-24') to integer for comparison."""
        try:
            year, week = week_str.split('-')
            return int(year) * 100 + int(week)
        except:
            return 0
    
    df_dose['week_int'] = df_dose['ISOweekDied'].apply(week_to_int)
    enrollment_week_int = 2021 * 100 + 24  # 2021_24
    end_week_int = 2024 * 100 + 16  # 2024_16
    
    # Filter to time window
    df_dose = df_dose[(df_dose['week_int'] >= enrollment_week_int) & 
                      (df_dose['week_int'] <= end_week_int)].copy()
    
    if df_dose.empty:
        return np.nan
    
    # Aggregate Dead and Alive across YearOfBirth values within the age band/dose FIRST
    # This must be done BEFORE computing cumulative hazards
    df_agg = df_dose.groupby('ISOweekDied').agg({
        'Dead': 'sum',
        'Alive': 'sum'
    }).reset_index()
    
    # Sort by ISOweekDied to ensure chronological order
    df_agg = df_agg.sort_values('ISOweekDied').reset_index(drop=True)
    
    # Compute MR from aggregated counts
    df_agg['MR'] = np.where(
        df_agg['Alive'] > 0,
        df_agg['Dead'] / (df_agg['Alive'] + EPS),
        np.nan
    )
    
    # Compute hazard from aggregated MR
    df_agg['hazard'] = hazard_from_mr_improved(df_agg['MR'].values)
    
    # Create time index (week number since enrollment)
    df_agg['t'] = df_agg.index.astype(float)
    
    # Apply DYNAMIC_HVE_SKIP_WEEKS: start accumulation after skip weeks
    df_agg['hazard_eff'] = np.where(
        df_agg['t'] >= float(DYNAMIC_HVE_SKIP_WEEKS),
        df_agg['hazard'],
        0.0
    )
    
    # Compute cumulative hazard
    df_agg['CH'] = df_agg['hazard_eff'].cumsum()
    
    # Find value at week 2024-16 (end of follow-up)
    final_row = df_agg[df_agg['ISOweekDied'] == '2024-16']
    if final_row.empty:
        # If exact week not found, use last available week
        final_row = df_agg.iloc[[-1]]
    
    if final_row.empty:
        return np.nan
    
    return final_row['CH'].iloc[0]


def main():
    """Main function to extract Table 3 values."""
    # Path to KCOR_CMR.xlsx
    xlsx_path = Path(__file__).parent.parent.parent / 'data' / 'Czech' / 'KCOR_CMR.xlsx'
    
    if not xlsx_path.exists():
        print(f"Error: File not found: {xlsx_path}")
        return
    
    print(f"Reading {xlsx_path}...")
    df = pd.read_excel(xlsx_path, sheet_name='2021_24')
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Age bands matching Table 2
    age_bands = [
        "40–49",
        "50–59",
        "60–69",
        "70–79",
        "80–89",
        "90–99",
        "All ages (full population)"
    ]
    
    results = []
    
    for age_band in age_bands:
        print(f"\nProcessing {age_band}...")
        
        # Compute cumulative hazards for Dose 0 and Dose 2
        ch_dose0 = compute_cumulative_hazards(df, dose=0, age_band=age_band)
        ch_dose2 = compute_cumulative_hazards(df, dose=2, age_band=age_band)
        
        # Compute ratio
        if not np.isnan(ch_dose0) and not np.isnan(ch_dose2) and ch_dose2 > 0:
            ratio = ch_dose0 / ch_dose2
        else:
            ratio = np.nan
        
        results.append({
            'age_band': age_band,
            'dose0_ch': ch_dose0,
            'dose2_ch': ch_dose2,
            'ratio': ratio
        })
        
        print(f"  Dose 0 CH: {ch_dose0:.6f}")
        print(f"  Dose 2 CH: {ch_dose2:.6f}")
        print(f"  Ratio: {ratio:.4f}")
    
    # Print table format
    print("\n" + "="*80)
    print("Table 3 Values:")
    print("="*80)
    print("| Age band (years) | Dose 0 cumulative hazard | Dose 2 cumulative hazard | Ratio |")
    print("| ---------------- | ----------------------: | -----------------------: | ----: |")
    
    for r in results:
        age_label = r['age_band']
        ch0 = r['dose0_ch']
        ch2 = r['dose2_ch']
        ratio = r['ratio']
        
        if np.isnan(ch0) or np.isnan(ch2) or np.isnan(ratio):
            print(f"| {age_label:<17} | {'N/A':>23} | {'N/A':>24} | N/A |")
        else:
            print(f"| {age_label:<17} | {ch0:>23.6f} | {ch2:>24.6f} | {ratio:>5.4f} |")
    
    print("\nDone!")


if __name__ == '__main__':
    main()


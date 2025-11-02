# 
#   KCOR_ts.py
#
# Time series analysis: Compute mortality rate per week of cohorts defined relative to the time they got their shot.
# Unlike KCOR_CMR which uses fixed enrollment dates, this groups cohorts by the time they received their dose.
#
# Each dose (1-5) is processed independently as if it is the ONLY dose the person received.
# When processing dose N, ignore all other doses (both earlier and later).
#
# USAGE:
#   cd code; make ts        # Run time series aggregation
#   python KCOR_ts.py <input.csv> <output.xlsx>
#
# Output file:
#   KCOR/data/Czech/KCOR_ts.xlsx (configurable via Makefile)
#
# Output columns:
#   dose, Decade_of_Birth, week_after_dose, alive, dead, h(t)
#

import pandas as pd
import numpy as np
import sys
import os
from datetime import date, timedelta
from typing import Optional

def _read_csv_flex(path: str) -> pd.DataFrame:
    """Read CSV with robust delimiter/encoding handling.
    
    Tries common delimiters and encodings. Returns as strings to preserve ISO week formats.
    """
    # Try fast path first (standard comma, UTF-8/BOM)
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False, encoding='utf-8')
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    for enc in ("utf-8-sig", None, "latin1"):
        # Attempt a few delimiter strategies, preferring exact separators first
        attempts = (
            {"sep": ","},
            {"sep": ";", "engine": "python"},
            {"sep": "\t", "engine": "python"},
            {"sep": None, "engine": "python"},  # sniff
        )
        for opts in attempts:
            try:
                common_kwargs = {"dtype": str, "encoding": enc}
                # low_memory is only valid for the C engine; omit for python engine
                if opts.get("engine") != "python":
                    common_kwargs["low_memory"] = False
                df = pd.read_csv(path, **opts, **common_kwargs)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    # Final fallback: return whatever we could parse (likely 1 column); caller will validate
    return pd.read_csv(path, dtype=str, engine='python', sep=None)

def iso_week_to_monday(iso_week_str: str) -> Optional[date]:
    """Convert ISO week string (YYYY-WW) to Monday date."""
    try:
        return pd.to_datetime(iso_week_str + '-1', format='%G-%V-%u').date()
    except Exception:
        return None

def monday_to_iso_week(monday_date: date) -> str:
    """Convert Monday date to ISO week string (YYYY-WW)."""
    iso = monday_date.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def weeks_after_dose(dose_date: date, target_date: date) -> int:
    """Calculate number of weeks after dose date (0 = week of dose, 1 = next week, etc.)."""
    if target_date < dose_date:
        return -1  # Before dose
    # Get Monday of dose week
    dose_monday = dose_date - timedelta(days=dose_date.weekday())
    # Get Monday of target week
    target_monday = target_date - timedelta(days=target_date.weekday())
    # Calculate weeks difference
    days_diff = (target_monday - dose_monday).days
    return days_diff // 7

def decade_from_birth_year(birth_year: int) -> Optional[int]:
    """Convert birth year to decade (1920-1929 -> 1920, etc.). Returns None for invalid years."""
    if pd.isna(birth_year) or birth_year < 1910 or birth_year > 2005:
        return None
    # Round down to nearest decade
    return (birth_year // 10) * 10

def main():
    import datetime as dt
    print(f"KCOR_ts start: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        excel_out_path = sys.argv[2]
    elif len(sys.argv) >= 2:
        excel_out_path = sys.argv[1]
        input_file = "../data/Czech/records.csv"  # default fallback
    else:
        excel_out_path = "../data/Czech/KCOR_ts.xlsx"  # default fallback
        input_file = "../data/Czech/records.csv"  # default fallback
    
    print(f"  Reading the input file: {input_file}")
    
    # Load the dataset
    a = _read_csv_flex(input_file)
    
    # Validate column count
    expected_col_count = 53
    if a.shape[1] != expected_col_count:
        print(f"ERROR: Input parsed into {a.shape[1]} columns, expected {expected_col_count}.")
        sys.exit(1)
    
    # Rename columns to English (same as KCOR_CMR.py)
    a.columns = [
        'ID', 'Infection', 'Sex', 'YearOfBirth', 'DateOfPositiveTest', 'DateOfResult', 'Recovered', 'Date_COVID_death',
        'Symptom', 'TestType', 'Date_FirstDose', 'Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose',
        'Date_FifthDose', 'Date_SixthDose', 'Date_SeventhDose', 'VaccineCode_FirstDose', 'VaccineCode_SecondDose',
        'VaccineCode_ThirdDose', 'VaccineCode_FourthDose', 'VaccineCode_FifthDose', 'VaccineCode_SixthDose',
        'VaccineCode_SeventhDose', 'PrimaryCauseHospCOVID', 'bin_Hospitalization', 'min_Hospitalization',
        'days_Hospitalization', 'max_Hospitalization', 'bin_ICU', 'min_ICU', 'days_ICU', 'max_ICU', 'bin_StandardWard',
        'min_StandardWard', 'days_StandardWard', 'max_StandardWard', 'bin_Oxygen', 'min_Oxygen', 'days_Oxygen',
        'max_Oxygen', 'bin_HFNO', 'min_HFNO', 'days_HFNO', 'max_HFNO', 'bin_MechanicalVentilation_ECMO',
        'min_MechanicalVentilation_ECMO', 'days_MechanicalVentilation_ECMO', 'max_MechanicalVentilation_ECMO',
        'Mutation', 'DateOfDeath', 'Long_COVID', 'DCCI']
    
    # Filter to single-infection rows (remove duplicates)
    a = a[(a['Infection'].fillna(0).astype(int) <= 1)].copy()
    
    # Extract birth year
    a['birth_year'] = a['YearOfBirth'].str.extract(r'(\d{4})').astype(float)
    a['birth_year'] = a['birth_year'].astype(str).str[:4]
    a['birth_year'] = pd.to_numeric(a['birth_year'], errors='coerce')
    
    # Filter birth years 1910-2005 (to compute decades 1920-1970)
    before = len(a)
    a = a[((a['birth_year'] >= 1910) & (a['birth_year'] <= 2005)) | (a['birth_year'].isna())].copy()
    after = len(a)
    print(f"  Filtered YearOfBirth to 1910-2005: kept {after}/{before} records")
    
    # Parse dose dates (ISO week format: YYYY-WW)
    dose_date_columns = ['Date_FirstDose', 'Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose', 'Date_FifthDose']
    print(f"  Parsing dose date columns...")
    for col in dose_date_columns:
        # Parse ISO week format: YYYY-WW + '-1' -> datetime
        a[col] = pd.to_datetime(a[col] + '-1', format='%G-%V-%u', errors='coerce')
    
    # Parse death date
    a['DateOfDeath'] = pd.to_datetime(a['DateOfDeath'].str.replace(r'[^0-9-]', '', regex=True) + '-1', format='%G-%V-%u', errors='coerce')
    
    # Convert dose dates and death dates to Python date objects for week calculations
    for col in dose_date_columns:
        a[col + '_date'] = a[col].apply(lambda x: x.date() if pd.notna(x) else None)
    a['DateOfDeath_date'] = a['DateOfDeath'].apply(lambda x: x.date() if pd.notna(x) else None)
    
    print(f"  Total records: {len(a)}")
    
    # Process each dose independently, then by vaccination month
    all_results = []
    
    for dose_num in range(1, 5):  # Doses 1-4
        dose_col = f'Date_{["First", "Second", "Third", "Fourth"][dose_num-1]}Dose'
        dose_date_col = dose_col + '_date'
        
        print(f"\n  Processing dose {dose_num}...")
        
        # Filter to people who have this dose
        dose_df = a[a[dose_date_col].notna()].copy()
        print(f"    People with dose {dose_num}: {len(dose_df)}")
        
        if len(dose_df) == 0:
            continue
        
        # Extract vaccination month (1-12) for all people at once
        dose_df['vaccination_month'] = dose_df[dose_date_col].apply(lambda x: x.month if x is not None else None)
        
        # Process by vaccination month to avoid iterating over everyone multiple times
        for month in range(1, 13):  # Months 1-12
            month_df = dose_df[dose_df['vaccination_month'] == month].copy()
            
            if len(month_df) == 0:
                continue
            
            print(f"      Processing month {month}: {len(month_df):,} people")
            
            # Calculate decades for all people in this month (vectorized)
            month_df['decade'] = month_df['birth_year'].apply(decade_from_birth_year)
            
            # Filter to valid decades (1920-1970)
            month_df = month_df[(month_df['decade'].notna()) & 
                                (month_df['decade'] >= 1920) & 
                                (month_df['decade'] <= 1970)].copy()
            
            if len(month_df) == 0:
                continue
            
            # Calculate death_week for all people (vectorized where possible)
            has_death = month_df['DateOfDeath_date'].notna()
            death_weeks = pd.Series(None, index=month_df.index, dtype='float64')
            
            if has_death.any():
                # Calculate weeks after dose for people with deaths
                death_week_values = []
                for idx in month_df[has_death].index:
                    dose_date = month_df.loc[idx, dose_date_col]
                    death_date = month_df.loc[idx, 'DateOfDeath_date']
                    week = weeks_after_dose(dose_date, death_date)
                    if week < 0:
                        week = -1  # Died before dose
                    elif week > 200:
                        week = None  # Died after week 200
                    death_week_values.append(week)
                
                death_weeks[has_death] = death_week_values
            
            month_df['death_week'] = death_weeks
            
            # Filter out people who died before dose
            month_df = month_df[month_df['death_week'] != -1].copy()
            
            if len(month_df) == 0:
                continue
            
            # Process each week 0-200 using vectorized operations
            result_rows = []
            for week in range(0, 201):
                # Vectorized masks: who is alive at start of this week?
                alive_mask = (month_df['death_week'].isna()) | (month_df['death_week'] >= week)
                
                # Vectorized mask: who dies during this week?
                dead_mask = (month_df['death_week'] == week)
                
                # Group by decade and sum the masks
                alive_by_decade = month_df[alive_mask].groupby('decade').size()
                dead_by_decade = month_df[dead_mask].groupby('decade').size()
                
                # Combine results for this week
                for decade in alive_by_decade.index:
                    result_rows.append({
                        'dose': dose_num,
                        'vaccination_month': month,
                        'decade': int(decade),
                        'week_after_dose': week,
                        'alive': int(alive_by_decade[decade]),
                        'dead': int(dead_by_decade.get(decade, 0))
                    })
            
            if result_rows:
                month_result = pd.DataFrame(result_rows)
                all_results.append(month_result)
        
        print(f"    Completed dose {dose_num}")
    
    # Combine all results
    if not all_results:
        print("ERROR: No results generated")
        sys.exit(1)
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Ensure we have all weeks 0-200 for each (dose, vaccination_month, decade) combination
    # Create full grid for all output decades 1920-1970
    decades = list(range(1920, 1980, 10))  # 1920, 1930, ..., 1970
    weeks = list(range(0, 201))
    months = list(range(1, 13))
    
    grid = []
    for dose_num in range(1, 5):
        for month in months:
            for decade in decades:
                for week in weeks:
                    grid.append({
                        'dose': dose_num,
                        'vaccination_month': month,
                        'decade': decade,
                        'week_after_dose': week
                    })
    
    grid_df = pd.DataFrame(grid)
    final_df = grid_df.merge(final_df, on=['dose', 'vaccination_month', 'decade', 'week_after_dose'], how='left')
    final_df = final_df.fillna(0)
    
    # Sort and ensure proper types
    final_df = final_df.sort_values(['dose', 'vaccination_month', 'decade', 'week_after_dose'])
    final_df['alive'] = final_df['alive'].astype(int)
    final_df['dead'] = final_df['dead'].astype(int)
    
    # Calculate h(t) = -(ln(1 - dead/alive))
    # Handle edge cases: alive=0 or dead=0
    final_df['hazard'] = np.where(
        final_df['alive'] > 0,
        -np.log(np.maximum(1e-10, 1 - (final_df['dead'] / final_df['alive']))),
        np.nan
    )
    
    # Rename columns to match spec
    final_df = final_df.rename(columns={
        'dose': 'dose',
        'vaccination_month': 'vaccination_month',
        'decade': 'Decade_of_Birth',
        'week_after_dose': 'week_after_dose',
        'alive': 'alive',
        'dead': 'dead',
        'hazard': 'h(t)'
    })
    
    # Calculate cumulative hazard cum h(t) for each (dose, vaccination_month, Decade_of_Birth) group
    # Sort to ensure proper ordering before cumulative sum
    final_df = final_df.sort_values(['dose', 'vaccination_month', 'Decade_of_Birth', 'week_after_dose'])
    
    # Compute cumulative sum of h(t) within each (dose, vaccination_month, Decade_of_Birth) group
    # Replace NaN with 0 for cumulative calculation (treat missing as 0 hazard)
    # Use transform to preserve index alignment
    final_df['cum h(t)'] = (
        final_df.groupby(['dose', 'vaccination_month', 'Decade_of_Birth'])['h(t)']
        .transform(lambda x: x.fillna(0).cumsum())
    )
    
    # Reorder columns
    final_df = final_df[['dose', 'vaccination_month', 'Decade_of_Birth', 'week_after_dose', 'alive', 'dead', 'h(t)', 'cum h(t)']]
    
    # Filter to output decades 1920-1970
    final_df = final_df[(final_df['Decade_of_Birth'] >= 1920) & (final_df['Decade_of_Birth'] <= 1970)]
    
    # Write to Excel
    print(f"\n  Writing to Excel: {excel_out_path}")
    with pd.ExcelWriter(excel_out_path, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name='TimeSeries', index=False)
    
    print(f"  Total rows written: {len(final_df)}")
    print(f"KCOR_ts complete: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()


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
    if pd.isna(birth_year) or birth_year < 1920 or birth_year > 1979:
        return None
    # Round down to nearest decade
    return (birth_year // 10) * 10

def main():
    import datetime as dt
    import time
    start_time = time.time()
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
    
    # Filter birth years 1920-1979 (to compute decades 1920-1970)
    before = len(a)
    a = a[((a['birth_year'] >= 1920) & (a['birth_year'] <= 1979)) | (a['birth_year'].isna())].copy()
    after = len(a)
    print(f"  Filtered YearOfBirth to 1920-1979: kept {after}/{before} records")
    
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
    all_results_censored = []  # For censored tab
    
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
            
            # Calculate censoring weeks first (for censored tab)
            # Determine censoring week: when people got the next dose (dose_num + 1)
            if dose_num < 4:  # Only doses 1-3 can have next dose
                next_dose_col = f'Date_{["Second", "Third", "Fourth"][dose_num - 1]}Dose_date'
                has_next_dose = month_df[next_dose_col].notna()
                
                # Calculate censoring week for people who got next dose
                censor_weeks = pd.Series(None, index=month_df.index, dtype='float64')
                
                if has_next_dose.any():
                    censor_week_values = []
                    for idx in month_df[has_next_dose].index:
                        dose_date = month_df.loc[idx, dose_date_col]
                        next_dose_date = month_df.loc[idx, next_dose_col]
                        week = weeks_after_dose(dose_date, next_dose_date)
                        if week < 0:
                            week = None  # Next dose before current dose (shouldn't happen)
                        elif week > 200:
                            week = None  # Next dose after week 200, treat as no censoring
                        censor_week_values.append(week)
                    
                    censor_weeks[has_next_dose] = censor_week_values
                
                month_df['censor_week'] = censor_weeks
            else:
                # Dose 4: no next dose, so no censoring
                month_df['censor_week'] = None
            
            # Build both uncensored and censored arrays in a single pass
            num_people = len(month_df)
            alive_arrays = np.zeros((num_people, 201), dtype=np.uint8)
            alive_arrays_c = np.zeros((num_people, 201), dtype=np.uint8)
            dead_arrays_c = np.zeros((num_people, 201), dtype=np.uint8)
            censored_arrays_c = np.zeros((num_people, 201), dtype=np.uint8)
            
            death_weeks = month_df['death_week'].values
            censor_weeks = month_df['censor_week'].values
            
            # Single pass: build both uncensored and censored arrays
            for idx in range(num_people):
                death_w = death_weeks[idx] if not pd.isna(death_weeks[idx]) else None
                censor_w = censor_weeks[idx] if not pd.isna(censor_weeks[idx]) else None
                
                # ===== UNCENSORED VERSION =====
                if death_w is None:
                    # Survived: alive for all 201 weeks
                    alive_arrays[idx, :] = 1
                else:
                    # Died: alive from week 0 to death_week (inclusive)
                    alive_arrays[idx, :int(death_w) + 1] = 1
                
                # ===== CENSORED VERSION =====
                # Determine last week person contributes (minimum of death_week and censor_week)
                if death_w is not None and censor_w is not None:
                    last_week = min(int(death_w), int(censor_w))
                    is_death = (death_w <= censor_w)
                elif death_w is not None:
                    last_week = int(death_w)
                    is_death = True
                elif censor_w is not None:
                    last_week = int(censor_w)
                    is_death = False
                else:
                    last_week = 200  # Survived and not censored
                    is_death = False
                
                # Person is alive from week 0 to last_week (inclusive)
                alive_arrays_c[idx, :last_week + 1] = 1
                
                # If they died (before or at censoring), mark death
                if is_death and death_w is not None:
                    dead_arrays_c[idx, int(death_w)] = 1
                
                # If they were censored (before or at death), mark censoring
                if not is_death and censor_w is not None:
                    censored_arrays_c[idx, int(censor_w)] = 1
            
            # Derive dead_arrays from alive_arrays for uncensored version
            dead_arrays = np.zeros((num_people, 201), dtype=np.uint8)
            dead_arrays[:, 0:200] = alive_arrays[:, 0:200] * (1 - alive_arrays[:, 1:201])
            
            # Get decades as array for grouping
            decades_array = month_df['decade'].values.astype(int)
            
            # Group by decade and sum arrays for both versions
            result_rows = []
            result_rows_c = []
            for decade in np.unique(decades_array):
                decade_mask = (decades_array == decade)
                
                # Uncensored aggregates
                decade_alive = alive_arrays[decade_mask, :].sum(axis=0)
                decade_dead = dead_arrays[decade_mask, :].sum(axis=0)
                
                # Censored aggregates
                decade_alive_c = alive_arrays_c[decade_mask, :].sum(axis=0)
                decade_dead_c = dead_arrays_c[decade_mask, :].sum(axis=0)
                decade_censored_c = censored_arrays_c[decade_mask, :].sum(axis=0)
                
                # Create rows for each week
                for week in range(201):
                    result_rows.append({
                        'dose': dose_num,
                        'vaccination_month': month,
                        'decade': int(decade),
                        'week_after_dose': week,
                        'alive': int(decade_alive[week]),
                        'dead': int(decade_dead[week])
                    })
                    
                    result_rows_c.append({
                        'dose': dose_num,
                        'vaccination_month': month,
                        'decade': int(decade),
                        'week_after_dose': week,
                        'alive': int(decade_alive_c[week]),
                        'dead': int(decade_dead_c[week]),
                        'censored': int(decade_censored_c[week])
                    })
            
            if result_rows:
                month_result = pd.DataFrame(result_rows)
                all_results.append(month_result)
            
            if result_rows_c:
                month_result_c = pd.DataFrame(result_rows_c)
                all_results_censored.append(month_result_c)
        
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
    
    # Rename columns to match spec
    final_df = final_df.rename(columns={
        'dose': 'dose',
        'vaccination_month': 'vaccination_month',
        'decade': 'Decade_of_Birth',
        'week_after_dose': 'week_after_dose',
        'alive': 'alive',
        'dead': 'dead'
    })
    
    # Reorder columns
    final_df = final_df[['dose', 'vaccination_month', 'Decade_of_Birth', 'week_after_dose', 'alive', 'dead']]
    
    # Filter to output decades 1920-1970
    final_df = final_df[(final_df['Decade_of_Birth'] >= 1920) & (final_df['Decade_of_Birth'] <= 1970)]
    
    # Process censored results
    if all_results_censored:
        final_df_censored = pd.concat(all_results_censored, ignore_index=True)
        
        # Create same grid for censored data
        grid_df_censored = pd.DataFrame(grid)
        final_df_censored = grid_df_censored.merge(final_df_censored, on=['dose', 'vaccination_month', 'decade', 'week_after_dose'], how='left')
        final_df_censored = final_df_censored.fillna(0)
        
        # Sort and ensure proper types
        final_df_censored = final_df_censored.sort_values(['dose', 'vaccination_month', 'decade', 'week_after_dose'])
        final_df_censored['alive'] = final_df_censored['alive'].astype(int)
        final_df_censored['dead'] = final_df_censored['dead'].astype(int)
        final_df_censored['censored'] = final_df_censored['censored'].astype(int)
        
        # Rename columns
        final_df_censored = final_df_censored.rename(columns={
            'dose': 'dose',
            'vaccination_month': 'vaccination_month',
            'decade': 'Decade_of_Birth',
            'week_after_dose': 'week_after_dose',
            'alive': 'alive',
            'dead': 'dead',
            'censored': 'censored'
        })
        
        # Reorder columns
        final_df_censored = final_df_censored[['dose', 'vaccination_month', 'Decade_of_Birth', 'week_after_dose', 'alive', 'dead', 'censored']]
        
        # Filter to output decades 1920-1970
        final_df_censored = final_df_censored[(final_df_censored['Decade_of_Birth'] >= 1920) & (final_df_censored['Decade_of_Birth'] <= 1970)]
    else:
        final_df_censored = pd.DataFrame()
    
    # Write to Excel
    print(f"\n  Writing to Excel: {excel_out_path}")
    with pd.ExcelWriter(excel_out_path, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name='TimeSeries', index=False)
        if not final_df_censored.empty:
            final_df_censored.to_excel(writer, sheet_name='Censored', index=False)
    
    print(f"  Total rows written (TimeSeries): {len(final_df)}")
    if not final_df_censored.empty:
        print(f"  Total rows written (Censored): {len(final_df_censored)}")
    
    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    if hours > 0:
        elapsed_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        elapsed_str = f"{minutes}m {seconds}s"
    else:
        elapsed_str = f"{seconds}s"
    
    print(f"KCOR_ts complete: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed time: {elapsed_str}")

if __name__ == '__main__':
    main()


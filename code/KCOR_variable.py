#
#   KCOR_variable.py
#
# Variable-cohort weekly tally by CURRENT dose group.
#
# Reads the Czech population CSV (same as KCOR_CMR.py input), parses death and dose
# weeks, and produces a single-sheet Excel with the same columns as KCOR_CMR output
# (ISOweekDied, DateDied, YearOfBirth, Sex, Dose, Alive, Dead), except this file
# contains variable cohorts: each person contributes to the CURRENT dose group per
# week, transitioning on their dose weeks; deaths are counted in the week of death
# and population is decremented starting the week AFTER death.
#
# Reads the entire CSV.
#
# Usage:
#   python KCOR_variable.py <input.csv> <output.xlsx>
#

import os
import sys
import pandas as pd
import numpy as np
from datetime import date, timedelta
from functools import lru_cache


def to_iso_week_str(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return None
    return dt.strftime('%G-%V')


def sex_to_alpha(sex_val):
    if pd.isna(sex_val) or sex_val == '':
        return 'O'
    s = str(sex_val)
    if s == '1':
        return 'M'
    if s == '2':
        return 'F'
    return 'O'


def week_range_inclusive(start_monday: date, end_monday: date):
    d = start_monday
    while d <= end_monday:
        yield d
        d += timedelta(days=7)


@lru_cache(maxsize=None)
def monday_from_iso_week_str(iso_week: str) -> date:
    # iso_week like '2021-24' (ISO year-week)
    return pd.to_datetime(iso_week + '-1', format='%G-%V-%u').date()


def main():
    # Resolve CLI args and defaults
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        excel_out_path = sys.argv[2]
    elif len(sys.argv) == 2:
        excel_out_path = sys.argv[1]
        input_file = "../../Czech/data/vax_24.csv"
    else:
        excel_out_path = "../data/KCOR_variable.xlsx"
        input_file = "../../Czech/data/vax_24.csv"

    print(f"[variable] Reading input file: {input_file}")
    a = pd.read_csv(
        input_file,
        dtype=str,
        low_memory=False
    )

    # Standardize columns to English (same mapping as KCOR_CMR.py)
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
        'Mutation', 'DateOfDeath', 'Long_COVID', 'DCCI'
    ]

    # Remove duplicate infection records (keep Infection <= 1)
    a = a[(a['Infection'].fillna('0').astype(int) <= 1)]

    # Normalize Sex to M/F/O
    a['Sex'] = a['Sex'].apply(sex_to_alpha)

    # Birth year: extract first 4 digits, convert to int or -1
    a['birth_year'] = a['YearOfBirth'].str.extract(r'(\d{4})')
    a['birth_year'] = pd.to_numeric(a['birth_year'], errors='coerce')
    a['YearOfBirth'] = a['birth_year'].apply(lambda x: int(x) if pd.notnull(x) else -1)

    # Parse death week (LPZ death date field is already ISO week); convert to Monday dates
    a['DateOfDeath'] = pd.to_datetime(
        a['DateOfDeath'].str.replace(r'[^0-9-]', '', regex=True) + '-1',
        format='%G-%V-%u',
        errors='coerce'
    )
    a['WeekOfDeath'] = a['DateOfDeath'].dt.strftime('%G-%V')
    a.loc[a['DateOfDeath'].isna(), 'WeekOfDeath'] = pd.NA

    # Ensure all dose date columns exist
    for col in ['Date_FirstDose', 'Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose', 'Date_FifthDose', 'Date_SixthDose', 'Date_SeventhDose']:
        if col not in a.columns:
            a[col] = pd.NaT

    # Parse dose ISO weeks as Monday Timestamps
    dose_cols = ['Date_FirstDose', 'Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose', 'Date_FifthDose', 'Date_SixthDose', 'Date_SeventhDose']
    for col in dose_cols:
        a[col] = pd.to_datetime(a[col] + '-1', format='%G-%V-%u', errors='coerce')

    # Filter out rows where dose date is after death date (LPZ death precedence)
    a = a[~((a['DateOfDeath'].notnull()) & (a['Date_FirstDose'] > a['DateOfDeath']))]

    # Determine global min and max Monday dates across all events (dose + death)
    all_event_dates = pd.concat([
        a['Date_FirstDose'], a['Date_SecondDose'], a['Date_ThirdDose'], a['Date_FourthDose'],
        a['Date_FifthDose'], a['Date_SixthDose'], a['Date_SeventhDose'], a['DateOfDeath']
    ], ignore_index=True).dropna()

    if all_event_dates.empty:
        print("[variable] No valid dates found in head; writing empty skeleton.")
        with pd.ExcelWriter(excel_out_path, engine='xlsxwriter') as writer:
            empty = pd.DataFrame(columns=['ISOweekDied', 'DateDied', 'YearOfBirth', 'Sex', 'Dose', 'Alive', 'Dead'])
            empty.to_excel(writer, sheet_name='KCOR_variable', index=False)
        print(f"[variable] Wrote empty file: {excel_out_path}")
        return

    start_monday = all_event_dates.min().date()
    end_monday = all_event_dates.max().date()

    # Generate full inclusive list of ISO weeks (as strings) between start and end
    all_weeks = [f"{d.isocalendar().year}-{d.isocalendar().week:02d}" for d in week_range_inclusive(start_monday, end_monday)]
    # Precompute Monday dates for all weeks (populates lru_cache too)
    week_to_date = {w: monday_from_iso_week_str(w) for w in all_weeks}

    # Prepare event delta maps
    # alive_deltas[(week, yob, sex, dose)] => integer delta to apply at beginning of week
    # dead_counts[(week, yob, sex, dose)] => death count for that week (attributed to current dose at death)
    alive_deltas = {}
    dead_counts = {}

    def add_delta(week: str, yob: int, sex: str, dose: int, delta: int):
        key = (week, yob, sex, dose)
        alive_deltas[key] = alive_deltas.get(key, 0) + delta

    def add_dead(week: str, yob: int, sex: str, dose: int, inc: int = 1):
        key = (week, yob, sex, dose)
        dead_counts[key] = dead_counts.get(key, 0) + inc

    # Per-person single pass to build deltas
    print(f"[variable] Building per-person dose transitions and death events...")
    total = len(a)
    print(f"[variable] Records: {total}")
    base_monday = week_to_date[all_weeks[0]]
    start_ts = pd.Timestamp(base_monday)
    for idx, row in enumerate(a.itertuples(index=False), 1):
        if idx % 1000000 == 0:
            from datetime import datetime
            print(f"[variable] Processed {idx:,}/{total:,} rows at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        yob = int(getattr(row, 'YearOfBirth')) if str(getattr(row, 'YearOfBirth')).strip() != '' else -1
        sex_val = getattr(row, 'Sex')
        sex = sex_val if sex_val in ('M', 'F', 'O') else 'O'

        # Collect dose dates
        dose_dates = []
        for dose_num, col in enumerate(dose_cols, start=1):
            dt = getattr(row, col)
            if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
                dose_dates.append((dose_num, dt))
        dose_dates.sort(key=lambda x: x[1])

        # Determine starting dose at global start week
        current_dose = 0
        for dnum, dts in dose_dates:
            if dts <= start_ts:
                current_dose = dnum
            else:
                break

        # +1 at start week for initial presence in current dose
        add_delta(all_weeks[0], yob, sex, current_dose, +1)

        # Apply transitions on their dose weeks (> start week)
        for dnum, dts in dose_dates:
            w = to_iso_week_str(dts)
            if w is None:
                continue
            # Only transition if after start week and within range
            w_monday = week_to_date.get(w) or monday_from_iso_week_str(w)
            if w_monday < base_monday:
                continue
            # Move from previous dose to new dose in that week
            add_delta(w, yob, sex, current_dose, -1)
            add_delta(w, yob, sex, dnum, +1)
            current_dose = dnum

        # Death handling: assign death to current dose at death week; decrement alive next week
        death_ts = getattr(row, 'DateOfDeath') if isinstance(getattr(row, 'DateOfDeath'), pd.Timestamp) else pd.NaT
        if pd.notna(death_ts):
            death_week = to_iso_week_str(death_ts)
            # Dose at death: highest dose <= death_ts
            dose_at_death = 0
            for dnum, dts in dose_dates:
                if dts <= death_ts:
                    dose_at_death = dnum
                else:
                    break
            add_dead(death_week, yob, sex, dose_at_death, 1)

            next_week_monday = (death_ts.date() + timedelta(days=7))
            if next_week_monday <= end_monday:
                next_week = f"{next_week_monday.isocalendar().year}-{next_week_monday.isocalendar().week:02d}"
                add_delta(next_week, yob, sex, dose_at_death, -1)

    # Enumerate axes
    cohorts = sorted(set(int(x) for x in a['YearOfBirth'].unique()))
    sexes = sorted(set(a['Sex'].unique()))
    doses = list(range(0, 8))

    # Build output rows via cumulative application of deltas per (yob, sex, dose)
    print(f"[variable] Aggregating weeks into output table...")
    rows = []
    # Initialize running alive counts
    running_alive = {(y, s, d): 0 for y in cohorts for s in sexes for d in doses}

    for i, week in enumerate(all_weeks, 1):
        week_monday = week_to_date[week]
        for y in cohorts:
            for s in sexes:
                for d in doses:
                    # Apply delta at beginning of week
                    delta = alive_deltas.get((week, y, s, d), 0)
                    if delta:
                        running_alive[(y, s, d)] = running_alive[(y, s, d)] + delta

                    dead = dead_counts.get((week, y, s, d), 0)
                    alive = running_alive[(y, s, d)]

                    rows.append({
                        'ISOweekDied': week,
                        'DateDied': week_monday,  # pure date (no time)
                        'YearOfBirth': y,
                        'Sex': s,
                        'Dose': d,
                        'Alive': int(alive),
                        'Dead': int(dead)
                    })

    out_df = pd.DataFrame(rows)

    # Ensure column order matches KCOR_CMR (DateDied stays as date)
    out_df = out_df[['ISOweekDied', 'DateDied', 'YearOfBirth', 'Sex', 'Dose', 'Alive', 'Dead']]

    # Write single sheet
    os.makedirs(os.path.dirname(excel_out_path), exist_ok=True)
    with pd.ExcelWriter(excel_out_path, engine='xlsxwriter') as writer:
        out_df.to_excel(writer, sheet_name='KCOR_variable', index=False)
        # Apply date format for DateDied column (column B)
        workbook = writer.book
        worksheet = writer.sheets['KCOR_variable']
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        worksheet.set_column('B:B', 12, date_fmt)

    print(f"[variable] Wrote variable-cohort table to {excel_out_path}")


if __name__ == '__main__':
    main()



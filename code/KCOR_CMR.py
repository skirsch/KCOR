# 
#   KCOR_CMR.py
#
# Take the Czech .csv file and output a .xlsx file with the CMR for each dose group by week, birth cohort, sex, and vaccination status.
# 
# So basically this aggregates the Czech data by dose group.
#
# You can then call KCORv4 on the output file to get the KCOR for each dose group by week, birth cohort, sex, and vaccination status.
# 
# This is the CMR aggregation step of the KCOR pipeline.
# It is based on the vax_24.py program which does the same thing but this uses the Nov 2024 data.
# 
# NOTE (KCORv4 correctness vs earlier versions):
#   KCORv4 eliminates duplicate individual records before analysis (e.g., multiple COVID infections per person),
#   so downstream KCOR death counts will be lower than earlier KCOR implementations that double-counted duplicates.
#   The KCORv4 behavior is correct; prior KCOR totals were inflated by counting duplicate entries.
#
#
# I run it from VS Code (execute the buffer). It takes about 10 minutes to run for each enrollment date.
# Be sure you have pandas, numpy, matplotlib, and seaborn installed in your python environment.
# You can install them with pip if needed:   
#  pip install pandas numpy matplotlib seaborn (or apt install python3-pandas python3-numpy python3-matplotlib python3-seaborn on WSL)

# You can also run it from the command line but be sure you have seaborn installed in your python environment.
# You can install seaborn with pip if needed:   pip install seaborn (or apt install python3-seaborn on WSL)

# USAGE:
#   cd code; make CMR        # Run only CMR aggregation
#   cd code; make KCOR       # Run complete pipeline (CMR + KCORv4 analysis)
#   
#   Or run directly:
#   python KCOR_CMR.py <input.csv> <output.xlsx>

# Output file:
#   KCOR/data/Czech/KCOR_CMR.xlsx (configurable via Makefile)
#
# To analyze the output file, use make KCOR (runs complete pipeline)
#
# The output file contains the CMR for each dose group by week, birth cohort, sex, and vaccination status.
# The output file contains multiple sheets, one for each enrollment date.
# The data is structured with Dose as an index column and simplified Alive/Dead value columns.
# Each row represents a unique combination of week, birth cohort, sex, and dose group.
# The output file columns are:
#   ISOweekDied: ISO week (YYYY-WW format, e.g., 2020-10)
#   DateDied: Monday date of the ISO week (YYYY-MM-DD format, e.g., 2020-03-02)
#   YearOfBirth: birth year (e.g., "1970") or "ASMR" for age-standardized rows and "UNK" for unknown birth year
#   Sex: alphabetic code (M=Male, F=Female, O=Other/Unknown)
#   Dose: dose group (0=unvaccinated, 1=1 dose, 2=2 doses, 3=3 doses, 4=4 doses, 5=5 doses, 6=6 doses, 7=7 doses)
#   Alive: population count alive in this dose group
#   Dead: death count in this dose group (age-standardized for ASMR rows with YearOfBirth="ASMR")
#
# The population counts are adjusted for deaths over time (attrition).
#
# The data is then imported into this spreadsheet for analysis.
#   Czech/analysis/fixed_cohort_cmr_dosegroups_analysis.xlsx
#
# This script processes vaccination and death data to compute CMR (Crude Mortality Rate) for each age cohort and vaccination dose group (0, 1, 2, 3, 4, 5, 6, 7). 
# It loads the Czech dataset, processes it to extract relevant information, computes weekly death counts for vaccinated and unvaccinated individuals, and calculates CMR per 100,000 population per year.
# Computes ages for birth year cohorts from 1900 to 2020 using Czech demographic standardization.
# 
# It also shows deaths by birth cohort and vaccination status over time, allowing for analysis of mortality trends in relation to vaccination status.
#
# This creates output files that are analyzed in files in analysis/fixed_cohort_CMR.... files.
#
# This is the main KCOR analysis script. It generates output to allow computation 
# of CMR (Crude Mortality Rate) for dose 0, 1, 2, 3, 4, 5, 6, and 7 by outputting alive and dead counts by week, birth cohort, and vaccination status.
#
# You can now compute the instantaneous CMR for each dose group by week, birth cohort, and vaccination status.
# 
# You can also compute the HR (Hazard Ratio) for each dose group by week, birth cohort, and vaccination status
# by dividing the CMR of the vaccinated group by the CMR of the unvaccinated group.
# More importantly, can compute the HR for dose 2 vs. dose 1, dose 3 vs. dose 2, and dose 4 vs. dose 3, etc.
# 
# Comparing HRs between dose groups (dose 1 or more) can provide insights into the relative mortality risk associated with each vaccination dose.
# and it eliminates any HVE bias since it doesn't compare vaccinated to unvaccinated.
#
# # This script is designed to analyze mortality trends in relation to vaccination status and there is a list of
# enrollment dates for dose groups.
# 
# It is a replacement for the old KCOR.py, providing enhanced analysis capabilities with population and death data.
# enabling the the analysis of mortality trends in relation to vaccination status.
# It uses the same data format as the old KCOR.py.
# it does not require the old KCOR.py script to run, but it uses the same data format.
# It only looks at first dose vaccination data and ACM death dates.
#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Czech Reference Population for ASMR calculation (by 5-year birth cohorts)
# Source: Czech demographic data
CZECH_REFERENCE_POP = {
    1900: 13,       # 1900-1904
    1905: 23,       # 1905-1909
    1910: 32,       # 1910-1914
    1915: 45,       # 1915-1919
    1920: 1068,     # 1920-1924
    1925: 9202,     # 1925-1929
    1930: 35006,    # 1930-1934
    1935: 72997,    # 1935-1939
    1940: 150323,   # 1940-1944
    1945: 246393,   # 1945-1949
    1950: 297251,   # 1950-1954
    1955: 299766,   # 1955-1959
    1960: 313501,   # 1960-1964
    1965: 335185,   # 1965-1969
    1970: 415319,   # 1970-1974
    1975: 456701,   # 1975-1979
    1980: 375605,   # 1980-1984
    1985: 357674,   # 1985-1989
    1990: 338424,   # 1990-1994
    1995: 256900,   # 1995-1999
    2000: 251049,   # 2000-2004
    2005: 287094,   # 2005-2009
    2010: 275837,   # 2010-2014
    2015: 238952,   # 2015-2019
    2020: 84722,    # 2020-2024
}

# Age bins for ASMR (Age-Standardized Mortality Rate) computation
# These bins are used to group individuals by age for standardization against Czech reference population
# Aligned with 5-year cohorts to match Czech demographic data (birth years 1900-2020)
AGE_BINS = [
    ("0-4",   0,   4),    # ages 0-4 years (aligned with 5-year cohorts)
    ("5-9",   5,   9),
    ("10-14", 10, 14),
    ("15-19", 15, 19),
    ("20-24", 20, 24),
    ("25-29", 25, 29),
    ("30-34", 30, 34),
    ("35-39", 35, 39),
    ("40-44", 40, 44),
    ("45-49", 45, 49),
    ("50-54", 50, 54),
    ("55-59", 55, 59),
    ("60-64", 60, 64),
    ("65-69", 65, 69),
    ("70-74", 70, 74),
    ("75-79", 75, 79),
    ("80-84", 80, 84),
    ("85-89", 85, 89),
    ("90-94", 90, 94),
    ("95-99", 95, 99),
    ("100-104", 100, 104),
    ("105-109", 105, 109),
    ("110-114", 110, 114),
    ("115-119", 115, 119),
    ("120-124", 120, 124),
    ("125+", 125, 200),   # cap at 200 for very old ages
]

def age_to_group(age_years: int) -> str:
    """Map integer age to age group label for ASMR calculation.
    
    Used in ASMR (Age-Standardized Mortality Rate) computation to group individuals
    by age before applying Czech reference population standardization.
    """
    if pd.isna(age_years):
        return None
    age = int(age_years)
    for label, a0, a1 in AGE_BINS:
        if a0 <= age <= a1:
            return label
    return None

def approx_age_from_born(week_date, born_year) -> int:
    """
    Approx age in whole years at week_date when only a birth YEAR is known.
    Uses July 1st as mid-year proxy.
    """
    if pd.isna(born_year) or born_year == -1:
        return None
    from datetime import date
    # Convert week_date to date if it's a pandas Timestamp
    if hasattr(week_date, 'date'):
        week_d = week_date.date()
    else:
        week_d = pd.to_datetime(week_date).date()
    dob_proxy = date(int(born_year), 7, 1)
    return max(0, int((week_d - dob_proxy).days // 365.2425))

def birth_year_to_age_at_week(birth_year: int, week_date) -> int:
    """Calculate age at specific week date from birth year."""
    if pd.isna(birth_year) or birth_year == -1:
        return None
    return approx_age_from_born(week_date, birth_year)

def get_czech_reference_pop_for_birth_year(birth_year: int) -> int:
    """Get Czech reference population count for a specific birth year by mapping to 5-year cohort."""
    if pd.isna(birth_year) or birth_year < 1900:
        return 0
    
    # Map birth year to 5-year cohort (1900-1904 -> 1900, 1905-1909 -> 1905, etc.)
    cohort_year = ((birth_year - 1900) // 5) * 5 + 1900
    return CZECH_REFERENCE_POP.get(cohort_year, 0)

# define the output Excel file path
# This will contain the CMR for each dose group by week, birth cohort, and vaccination status.
# This is used to compute the HR (Hazard Ratio) for each dose group by week, birth cohort, and vaccination status
# and it eliminates any HVE bias since it doesn't compare vaccinated to unvaccinated.
if len(sys.argv) >= 3:
    input_file = sys.argv[1]
    excel_out_path = sys.argv[2]
elif len(sys.argv) >= 2:
    excel_out_path = sys.argv[1]
    input_file = "../../Czech/data/vax_24.csv"  # default fallback
else:
    excel_out_path = "../data/KCOR_output.xlsx"  # default fallback
    input_file = "../../Czech/data/vax_24.csv"  # default fallback
excel_writer = pd.ExcelWriter(excel_out_path, engine='xlsxwriter')


# Define enrollment dates for dose groups
# These dates are used to determine the dose group for each individual based on their vaccination dates.
# These are ISO week format: YYYY-WW
# The enrollment date is the date when the individual is considered to be part of the study cohort.
# 2021-W13 is 03-29-2021, near the start of the vaccination campaign so we can capture impact on older people.
# 2021-W24 is 06-14-2021, when everyone 40+ was eligible for first dose.
# 2021-W41 is 10-11-2021, which is a late enrollment date before the winter wave; not super useful.
# 2022-W06 is 02-07-2022, which is the best booster #1 enrollment since it is just after everyone got 1st booster.
# 2022-W47 is 11-21-2022, which is the best booster #2 enrollment since it is just after everyone got 2nd booster.
# 2024-W01 is 12-30-2023, which is the best booster #3 enrollment since it is just after everyone got 3rd booster, but too late to be useful
# because the deaths start declining in Q2 of 2024
enrollment_dates = ['2021-13', '2021-24', '2022-06', '2022-47']  # Full set of enrollment dates

# Optional override via environment variable ENROLLMENT_DATES (comma-separated, e.g., "2021-24" or "2021-13,2021-24")
_env_dates = os.environ.get('ENROLLMENT_DATES')
if _env_dates:
    _parsed = [d.strip().replace('_', '-') for d in _env_dates.split(',') if d.strip()]
    if _parsed:
        enrollment_dates = _parsed

# enrollment_dates = ['2021-24']  # For testing, just do one enrollment date

## Load the dataset with explicit types and rename columns to English
import datetime as _dt
print(f"KCOR_CMR start: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Reading the input file: {input_file}")


a = pd.read_csv(
    input_file,
    dtype=str,  # preserve ISO week format and avoid type inference
    low_memory=False
)

# rename the columns in English (same as KCOR.py)
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

# Ensure DCCI is mapped into 4 buckets: 0, 1, 3 (catch-all), 5
a['DCCI'] = pd.to_numeric(a['DCCI'], errors='coerce')
a['DCCI'] = a['DCCI'].clip(lower=0, upper=5)
a['DCCI'] = a['DCCI'].where(a['DCCI'].isin([0, 1, 5]), 3).astype('Int8')

# Immediately drop columns we won't use to reduce memory
needed_cols = [
    'Infection', 'Sex', 'YearOfBirth',
    'Date_FirstDose', 'Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose',
    'DateOfDeath', 'DCCI'
]
a = a[needed_cols]

# if you got infected more than once, it will create a duplicate record (with a different ID) so
# remove those records so we don't double count the deaths.

# Remove records where Infection > 1
a = a[(a['Infection'].fillna(0).astype(int) <= 1)]

# Convert Sex to alphabetic codes: M, F, O
def sex_to_alpha(sex_val):
    if pd.isna(sex_val) or sex_val == '':
        return 'O'  # Other/Unknown
    elif str(sex_val) == '1':
        return 'M'  # Male
    elif str(sex_val) == '2':
        return 'F'  # Female
    else:
        return 'O'  # Other/Unknown

a['Sex'] = a['Sex'].apply(sex_to_alpha)

# Debug: Check data quality after Sex conversion
print(f"Records after Sex conversion: {len(a)}")
print(f"Sex distribution: {a['Sex'].value_counts()}")



# Convert relevant columns to datetime (ISO format assumed: YYYY-MM-DD)
# Extract cohort year from birth year range (e.g., '1970-1974' -> 1970)
a['birth_year'] = a['YearOfBirth'].str.extract(r'(\d{4})').astype(float)
# Limit to cohorts born 1900-2020
# This will also convert NaN birth years to NaN, which we can handle later
## Remove birth year filtering so all birthdates, including blanks, are included

# Parse ISO week format for death date only (first_dose_date will be parsed later)
a['DateOfDeath'] = pd.to_datetime(a['DateOfDeath'].str.replace(r'[^0-9-]', '', regex=True) + '-1', format='%G-%V-%u', errors='coerce')
# Keep WeekOfDeath in original ISO week format (YYYY-WW) for exact matching
a['WeekOfDeath'] = a['DateOfDeath'].dt.strftime('%G-%V')
# Set WeekOfDeath to NaN for invalid death dates
a.loc[a['DateOfDeath'].isna(), 'WeekOfDeath'] = pd.NA

# Debug: Check death data quality
print(f"Total records: {len(a)}")
print(f"Records with deaths: {a['DateOfDeath'].notnull().sum()}")
print(f"Records with valid WeekOfDeath: {a['WeekOfDeath'].notna().sum()}")

# Extract year from birth_year string (first 4 chars)
a['birth_year'] = a['birth_year'].astype(str).str[:4]
a['birth_year'] = pd.to_numeric(a['birth_year'], errors='coerce')

# --------- Parse all dose dates ONCE before the enrollment loop ---------
print(f"Parsing dose date columns (one time only)...")
# Add dose date columns if not already present (up to 4th dose only)
for col in ['Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose']:
    if col not in a.columns:
        a[col] = pd.NaT

# Use the fast vectorized ISO week parsing approach (up to 4th dose only)
dose_date_columns = ['Date_FirstDose', 'Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose']
for col in dose_date_columns:
    print(f"  Parsing {col}...")
    # Fast vectorized ISO week parsing: YYYY-WW + '-1' -> datetime (keep as Timestamp, not .date)
    a[col] = pd.to_datetime(a[col] + '-1', format='%G-%V-%u', errors='coerce')
print(f"Dose date parsing complete for all columns.")

# Fix death dates - now we can compare properly since all dates are pandas Timestamps
## Only use LPZ death date, ignore other death date
a = a[~((a['DateOfDeath'].notnull()) & (a['Date_FirstDose'] > a['DateOfDeath']))]

# Convert birth years to integers once (outside the enrollment loop)
print(f"Converting birth years to integers (one time only)...")
a['YearOfBirth'] = a['birth_year'].apply(lambda x: int(x) if pd.notnull(x) else -1)
print(f"Birth year conversion complete.")

# Collapse birth years to buckets: <=1920 → 1920, >=2000 → 2000, keep -1 as unknown
known_birth_mask = a['YearOfBirth'] != -1
a.loc[known_birth_mask, 'YearOfBirth'] = a.loc[known_birth_mask, 'YearOfBirth'].clip(lower=1920, upper=2000)

# --------- NEW: Dose group analysis for multiple enrollment dates ---------

### YOU CAN RESTART HERE if code bombs out. This saves time.

# Dose date columns
dose_date_cols = [
    (0, None),
    (1, 'Date_FirstDose'),
    (2, 'Date_SecondDose'),
    (3, 'Date_ThirdDose'),
    (4, 'Date_FourthDose'),  # 4 means 4+
]

for enroll_date_str in enrollment_dates:
    # Parse ISO week string as Monday of that week
    import datetime
    print(f"Processing enrollment date {enroll_date_str} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    enrollment_date = pd.to_datetime(enroll_date_str + '-1', format='%G-%V-%u', errors='coerce')
    enroll_week_str = enrollment_date.strftime('%G-%V')
    print(f"  Creating copy of dataset ({len(a)} records)...")
    a_copy = a.copy()
    print(f"  Keeping all records including deaths before enrollment date...")
    # Keep all individuals, including those who died before the enrollment date
    # This allows us to see pre-enrollment deaths in their correct dose groups
    print(f"  Records in analysis: {len(a_copy)}")
    
    # Assign dose group as of enrollment date (highest dose <= enrollment date) - VECTORIZED VERSION
    # For people who died before enrollment, use their death date instead of enrollment date
    print(f"  Assigning dose groups...")
    
    # For each person, determine the reference date for dose group assignment
    # If they died before enrollment, use death date; otherwise use enrollment date
    reference_dates = a_copy['DateOfDeath'].where(
        a_copy['DateOfDeath'].notna() & (a_copy['DateOfDeath'] < enrollment_date),
        enrollment_date
    )
    
    # Create boolean masks for each dose being valid (not null and <= reference_date)
    dose1_valid = a_copy['Date_FirstDose'].notna() & (a_copy['Date_FirstDose'] <= reference_dates)
    dose2_valid = a_copy['Date_SecondDose'].notna() & (a_copy['Date_SecondDose'] <= reference_dates)
    dose3_valid = a_copy['Date_ThirdDose'].notna() & (a_copy['Date_ThirdDose'] <= reference_dates)
    dose4_valid = a_copy['Date_FourthDose'].notna() & (a_copy['Date_FourthDose'] <= reference_dates)
    
    # Start with dose group 0 for everyone
    a_copy['dose_group'] = 0
    
    # Assign higher dose groups based on valid doses (order matters!)
    a_copy.loc[dose1_valid, 'dose_group'] = 1
    a_copy.loc[dose2_valid, 'dose_group'] = 2  
    a_copy.loc[dose3_valid, 'dose_group'] = 3
    a_copy.loc[dose4_valid, 'dose_group'] = 4
    # 4+ doses collapse to 4
    print(f"  Dose group assignment complete.")
    
    dose_groups = [0, 1, 2, 3, 4]
    # Compute population base: count of people in each (born, sex, dose_group)
    print(f"  Computing population base (alive at enrollment)...")
    # Freeze dose at enrollment for population base and exclude those who died before enrollment
    alive_at_enroll = a_copy[(a_copy['DateOfDeath'].isna()) | (a_copy['DateOfDeath'] >= enrollment_date)].copy()
    pop_base = alive_at_enroll.groupby(['YearOfBirth', 'Sex', 'DCCI', 'dose_group']).size().reset_index(name='pop')
    print(f"    Total population across all dose groups: {pop_base['pop'].sum()}")
    # Debug: print dose distribution at enrollment (collapsed to 0..4)
    dose_dist = pop_base.groupby('dose_group')['pop'].sum().sort_index()
    print(f"    Dose distribution at enrollment: " + ", ".join([f"{int(k)}={int(v)}" for k, v in dose_dist.items()]))
    
    # Compute deaths per week using two classifications:
    #  - Pre-enrollment weeks: deaths grouped by dose at time of death (variable cohorts)
    #  - Post-enrollment weeks: deaths grouped by frozen enrollment dose (fixed cohorts)
    print(f"  Computing deaths per week (variable pre-enrollment, fixed post-enrollment)...")
    # Dose at death (highest dose date <= DateOfDeath)
    death_ref = a_copy['DateOfDeath']
    d1_at = a_copy['Date_FirstDose'].notna() & (a_copy['Date_FirstDose'] <= death_ref)
    d2_at = a_copy['Date_SecondDose'].notna() & (a_copy['Date_SecondDose'] <= death_ref)
    d3_at = a_copy['Date_ThirdDose'].notna() & (a_copy['Date_ThirdDose'] <= death_ref)
    d4_at = a_copy['Date_FourthDose'].notna() & (a_copy['Date_FourthDose'] <= death_ref)
    a_copy['dose_at_death'] = 0
    a_copy.loc[d1_at, 'dose_at_death'] = 1
    a_copy.loc[d2_at, 'dose_at_death'] = 2
    a_copy.loc[d3_at, 'dose_at_death'] = 3
    a_copy.loc[d4_at, 'dose_at_death'] = 4  # 4 = 4+
    # Pre-enrollment deaths by dose-at-death
    mask_deaths = a_copy['DateOfDeath'].notnull() & a_copy['WeekOfDeath'].notna()
    deaths_pre = (
        a_copy[mask_deaths]
            .groupby(['WeekOfDeath', 'YearOfBirth', 'Sex', 'DCCI', 'dose_at_death'])
            .size()
            .reset_index(name='dead_pre')
    )
    # Post-enrollment deaths by frozen enrollment dose
    deaths_post = (
        a_copy[mask_deaths]
            .groupby(['WeekOfDeath', 'YearOfBirth', 'Sex', 'DCCI', 'dose_group'])
            .size()
            .reset_index(name='dead_post')
    )
    print(f"    Total deaths across all dose groups: {int(deaths_post['dead_post'].sum())}")
    print(f"    Unique weeks with deaths: {len(deaths['WeekOfDeath'].unique())}")
    # Get all weeks in the study period (from database start to end, including pre-enrollment period)
    # Use all vaccination and death dates to get the full week range
    print(f"  Computing week range for entire study period (including pre-enrollment)...")
    all_dates = pd.concat([
        a_copy['Date_FirstDose'],
        a_copy['Date_SecondDose'],
        a_copy['Date_ThirdDose'],
        a_copy['Date_FourthDose'],
        a_copy['DateOfDeath']
    ]).dropna()
    min_week = all_dates.min().isocalendar().week
    min_year = all_dates.min().isocalendar().year
    max_week = all_dates.max().isocalendar().week
    max_year = all_dates.max().isocalendar().year
    print(f"    Full week range: {min_year}-{min_week:02d} to {max_year}-{max_week:02d} (includes pre-enrollment period)")
    # Build all weeks between min and max
    from datetime import date, timedelta
    def week_year_iter(y1, w1, y2, w2):
        d = date.fromisocalendar(y1, w1, 1)
        dend = date.fromisocalendar(y2, w2, 1)
        while d <= dend:
            yield d.isocalendar()[:2]
            # next week
            d += timedelta(days=7)
    all_weeks = [f"{y}-{w:02d}" for y, w in week_year_iter(min_year, min_week, max_year, max_week)]
    print(f"  Preparing output structure (vectorized)...")
    # Build the minimal observed combinations for (YearOfBirth, Sex, DCCI, Dose)
    observed_pop = pop_base.rename(columns={'dose_group': 'Dose'})[['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'pop']]
    observed_deaths_pre = deaths_pre.rename(columns={'dose_at_death': 'Dose', 'WeekOfDeath': 'ISOweekDied'})[
        ['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI', 'Dose', 'dead_pre']
    ]
    observed_deaths_post = deaths_post.rename(columns={'dose_group': 'Dose', 'WeekOfDeath': 'ISOweekDied'})[
        ['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI', 'Dose', 'dead_post']
    ]

    # Cross join weeks with observed population combos
    weeks_df = pd.DataFrame({'ISOweekDied': all_weeks})
    combos_df = observed_pop[['YearOfBirth', 'Sex', 'DCCI', 'Dose']].drop_duplicates()
    combos_df['__k'] = 1
    weeks_df['__k'] = 1
    out = weeks_df.merge(combos_df, on='__k', how='left').drop(columns='__k')

    # Attach population to every week-row for the combo
    out = out.merge(observed_pop, on=['YearOfBirth', 'Sex', 'DCCI', 'Dose'], how='left')
    out.rename(columns={'pop': 'Alive'}, inplace=True)
    out['Alive'] = out['Alive'].fillna(0).astype(int)

    # Attach both pre- and post-enrollment death counts; we'll select per-week after we build DateDied
    out = out.merge(observed_deaths_pre, on=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI', 'Dose'], how='left')
    out = out.merge(observed_deaths_post, on=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI', 'Dose'], how='left')
    out[['dead_pre', 'dead_post']] = out[['dead_pre', 'dead_post']].fillna(0).astype(int)

    out['ISOweekDied'] = out['ISOweekDied'].astype(str)
    # Optimize dtypes for speed and memory
    out['Sex'] = out['Sex'].astype('category')
    out['Dose'] = out['Dose'].astype('int8')
    out['DCCI'] = out['DCCI'].astype('Int8')
    
    # Add a readable date column (Monday of the ISO week) right after ISOweekDied
    out['DateDied'] = out['ISOweekDied'].apply(lambda week: pd.to_datetime(week + '-1', format='%G-%V-%u').strftime('%Y-%m-%d'))
    
    # Reorder columns to put DateDied right after ISOweekDied
    cols = ['ISOweekDied', 'DateDied', 'YearOfBirth', 'Sex', 'DCCI', 'Dose', 'Alive', 'Dead']
    out = out[cols]

    # Overwrite population columns to reflect attrition from deaths (vectorized)
    print(f"  Computing population attrition from deaths (vectorized)...")
    out = out.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'ISOweekDied'])
    # Compute Monday date for comparison
    out['DateDied'] = out['ISOweekDied'].apply(lambda week: pd.to_datetime(week + '-1', format='%G-%V-%u'))
    post_mask = out['DateDied'] >= enrollment_date

    # -------- Pre-enrollment Alive (variable cohorts at start of week) --------
    # Base total per (YearOfBirth, Sex, DCCI)
    base_total = a_copy.groupby(['YearOfBirth', 'Sex', 'DCCI']).size().rename('base_total').reset_index()
    out = out.merge(base_total, on=['YearOfBirth', 'Sex', 'DCCI'], how='left')
    out['base_total'] = out['base_total'].fillna(0).astype(int)

    # Weekly dose transitions counts per combo (to doses 1..4)
    dose_week_cols = [
        ('trans_1', 'Date_FirstDose'),
        ('trans_2', 'Date_SecondDose'),
        ('trans_3', 'Date_ThirdDose'),
        ('trans_4', 'Date_FourthDose'),
    ]
    trans_frames = []
    for label, col in dose_week_cols:
        wk = a_copy[col].dt.strftime('%G-%V')
        t = a_copy.loc[wk.notna(), ['YearOfBirth', 'Sex', 'DCCI']].copy()
        t['ISOweekDied'] = wk[wk.notna()].values
        t[label] = 1
        t = t.groupby(['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'])[label].sum().reset_index()
        trans_frames.append(t)
    if len(trans_frames) > 0:
        transitions = trans_frames[0]
        for tf in trans_frames[1:]:
            transitions = transitions.merge(tf, on=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'], how='outer')
    else:
        transitions = pd.DataFrame(columns=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'])
    for col_ in ['trans_1', 'trans_2', 'trans_3', 'trans_4']:
        if col_ not in transitions.columns:
            transitions[col_] = 0
    transitions = transitions.fillna(0)

    # Merge transitions and compute cumulative sums per combo, then shift to get prior-week totals
    out = out.merge(transitions, on=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'], how='left')
    out[['trans_1', 'trans_2', 'trans_3', 'trans_4']] = out[['trans_1', 'trans_2', 'trans_3', 'trans_4']].fillna(0).astype(int)
    cumT = (
        out.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'ISOweekDied'])
           .groupby(['YearOfBirth', 'Sex', 'DCCI'])[['trans_1', 'trans_2', 'trans_3', 'trans_4']]
           .cumsum()
           .shift(fill_value=0)
    )
    cumT.columns = ['cumT1_prev', 'cumT2_prev', 'cumT3_prev', 'cumT4_prev']
    out = pd.concat([out, cumT], axis=1)

    # Build Dead using pre/post series
    out['Dead'] = np.where(out['DateDied'] < enrollment_date, out['dead_pre'], out['dead_post']).astype(int)
    # Cumulative prior-week deaths per combo+dose
    out['cumD_prev'] = (
        out.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'ISOweekDied'])
           .groupby(['YearOfBirth', 'Sex', 'DCCI', 'Dose'])['Dead']
           .cumsum()
           .shift(fill_value=0)
    )

    # Compute Alive_pre at start of week for variable cohorts using dose-specific formula
    dose_vals = out['Dose'].values
    alive_pre = np.where(dose_vals == 0, out['base_total'] - out['cumT1_prev'] - out['cumD_prev'],
                 np.where(dose_vals == 1, out['cumT1_prev'] - out['cumT2_prev'] - out['cumD_prev'],
                 np.where(dose_vals == 2, out['cumT2_prev'] - out['cumT3_prev'] - out['cumD_prev'],
                 np.where(dose_vals == 3, out['cumT3_prev'] - out['cumT4_prev'] - out['cumD_prev'],
                                      out['cumT4_prev'] - out['cumD_prev']))))
    alive_pre = np.maximum(alive_pre, 0)
    # Fill pre-enrollment Alive with alive_pre
    out.loc[~post_mask, 'Alive'] = alive_pre[~post_mask]

    # Attrition only on post-enrollment rows (fixed cohorts)
    out.loc[post_mask, 'cum_dead_prev'] = out.loc[post_mask].groupby(['YearOfBirth', 'Sex', 'DCCI', 'Dose'], observed=True)['Dead'].cumsum().shift(fill_value=0)
    out.loc[post_mask, 'Alive'] = (out.loc[post_mask, 'Alive'] - out.loc[post_mask, 'cum_dead_prev']).clip(lower=0)
    # Drop helper columns
    out.drop(columns=[c for c in ['dead_pre','dead_post','cum_dead_prev','base_total','trans_1','trans_2','trans_3','trans_4','cumT1_prev','cumT2_prev','cumT3_prev','cumT4_prev','cumD_prev'] if c in out.columns], inplace=True)

    # Convert DateDied back to string for Excel after computations
    out['DateDied'] = out['DateDied'].dt.strftime('%Y-%m-%d')

    # Sanity check: Alive at enrollment week should match pop_base per Dose
    try:
        for d in dose_groups:
            sum_alive = int(out[(out['ISOweekDied'] == enroll_week_str) & (out['Dose'] == d)]['Alive'].sum())
            pop_d = int(pop_base[pop_base['dose_group'] == d]['pop'].sum())
            if sum_alive != pop_d:
                print(f"CAUTION: Enrollment-week Alive sum mismatch for Dose {d} at {enroll_week_str}: Alive={sum_alive} vs pop_base={pop_d}")
    except Exception as e:
        print(f"CAUTION: Enrollment-week Alive vs pop_base check failed: {e}")
    
    # ASMR calculation - BYPASSED for now (should be in analysis code, not summary code)
    # This computation is bypassed to keep the output file simple and avoid mixed data types
    print(f"  ASMR computation bypassed - keeping output file simple...")
    
    # (ASMR code remains here but is not executed)
    # ASMR calculation - scale death counts DOWN based on standard population
    # print(f"  Computing ASMR (Age-Standardized Mortality Rates)...")
    # asmr_rows = []
    # 
    # for week in all_weeks:
    #     week_data = out[out['ISOweekDied'] == week].copy()
    #     # Filter to reasonable birth years only (1900-2020, those in Czech reference population)
    #     current_year = int(week[:4])
    #     week_data = week_data[week_data['YearOfBirth'].apply(lambda x: str(x).isdigit() and 1900 <= int(x) <= 2020)]
    #     if len(week_data) == 0:
    #         continue
    #     # Calculate age for each birth cohort in this week
    #     week_date = pd.to_datetime(week + '-1', format='%G-%V-%u')  # Convert ISO week back to Monday date
    #     week_data['age'] = week_data['YearOfBirth'].apply(lambda YearOfBirth: approx_age_from_born(week_date, int(YearOfBirth)) if str(YearOfBirth).isdigit() else None)
    #     week_data['age_group'] = week_data['age'].apply(age_to_group)
    #     # Remove rows with invalid age groups
    #     week_data = week_data[week_data['age_group'].notna()]
    #     if len(week_data) == 0:
    #         continue
    #     
    #     # Calculate the full Czech reference population total
    #     full_standard_pop = sum(CZECH_REFERENCE_POP.values())  # Czech reference population total
    #     
    #     for d in dose_groups:
    #         dose_data = week_data[week_data['Dose'] == d]
    #         if len(dose_data) == 0:
    #             continue
    #         
    #         # Group by sex and sum across age groups for this dose
    #         sex_summary = dose_data.groupby('Sex').agg({
    #             'Alive': 'sum',
    #             'Dead': 'sum'
    #         }).reset_index()
    #         
    #         for _, sex_row in sex_summary.iterrows():
    #             sex_val = sex_row['Sex']
    #             total_alive = sex_row['Alive']
    #             total_dead = sex_row['Dead']
    #             
    #             total_scaled_deaths = 0
    #             
    #             # Now calculate birth-year-standardized deaths for this sex/dose combination
    #             dose_sex_data = dose_data[dose_data['Sex'] == sex_val]
    #             birth_year_summary = dose_sex_data.groupby('YearOfBirth').agg({
    #                 'Alive': 'sum',
    #                 'Dead': 'sum'
    #             }).reset_index()
    #             
    #             for _, row in birth_year_summary.iterrows():
    #                 birth_year_str = row['YearOfBirth']
    #                 actual_pop = row['Alive']
    #                 actual_deaths = row['Dead']
    #                 
    #                 # Convert birth year string to int and map to 5-year cohort
    #                 if str(birth_year_str).isdigit():
    #                     birth_year = int(birth_year_str)
    #                     # Map to 5-year cohort (1900-1904 -> 1900, 1905-1909 -> 1905, etc.)
    #                     cohort_year = ((birth_year - 1900) // 5) * 5 + 1900
    #                     
    #                     if cohort_year in CZECH_REFERENCE_POP and actual_pop > 0:
    #                         reference_pop = CZECH_REFERENCE_POP[cohort_year]
    #                         
    #                         # Scale to match reference population (up or down as needed)
    #                         scale_factor = reference_pop / actual_pop
    #                         scaled_deaths = actual_deaths * scale_factor
    #                         
    #                         total_scaled_deaths += scaled_deaths
    #             
    #             # Create ASMR row for this sex/dose combination
    #             asmr_row = {
    #                 'ISOweekDied': week, 
    #                 'YearOfBirth': 'ASMR',  # Use 'ASMR' string instead of 0
    #                 'Sex': sex_val,         # Preserve original sex
    #                 'Dose': d,              # Preserve original dose number
    #                 'Alive': full_standard_pop,
    #                 'Dead': int(round(total_scaled_deaths))
    #             }
    #             asmr_rows.append(asmr_row)
    # 
    # # Convert ASMR rows to DataFrame and append
    # if asmr_rows:
    #     asmr_df = pd.DataFrame(asmr_rows)
    #     # Add DateDied column to ASMR rows
    #     asmr_df['DateDied'] = asmr_df['ISOweekDied'].apply(lambda week: pd.to_datetime(week + '-1', format='%G-%V-%u').strftime('%Y-%m-%d'))
    #     # Reorder columns to match main DataFrame
    #     asmr_df = asmr_df[['ISOweekDied', 'DateDied', 'YearOfBirth', 'Sex', 'Dose', 'Alive', 'Dead']]
    #     
    #     out = pd.concat([out, asmr_df], ignore_index=True)
    #     print(f"    Added {len(asmr_rows)} ASMR rows")
    
    # Write to Excel sheets (main + enrollment-week summary)
    print(f"  Writing to Excel sheet...")
    sheet_name = enroll_date_str.replace('-', '_')
    out.to_excel(excel_writer, sheet_name=sheet_name, index=False)
    # Enrollment-week per-dose totals for quick cross-check
    try:
        summary = (
            out[out['ISOweekDied'] == enroll_week_str]
              .groupby('Dose', observed=True)[['Alive', 'Dead']]
              .sum()
              .reset_index()
              .sort_values('Dose')
        )
        summary.to_excel(excel_writer, sheet_name=sheet_name + "_summary", index=False)
    except Exception as e:
        print(f"CAUTION: Failed to write summary sheet: {e}")
    
    # Format YearOfBirth column as text to avoid Excel warnings
    workbook = excel_writer.book
    worksheet = excel_writer.sheets[sheet_name]
    # the following line is commented out because it simply didn't work. Excel flags the YearOfBirth column as numbers anyway so you have to ignore the warning in Excel and save.
    # text_format = workbook.add_format({'num_format': '@'})  # '@' = text format
    # worksheet.set_column('C:C', None, text_format)  # Column C is YearOfBirth
    
    print(f"Added sheet {sheet_name} to {excel_out_path}")
    print(f"Completed enrollment date {enroll_date_str} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

# Save the Excel file after all sheets are added
excel_writer.close()
print(f"Wrote all dose group CMRs to {excel_out_path}")

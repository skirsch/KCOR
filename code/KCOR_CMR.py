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
from typing import Optional

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

# Process latest enrollment first to avoid any chance of state leakage across runs
try:
    _ed_sorted = sorted(enrollment_dates, key=lambda s: pd.to_datetime(s + '-1', format='%G-%V-%u'), reverse=True)
    if _ed_sorted != enrollment_dates:
        print(f"Reordering enrollment dates (latest first): {', '.join(_ed_sorted)}")
        enrollment_dates = _ed_sorted
except Exception as _e_sort:
    print(f"CAUTION: Could not sort enrollment dates: {_e_sort}")

# enrollment_dates = ['2021-24']  # For testing, just do one enrollment date

## Load the dataset with explicit types and rename columns to English
import datetime as _dt
print(f"KCOR_CMR start: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Reading the input file: {input_file}")


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

a = _read_csv_flex(input_file)

# Validate column count before renaming; FOIA or alternate sources may use different delimiters/encodings
expected_col_count = 53
if a.shape[1] != expected_col_count:
    print(f"ERROR: Input parsed into {a.shape[1]} columns, expected {expected_col_count}.\n"
          f"       This usually indicates a delimiter or encoding mismatch.\n"
          f"       First row preview: {a.head(1).to_dict(orient='records')}")
    sys.exit(1)

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

# Ensure DCCI buckets are {-1,0,1,3,5}; collapse 2–4 -> 3; map NaN/negatives -> -1
a['DCCI'] = pd.to_numeric(a['DCCI'], errors='coerce')
# NaN and negatives become -1 (unknown)
a['DCCI'] = a['DCCI'].where(a['DCCI'].notna(), -1)
a['DCCI'] = a['DCCI'].where(a['DCCI'] >= 0, -1)
# Collapse 2–4 to 3
a.loc[(a['DCCI'] >= 2) & (a['DCCI'] <= 4), 'DCCI'] = 3
# Enforce allowed set; unexpected values fallback to -1
a['DCCI'] = a['DCCI'].where(a['DCCI'].isin([-1, 0, 1, 3, 5]), -1).astype('Int8')

# Immediately drop columns we won't use to reduce memory
needed_cols = [
    'Infection', 'Sex', 'YearOfBirth',
    'Date_FirstDose', 'Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose',
    'VaccineCode_SecondDose', 'VaccineCode_ThirdDose', 'VaccineCode_FourthDose',
    'Date_COVID_death', 'DateOfDeath', 'DCCI'
]
# Take only needed columns as a fresh copy to avoid chained-assignment warnings
a = a.loc[:, needed_cols].copy()

# if you got infected more than once, it will create a duplicate record (with a different ID) so
# remove those records so we don't double count the deaths.

# Remove records where Infection > 1
# Filter to single-infection rows as a fresh copy
a = a[(a['Infection'].fillna(0).astype(int) <= 1)].copy()

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
print("Sex distribution:")
print(a['Sex'].value_counts())



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

# --------- Manufacturer mapping for doses 2-4 (P/M/O) ---------
def _install_czech_code_path():
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Czech/code'))
        if base_dir not in sys.path:
            sys.path.append(base_dir)
    except Exception:
        pass

_install_czech_code_path()

try:
    from mfg_codes import parse_mfg as _parse_mfg
except Exception as _e_mfg_import:
    print(f"CAUTION: Could not import mfg_codes.parse_mfg: {_e_mfg_import}. Using fallback mapping.")
    _parse_mfg = None

def _mfg_to_PMO(vcode: str) -> str:
    """Map Czech vaccine code (e.g., CO01/CO02/...) to simplified manufacturer: P, M, or O.

    Returns empty string for blank/NaN inputs.
    """
    if pd.isna(vcode) or vcode == '':
        return ''
    # Preferred path: use shared Czech mapping, then collapse to P/M/O
    if _parse_mfg is not None:
        try:
            mfg = _parse_mfg(vcode)
            if mfg == 'PF':
                return 'P'
            if mfg == 'MO':
                return 'M'
            return 'O'
        except Exception:
            # Fall through to direct mapping
            pass
    # Fallback: direct code mapping by observed manufacturer groupings
    code = str(vcode)
    if code in ('CO01','CO08','CO09','CO16','CO20','CO21','CO23'):
        return 'P'  # Pfizer
    if code in ('CO02','CO15','CO19'):
        return 'M'  # Moderna
    return 'O'      # Other manufacturers (e.g., Astra, J&J, Novavax, unknown)

# Add simplified manufacturer columns for doses 2-4
for _src, _dst in (
    ('VaccineCode_SecondDose', 'Dose2_MFG'),
    ('VaccineCode_ThirdDose',  'Dose3_MFG'),
    ('VaccineCode_FourthDose', 'Dose4_MFG'),
):
    if _src in a.columns:
        a[_dst] = a[_src].apply(_mfg_to_PMO).astype('string')

# Fix death dates - now we can compare properly since all dates are pandas Timestamps
## Only use LPZ death date, ignore other death date
a = a[~((a['DateOfDeath'].notnull()) & (a['Date_FirstDose'] > a['DateOfDeath']))]

# Convert birth years to integers once (outside the enrollment loop)
print(f"Converting birth years to integers (one time only)...")
a['YearOfBirth'] = a['birth_year'].apply(lambda x: int(x) if pd.notnull(x) else -1)
print(f"Birth year conversion complete.")

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
    # Restrict processing to birth years within [1920, 2000] inclusive, but keep -1 (unknown)
    before = len(a_copy)
    a_copy = a_copy[((a_copy['YearOfBirth'] >= 1920) & (a_copy['YearOfBirth'] <= 2005)) | (a_copy['YearOfBirth'] == -1)].copy()
    after = len(a_copy)
    print(f"  Filtered YearOfBirth to 1920-2000: kept {after}/{before} records")
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
    
    # Optionally freeze cohort moves after enrollment by zeroing post-enrollment dose dates
    # Set env BYPASS_FREEZE=1 to treat entire dataset as one variable cohort (no truncation)
    a_var = a_copy.copy()
    _bypass_freeze = str(os.environ.get('BYPASS_FREEZE','')).strip().lower() in ('1','true','yes')
    if not _bypass_freeze:
        for _col in ['Date_FirstDose','Date_SecondDose','Date_ThirdDose','Date_FourthDose']:
            # Freeze transitions on or after enrollment Monday to keep fixed cohorts post-enrollment
            a_var.loc[a_var[_col] >= enrollment_date, _col] = pd.NaT

    # Create boolean masks for each dose being valid (not null and <= reference_date) on the truncated dates
    dose1_valid = a_var['Date_FirstDose'].notna() & (a_var['Date_FirstDose'] <= reference_dates)
    dose2_valid = a_var['Date_SecondDose'].notna() & (a_var['Date_SecondDose'] <= reference_dates)
    dose3_valid = a_var['Date_ThirdDose'].notna() & (a_var['Date_ThirdDose'] <= reference_dates)
    dose4_valid = a_var['Date_FourthDose'].notna() & (a_var['Date_FourthDose'] <= reference_dates)
    
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
    # IMPORTANT: classify dose-at-death using strictly earlier dose dates
    # to maintain start-of-week semantics (no same-week promotion).
    death_ref = a_var['DateOfDeath']
    d1_at = a_var['Date_FirstDose'].notna() & (a_var['Date_FirstDose'] < death_ref)
    d2_at = a_var['Date_SecondDose'].notna() & (a_var['Date_SecondDose'] < death_ref)
    d3_at = a_var['Date_ThirdDose'].notna() & (a_var['Date_ThirdDose'] < death_ref)
    d4_at = a_var['Date_FourthDose'].notna() & (a_var['Date_FourthDose'] < death_ref)
    a_copy['dose_at_death'] = 0
    a_copy.loc[d1_at, 'dose_at_death'] = 1
    a_copy.loc[d2_at, 'dose_at_death'] = 2
    a_copy.loc[d3_at, 'dose_at_death'] = 3
    a_copy.loc[d4_at, 'dose_at_death'] = 4  # 4 = 4+
    # Pre-enrollment deaths will be recomputed after building week index to enforce start-of-week semantics
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
    # Stable numeric order for ISO weeks to avoid lexicographic mis-order (e.g., 2021-10 < 2021-2)
    week_index = {wk: idx for idx, wk in enumerate(all_weeks)}
    print(f"  Preparing output structure (vectorized)...")
    # Single death table aligned to start-of-week conventions
    mask_deaths = a_copy['DateOfDeath'].notnull() & a_copy['WeekOfDeath'].notna()
    # Determine dose at the start of the death week (Monday)
    week_monday = pd.to_datetime(a_copy['WeekOfDeath'] + '-1', format='%G-%V-%u', errors='coerce')
    d1_w = a_var['Date_FirstDose'].notna() & (a_var['Date_FirstDose'] < week_monday)
    d2_w = a_var['Date_SecondDose'].notna() & (a_var['Date_SecondDose'] < week_monday)
    d3_w = a_var['Date_ThirdDose'].notna() & (a_var['Date_ThirdDose'] < week_monday)
    d4_w = a_var['Date_FourthDose'].notna() & (a_var['Date_FourthDose'] < week_monday)
    a_copy['dose_at_week'] = 0
    a_copy.loc[d1_w, 'dose_at_week'] = 1
    a_copy.loc[d2_w, 'dose_at_week'] = 2
    a_copy.loc[d3_w, 'dose_at_week'] = 3
    a_copy.loc[d4_w, 'dose_at_week'] = 4
    # Attribute all deaths by dose at start of week (variable cohorts), with post-enrollment transitions frozen
    deaths_week = (
        a_copy[mask_deaths]
            .groupby(['WeekOfDeath','YearOfBirth','Sex','DCCI','dose_at_week'])
            .size()
            .reset_index(name='dead')
    )
    # COVID-attributed deaths keyed to DateOfDeath's week (only those with non-empty Date_COVID_death)
    covid_flag = a_copy['Date_COVID_death'].notna() & (a_copy['Date_COVID_death'].astype(str).str.strip() != '')
    deaths_week_covid = (
        a_copy[mask_deaths & covid_flag]
            .groupby(['WeekOfDeath','YearOfBirth','Sex','DCCI','dose_at_week'])
            .size()
            .reset_index(name='dead_covid')
    )
    observed_deaths = deaths_week.rename(columns={'dose_at_week':'Dose','WeekOfDeath':'ISOweekDied'})[
        ['ISOweekDied','YearOfBirth','Sex','DCCI','Dose','dead']
    ]
    observed_deaths_covid = deaths_week_covid.rename(columns={'dose_at_week':'Dose','WeekOfDeath':'ISOweekDied'})[
        ['ISOweekDied','YearOfBirth','Sex','DCCI','Dose','dead_covid']
    ]
    print(f"    Total deaths across all dose groups: {int(observed_deaths['dead'].sum())}")
    print(f"    Unique weeks with deaths: {len(a_copy.loc[mask_deaths, 'WeekOfDeath'].dropna().unique())}")
    # Single aligned death series covers both pre- and post-enrollment periods

    # Build dose-agnostic grid to avoid duplicating transition counts
    weeks_df = pd.DataFrame({'ISOweekDied': all_weeks})
    nodose_combos = a_copy[['YearOfBirth', 'Sex', 'DCCI']].drop_duplicates().copy()
    weeks_df['__k'] = 1
    nodose_combos['__k'] = 1
    grid_nodose = weeks_df.merge(nodose_combos, on='__k', how='left').drop(columns='__k')
    grid_nodose['WeekIdx'] = grid_nodose['ISOweekDied'].map(week_index).astype(int)

    # Cross join with Dose 0..4 to build full output grid
    dose_df = pd.DataFrame({'Dose': [0,1,2,3,4]})
    dose_df['__k'] = 1
    grid_nodose['__k'] = 1
    out = grid_nodose.merge(dose_df, on='__k', how='left').drop(columns='__k')

    # Attach single aligned death series
    out = out.merge(observed_deaths, on=['ISOweekDied','YearOfBirth','Sex','DCCI','Dose'], how='left')
    out['dead'] = out['dead'].fillna(0).astype(int)
    # Attach COVID-specific deaths aligned to the same keys
    out = out.merge(observed_deaths_covid, on=['ISOweekDied','YearOfBirth','Sex','DCCI','Dose'], how='left')
    out['dead_covid'] = out['dead_covid'].fillna(0).astype(int)

    out['ISOweekDied'] = out['ISOweekDied'].astype(str)
    out['WeekIdx'] = out['ISOweekDied'].map(week_index).astype(int)
    # Optimize dtypes for speed and memory
    out['Sex'] = out['Sex'].astype('category')
    out['Dose'] = out['Dose'].astype('int8')
    out['DCCI'] = out['DCCI'].astype('Int8')
    
    # Add DateDied as Timestamp (we'll convert to string just before writing)
    out['DateDied'] = out['ISOweekDied'].apply(lambda week: pd.to_datetime(week + '-1', format='%G-%V-%u'))

    # Overwrite population columns to reflect attrition from deaths (vectorized)
    print(f"  Computing population attrition from deaths (vectorized)...")
    out = out.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'WeekIdx'])
    # DateDied already computed above as Timestamp

    # -------- Pre-enrollment Alive (variable cohorts at start of week) --------
    # Base total per (YearOfBirth, Sex, DCCI) on nodose grid
    base_total = a_copy.groupby(['YearOfBirth', 'Sex', 'DCCI']).size().rename('base_total').reset_index()
    out = out.merge(base_total, on=['YearOfBirth', 'Sex', 'DCCI'], how='left')
    out['base_total'] = out['base_total'].fillna(0).astype(int)

    # Weekly dose transitions counts per combo (to doses 1..4) computed on nodose grid
    dose_week_cols = [
        ('trans_1', 'Date_FirstDose'),
        ('trans_2', 'Date_SecondDose'),
        ('trans_3', 'Date_ThirdDose'),
        ('trans_4', 'Date_FourthDose'),
    ]
    trans_frames = []
    for label, col in dose_week_cols:
        wk = a_var[col].dt.strftime('%G-%V')
        t = a_var.loc[wk.notna(), ['YearOfBirth', 'Sex', 'DCCI']].copy()
        # Record transition in its own week; Alive for week t uses cumulative up to t-1 via shift
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

    # Compute cumulative transitions on nodose grid
    trans_grid = grid_nodose.merge(transitions, on=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'], how='left')
    trans_grid[['trans_1', 'trans_2', 'trans_3', 'trans_4']] = trans_grid[['trans_1', 'trans_2', 'trans_3', 'trans_4']].fillna(0).astype(int)
    cumT = (
        trans_grid.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'WeekIdx'])
                 .groupby(['YearOfBirth', 'Sex', 'DCCI'])[['trans_1', 'trans_2', 'trans_3', 'trans_4']] 
                 .cumsum()
                 .shift(fill_value=0)
    )
    cumT.columns = ['cumT1_prev', 'cumT2_prev', 'cumT3_prev', 'cumT4_prev']
    trans_grid = pd.concat([trans_grid, cumT], axis=1)
    # Merge cumulative transitions into output grid
    out = out.merge(trans_grid[['ISOweekDied','YearOfBirth','Sex','DCCI','cumT1_prev','cumT2_prev','cumT3_prev','cumT4_prev']],
                    on=['ISOweekDied','YearOfBirth','Sex','DCCI'], how='left')
    out[['cumT1_prev','cumT2_prev','cumT3_prev','cumT4_prev']] = out[['cumT1_prev','cumT2_prev','cumT3_prev','cumT4_prev']].fillna(0).astype(int)

    # Build Dead directly from aligned series
    out['Dead'] = out['dead'].astype(int)
    out['Dead_COVID'] = out['dead_covid'].astype(int)
    # Cumulative prior-week deaths per combo+dose (single series)
    out['cumDead_prev'] = (
        out.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'WeekIdx'])
           .groupby(['YearOfBirth', 'Sex', 'DCCI', 'Dose'])['dead']
           .cumsum()
           .shift(fill_value=0)
    )
    # Ensure boundary conditions at the very first week (no prior transitions/deaths)
    first_week_str = all_weeks[0]
    first_mask = out['ISOweekDied'] == first_week_str
    out.loc[first_mask, ['cumT1_prev','cumT2_prev','cumT3_prev','cumT4_prev']] = 0

    # Compute Alive for all weeks using variable-cohort formula (no enrollment freezing)
    dose_vals = out['Dose'].values
    alive_all = np.where(dose_vals == 0, out['base_total'] - out['cumT1_prev'] - out['cumDead_prev'],
                 np.where(dose_vals == 1, out['cumT1_prev'] - out['cumT2_prev'] - out['cumDead_prev'],
                 np.where(dose_vals == 2, out['cumT2_prev'] - out['cumT3_prev'] - out['cumDead_prev'],
                 np.where(dose_vals == 3, out['cumT3_prev'] - out['cumT4_prev'] - out['cumDead_prev'],
                                      out['cumT4_prev'] - out['cumDead_prev']))))
    out['Alive'] = np.maximum(alive_all, 0).astype(int)
    # Ensure explicit first-week boundary: start-of-series counts live at first week
    first_week_str = all_weeks[0]
    first_mask = out['ISOweekDied'] == first_week_str
    out.loc[first_mask & (out['Dose'] == 0), 'Alive'] = out.loc[first_mask & (out['Dose'] == 0), 'base_total']
    out.loc[first_mask & (out['Dose'] > 0), 'Alive'] = 0

    # (trace instrumentation removed)

    # Sanity checks with detailed prints for first few offenders
    try:
        bad = out[out['Dead'] > out['Alive']]
        if not bad.empty:
            sample = bad[['ISOweekDied','YearOfBirth','Sex','DCCI','Dose','Alive','Dead']].head(10)
            print(f"ERROR: Dead > Alive in {len(bad)} rows. First 10:")
            for _, r in sample.iterrows():
                print(f"  {r['ISOweekDied']} YoB={r['YearOfBirth']} Sex={r['Sex']} DCCI={r['DCCI']} Dose={r['Dose']}: Alive={r['Alive']} Dead={r['Dead']}")
            # Extra detail for first offender to see components
            r0 = sample.iloc[0]
            m0 = (
                (out['ISOweekDied'] == r0['ISOweekDied']) &
                (out['YearOfBirth'] == r0['YearOfBirth']) &
                (out['Sex'] == r0['Sex']) &
                (out['DCCI'] == r0['DCCI']) &
                (out['Dose'] == r0['Dose'])
            )
            row = out.loc[m0].iloc[0]
            bt_row = base_total[(base_total['YearOfBirth']==r0['YearOfBirth']) & (base_total['Sex']==r0['Sex']) & (base_total['DCCI']==r0['DCCI'])]
            bt = int(bt_row['base_total'].iloc[0]) if not bt_row.empty else 0
            print(f"DETAIL: base_total={bt} Alive={int(row.get('Alive',-1))} cumDead_prev={int(row.get('cumDead_prev',-1))}")
            print(f"        dead={int(row.get('dead',-1))}")
        # Dose 0 monotonic check across all weeks
        out = out.sort_values(['YearOfBirth','Sex','DCCI','Dose','DateDied'])
        # NEW: Dose 0 alive must be monotonically non-increasing
        d0 = out[out['Dose'] == 0].copy()
        d0 = d0.sort_values(['YearOfBirth','Sex','DCCI','DateDied'])
        d0_prev = d0.groupby(['YearOfBirth','Sex','DCCI'])['Alive'].shift(1)
        violations = d0[d0['Alive'] > d0_prev]
        if not violations.empty:
            print(f"ERROR: Dose 0 Alive increased in {len(violations)} rows. First 10:")
            for _, r in violations[['ISOweekDied','YearOfBirth','Sex','DCCI','Alive']].head(10).iterrows():
                print(f"  {r['ISOweekDied']} YoB={r['YearOfBirth']} Sex={r['Sex']} DCCI={r['DCCI']}: Alive increased to {int(r['Alive'])}")
    except Exception as e:
        print(f"CAUTION: Sanity check (Dead<=Alive) failed: {e}")

    # Drop helper columns
    out.drop(columns=[c for c in [
        'dead',
        'dead_covid',
        'dead_pre','dead_post',
        'base_total',
        'trans_1','trans_2','trans_3','trans_4',
        'cumT1_prev','cumT2_prev','cumT3_prev','cumT4_prev',
        'cumDead_prev',
        'WeekIdx'
    ] if c in out.columns], inplace=True)

    # Ensure Dead_COVID column is placed right after Dead
    if 'Dead_COVID' in out.columns and 'Dead' in out.columns:
        cols = list(out.columns)
        cols.remove('Dead_COVID')
        insert_at = cols.index('Dead') + 1
        cols.insert(insert_at, 'Dead_COVID')
        out = out[cols]

    # Convert DateDied back to string for Excel after computations
    out['DateDied'] = out['DateDied'].dt.strftime('%Y-%m-%d')

    # Note: Enrollment-week Alive may differ from pop_base due to boundary semantics
    # (pop_base is inclusive of same-week dose dates; Alive here is start-of-week prior to same-week transitions).
    # We intentionally skip a strict equality check to avoid false alarms.
    
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
              .groupby('Dose', observed=True)[['Alive', 'Dead', 'Dead_COVID']]
              .sum()
              .reset_index()
              .sort_values('Dose')
        )
        # Add manufacturer composition at enrollment for doses 2-4
        def _mfg_counts(dose_num: int, mfg_col: str):
            dfc = alive_at_enroll[alive_at_enroll['dose_group'] == dose_num]
            counts = dfc[mfg_col].value_counts()
            p = int(counts.get('P', 0))
            m = int(counts.get('M', 0))
            o = int(counts.get('O', 0))
            return p, m, o

        for d in (2, 3, 4):
            for k in ('P', 'M', 'O'):
                col = f"Dose{d}_MFG_{k}"
                if col not in summary.columns:
                    summary[col] = 0
        for d, mcol in ((2, 'Dose2_MFG'), (3, 'Dose3_MFG'), (4, 'Dose4_MFG')):
            try:
                p, m, o = _mfg_counts(d, mcol)
                summary.loc[summary['Dose'] == d, [f'Dose{d}_MFG_P', f'Dose{d}_MFG_M', f'Dose{d}_MFG_O']] = [p, m, o]
            except Exception as _e_sum_mfg:
                print(f"CAUTION: Failed to compute summary MFG counts for dose {d}: {_e_sum_mfg}")
        summary.to_excel(excel_writer, sheet_name=sheet_name + "_summary", index=False)
    except Exception as e:
        print(f"CAUTION: Failed to write summary sheet: {e}")
    
    # Write MFG-specific per-week series for doses 2,3,4 as auxiliary sheets
    try:
        def build_mfg_out_for_dose(dose_num: int, mfg_label: str) -> pd.DataFrame:
            # Subset individuals whose dose{d} manufacturer matches label
            mcol = {2: 'Dose2_MFG', 3: 'Dose3_MFG', 4: 'Dose4_MFG'}.get(dose_num)
            if not mcol:
                return pd.DataFrame()
            sub = a_var[(a_var[mcol] == mfg_label)].copy()
            if sub.empty:
                return pd.DataFrame()
            # Deaths per week for those at dose==dose_num at week start
            wk_monday = pd.to_datetime(sub['WeekOfDeath'] + '-1', format='%G-%V-%u', errors='coerce')
            d_w = (
                (dose_num == 1) & (sub['Date_FirstDose'].notna() & (sub['Date_FirstDose'] < wk_monday))
            ) | (
                (dose_num == 2) & (sub['Date_SecondDose'].notna() & (sub['Date_SecondDose'] < wk_monday))
            ) | (
                (dose_num == 3) & (sub['Date_ThirdDose'].notna() & (sub['Date_ThirdDose'] < wk_monday))
            ) | (
                (dose_num == 4) & (sub['Date_FourthDose'].notna() & (sub['Date_FourthDose'] < wk_monday))
            )
            sub = sub.assign(_at_dose_week=np.where(d_w, dose_num, 0))
            mask_deaths_sub = sub['DateOfDeath'].notnull() & sub['WeekOfDeath'].notna() & (sub['_at_dose_week'] == dose_num)
            deaths_sub = (
                sub[mask_deaths_sub]
                  .groupby(['WeekOfDeath','YearOfBirth','Sex','DCCI'])
                  .size()
                  .reset_index(name='dead')
            )
            obs = deaths_sub.rename(columns={'WeekOfDeath':'ISOweekDied'})
            obs['ISOweekDied'] = obs['ISOweekDied'].astype(str)
            # Build nodose grid and attach WeekIdx
            weeks_df2 = pd.DataFrame({'ISOweekDied': all_weeks})
            nodose2 = sub[['YearOfBirth','Sex','DCCI']].drop_duplicates().copy()
            weeks_df2['__k'] = 1
            nodose2['__k'] = 1
            grid2 = weeks_df2.merge(nodose2, on='__k', how='left').drop(columns='__k')
            grid2['WeekIdx'] = grid2['ISOweekDied'].map(week_index).astype(int)
            # Attach deaths
            out2 = grid2.merge(obs, on=['ISOweekDied','YearOfBirth','Sex','DCCI'], how='left')
            out2['dead'] = out2['dead'].fillna(0).astype(int)
            # Transitions into dose d and out to d+1 for this subset
            date_col_d = {1:'Date_FirstDose',2:'Date_SecondDose',3:'Date_ThirdDose',4:'Date_FourthDose'}[dose_num]
            date_col_next = {1:'Date_SecondDose',2:'Date_ThirdDose',3:'Date_FourthDose',4:'Date_FifthDose'}.get(dose_num)
            # Inflow to d
            wk_in = sub[date_col_d].dt.strftime('%G-%V')
            t_in = sub.loc[wk_in.notna(), ['YearOfBirth','Sex','DCCI']].copy()
            t_in['ISOweekDied'] = wk_in[wk_in.notna()].values
            t_in = t_in.groupby(['ISOweekDied','YearOfBirth','Sex','DCCI']).size().reset_index(name='trans_in')
            # Outflow to d+1 (if next dose exists)
            if date_col_next in sub.columns:
                wk_out = sub[date_col_next].dt.strftime('%G-%V')
                t_out = sub.loc[wk_out.notna(), ['YearOfBirth','Sex','DCCI']].copy()
                t_out['ISOweekDied'] = wk_out[wk_out.notna()].values
                t_out = t_out.groupby(['ISOweekDied','YearOfBirth','Sex','DCCI']).size().reset_index(name='trans_out')
            else:
                t_out = pd.DataFrame(columns=['ISOweekDied','YearOfBirth','Sex','DCCI','trans_out'])
            # Merge transitions onto grid and compute cumulative prev (shifted)
            trans2 = grid2.merge(t_in, on=['ISOweekDied','YearOfBirth','Sex','DCCI'], how='left')
            trans2 = trans2.merge(t_out, on=['ISOweekDied','YearOfBirth','Sex','DCCI'], how='left')
            trans2[['trans_in','trans_out']] = trans2[['trans_in','trans_out']].fillna(0).astype(int)
            trans2 = trans2.sort_values(['YearOfBirth','Sex','DCCI','WeekIdx'])
            cum_in_prev = trans2.groupby(['YearOfBirth','Sex','DCCI'])['trans_in'].cumsum().shift(fill_value=0)
            cum_out_prev = trans2.groupby(['YearOfBirth','Sex','DCCI'])['trans_out'].cumsum().shift(fill_value=0)
            out2 = out2.merge(trans2[['ISOweekDied','YearOfBirth','Sex','DCCI']].assign(cum_in_prev=cum_in_prev, cum_out_prev=cum_out_prev),
                              on=['ISOweekDied','YearOfBirth','Sex','DCCI'], how='left')
            out2[['cum_in_prev','cum_out_prev']] = out2[['cum_in_prev','cum_out_prev']].fillna(0).astype(int)
            # Cumulative prior-week deaths per combo
            out2 = out2.sort_values(['YearOfBirth','Sex','DCCI','WeekIdx'])
            out2['cumDead_prev'] = (
                out2.groupby(['YearOfBirth','Sex','DCCI'])['dead']
                    .cumsum()
                    .shift(fill_value=0)
            )
            # Alive for dose d, mfg label
            out2['Alive'] = np.maximum(out2['cum_in_prev'] - out2['cum_out_prev'] - out2['cumDead_prev'], 0).astype(int)
            # First week boundary: 0 alive (since no prior inflow)
            first_mask2 = out2['ISOweekDied'] == all_weeks[0]
            out2.loc[first_mask2, 'Alive'] = 0
            out2['Dead'] = out2['dead'].astype(int)
            out2['Dose'] = dose_num
            out2['MFG'] = mfg_label
            out2['DateDied'] = pd.to_datetime(out2['ISOweekDied'] + '-1', format='%G-%V-%u').dt.strftime('%Y-%m-%d')
            return out2[['ISOweekDied','DateDied','YearOfBirth','Sex','DCCI','Dose','MFG','Alive','Dead']]

        # For specific doses per enrollment, create M/P sheets
        d_targets = []
        if enroll_date_str in ('2021-24','2021_24'):
            d_targets.append(2)
        if enroll_date_str in ('2022-06','2022_06'):
            d_targets.append(3)
        if enroll_date_str in ('2022-47','2022_47'):
            d_targets.append(4)
        for dnum in d_targets:
            frames = []
            for lab in ['P','M','O']:
                dfm = build_mfg_out_for_dose(dnum, lab)
                if not dfm.empty:
                    frames.append(dfm)
            if frames:
                mfg_sheet = f"{sheet_name}_MFG_D{dnum}"
                pd.concat(frames, ignore_index=True).to_excel(excel_writer, sheet_name=mfg_sheet, index=False)
                print(f"  Wrote manufacturer-specific sheet: {mfg_sheet}")
    except Exception as e:
        print(f"CAUTION: Failed to write MFG-specific sheets: {e}")

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

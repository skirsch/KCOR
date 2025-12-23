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
# latest changee was adding the 2021-W20 enrollment date.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import tempfile
import glob
from typing import Optional
from multiprocessing import Pool, cpu_count, get_context
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Check for Monte Carlo mode
MONTE_CARLO_MODE = str(os.environ.get('MONTE_CARLO', '')).strip().lower() in ('1', 'true', 'yes')
MC_ITERATIONS = int(os.environ.get('MC_ITERATIONS', '25'))
MC_THREADS = int(os.environ.get('MC_THREADS', '5'))  # Number of parallel threads for Monte Carlo iterations (reduced from 20 to 5 to avoid memory exhaustion)

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

# In Monte Carlo mode, override output path if not explicitly provided
if MONTE_CARLO_MODE and len(sys.argv) < 3:
    excel_out_path = "../data/KCOR_CMR_MC.xlsx"

excel_writer = pd.ExcelWriter(excel_out_path, engine='xlsxwriter')


# Define enrollment dates for dose groups
# These dates are used to determine the dose group for each individual based on their vaccination dates.
# These are ISO week format: YYYY-WW
# The enrollment date is the date when the individual is considered to be part of the study cohort.
# 2021-W13 is 03-29-2021, near the start of the vaccination campaign so we can capture impact on older people.
# 2021-W24 is 06-14-2021, when everyone 40+ was eligible for first dose.
# 2021-W41 is 10-11-2021, which is a late enrollment date before the winter wave; not super useful.
# 2022-W06 is 02-07-2022, which is the best booster #1 enrollment since it is just after everyone got 1st booster.
# 2022-W26 is 06-27-2022, which is the "5 months after booster" cohort. Purpose: to show all 3 dose groups will track each other at that point because all groups are now >15 weeks from a vaccine dose.
# 2022-W47 is 11-21-2022, which is the best booster #2 enrollment since it is just after everyone got 2nd booster.
# 2024-W01 is 12-30-2023, which is the best booster #3 enrollment since it is just after everyone got 3rd booster, but too late to be useful
# because the deaths start declining in Q2 of 2024
enrollment_dates = ['2021-13', '2021-20', '2021-24', '2021-30', '2022-06', '2022-26', '2022-47']  # Full set of enrollment dates

# In Monte Carlo mode, only process a single enrollment cohort (default 2022-06; override via MC_ENROLLMENT_DATE)
MC_ENROLL_DATE_STR = None
MC_MAX_DOSE_EFFECTIVE = None
if MONTE_CARLO_MODE:
    _env_mc_enroll = str(os.environ.get('MC_ENROLLMENT_DATE', '2022-06')).strip()
    if not _env_mc_enroll:
        _env_mc_enroll = '2022-06'
    _env_mc_enroll = _env_mc_enroll.replace('_', '-')
    _mc_ts = pd.to_datetime(_env_mc_enroll + '-1', format='%G-%V-%u', errors='coerce')
    if pd.isna(_mc_ts):
        print(f"ERROR: Invalid MC_ENROLLMENT_DATE={_env_mc_enroll!r} (expected YYYY-WW or YYYY_WW)", flush=True)
        sys.exit(2)
    MC_ENROLL_DATE_STR = _mc_ts.strftime('%G-%V')
    enrollment_dates = [MC_ENROLL_DATE_STR]

    # Dose-group cap for MC runs (matches cohort semantics used in KCOR analysis)
    if MC_ENROLL_DATE_STR in ('2021-13', '2021-20', '2021-24', '2021-30'):
        MC_MAX_DOSE_EFFECTIVE = 2
    elif MC_ENROLL_DATE_STR in ('2022-47',):
        MC_MAX_DOSE_EFFECTIVE = 4
    else:
        # Default (booster-1 style): treat dose 3 as \"3 or more doses\"
        MC_MAX_DOSE_EFFECTIVE = 3

    print(f"Monte Carlo mode: Processing only {MC_ENROLL_DATE_STR} enrollment with {MC_ITERATIONS} iterations", flush=True)
    print(f"  Using {MC_THREADS} parallel processes", flush=True)
    print(f"  Max dose group: {MC_MAX_DOSE_EFFECTIVE}", flush=True)

# Optional override via environment variable ENROLLMENT_DATES (comma-separated, e.g., "2021-24" or "2021-13,2021-24")
# (Only applies if not in Monte Carlo mode)
if not MONTE_CARLO_MODE:
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
    'VaccineCode_FirstDose', 'VaccineCode_SecondDose', 'VaccineCode_ThirdDose', 'VaccineCode_FourthDose',
    'Date_COVID_death', 'DateOfDeath', 'DCCI'
]
# Take only needed columns as a fresh copy to avoid chained-assignment warnings
a = a.loc[:, needed_cols].copy()

# if you got infected more than once, it will create a duplicate record (with a different ID) so
# remove those records so we don't double count the deaths.

# Remove records where Infection > 1
# Filter to single-infection rows as a fresh copy
a = a[(a['Infection'].fillna(0).astype(int) <= 1)].copy()

# Filter out records where person got a non-mRNA vaccine for any dose
try:
    from mfg_codes import parse_mfg, PFIZER, MODERNA
except ImportError as e:
    print(f"ERROR: Could not import mfg_codes: {e}")
    raise

# Check all vaccine codes (FirstDose through FourthDose) for non-mRNA vaccines
vaccine_code_cols = ['VaccineCode_FirstDose', 'VaccineCode_SecondDose', 'VaccineCode_ThirdDose', 'VaccineCode_FourthDose']
records_before_filter = len(a)

# Create a mask: True if record has any non-mRNA vaccine dose
has_non_mrna = pd.Series([False] * len(a), index=a.index)
for col in vaccine_code_cols:
    if col in a.columns:
        # Parse each vaccine code and check if it's non-mRNA
        mfg_values = a[col].apply(lambda x: parse_mfg(x) if pd.notna(x) and x != '' else None)
        # Mark records where this dose is non-mRNA (not PFIZER or MODERNA, and not None/empty)
        non_mrna_mask = (mfg_values.notna()) & (mfg_values != PFIZER) & (mfg_values != MODERNA)
        has_non_mrna = has_non_mrna | non_mrna_mask

# Filter out records with any non-mRNA vaccine dose
a = a[~has_non_mrna].copy()
records_after_filter = len(a)
records_removed = records_before_filter - records_after_filter
print(f"Filtered out {records_removed} records with non-mRNA vaccines (kept {records_after_filter} out of {records_before_filter} records)")

# Convert Sex to alphabetic codes: M, F, O
# For Monte Carlo mode: skip conversion since we aggregate across Sex anyway
if not MONTE_CARLO_MODE:
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
else:
    # For Monte Carlo: just ensure Sex column exists (use as-is or set to 'O' if needed)
    # We'll aggregate across Sex anyway, so exact values don't matter
    if 'Sex' not in a.columns or a['Sex'].isna().any():
        a['Sex'] = a['Sex'].fillna('O')



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
try:
    # mfg_codes may not be available; import is handled dynamically
    from mfg_codes import parse_mfg as _parse_mfg  # type: ignore[import-untyped]  # noqa: F401
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

# Monte Carlo mode: Extract master population for the requested enrollment
# Store as module-level variable so worker processes can access via fork() copy-on-write (no pickling!)
_master_monte_carlo_global = None
master_monte_carlo = None
if MONTE_CARLO_MODE:
    # For MC mode, we need to extract master population before the loop
    # Process the selected enrollment to get the master population
    enroll_date_str_mc = MC_ENROLL_DATE_STR or enrollment_dates[0]
    enrollment_date_mc = pd.to_datetime(enroll_date_str_mc + '-1', format='%G-%V-%u', errors='coerce')
    print(f"Monte Carlo mode: Extracting master population for {enroll_date_str_mc} enrollment...")
    a_copy_mc = a.copy()
    # Apply same filters as normal processing
    before_mc = len(a_copy_mc)
    a_copy_mc = a_copy_mc[((a_copy_mc['YearOfBirth'] >= 1920) & (a_copy_mc['YearOfBirth'] <= 2005)) | (a_copy_mc['YearOfBirth'] == -1)].copy()
    after_mc = len(a_copy_mc)
    print(f"  Filtered YearOfBirth to 1920-2005: kept {after_mc}/{before_mc} records")
    # Extract master_monte_carlo: people alive at start of enrollment week
    # Alive means: DateOfDeath is null OR DateOfDeath >= enrollment_date
    master_monte_carlo = a_copy_mc[
        (a_copy_mc['DateOfDeath'].isna()) | (a_copy_mc['DateOfDeath'] >= enrollment_date_mc)
    ].copy()
    # Store in module-level variable for fork() copy-on-write access (no pickling!)
    _master_monte_carlo_global = master_monte_carlo
    master_count = len(master_monte_carlo)
    print(f"  Master Monte Carlo population: {master_count:,} records (alive at start of enrollment week)")
    print(f"  Will perform {MC_ITERATIONS} bootstrap iterations using {MC_THREADS} parallel processes")

def process_enrollment_data(a_copy, enrollment_date, enroll_week_str, max_dose=4, monte_carlo_mode=False, iteration=None):
    """
    Core processing function for a given dataset and enrollment date.
    This function is shared between Monte Carlo and normal modes.
    
    Args:
        a_copy: DataFrame with the dataset to process
        enrollment_date: pandas Timestamp for enrollment date
        enroll_week_str: ISO week string (e.g., '2022-06')
        max_dose: Maximum dose to process (default 4). Doses above this are collapsed into max_dose.
                  For Monte Carlo mode, use max_dose=3 to treat dose 3 as "3 or more doses".
        monte_carlo_mode: If True, filter output to only post-enrollment weeks (reduces grid size).
        iteration: Optional iteration number for progress tracking (Monte Carlo mode only)
    
    Returns:
        tuple: (out, alive_at_enroll, all_weeks, week_index, pop_base)
            - out: DataFrame with processed results
            - alive_at_enroll: DataFrame of people alive at enrollment
            - all_weeks: List of all week strings (filtered for MC mode)
            - week_index: Dict mapping week strings to indices
            - pop_base: DataFrame with population base counts
    """
    import datetime
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta
    
    def _progress(msg):
        if monte_carlo_mode and iteration is not None:
            print(f"[Iteration {iteration}] {msg}", flush=True)
    
    _progress("Starting process_enrollment_data...")
    
    # Assign dose group as of enrollment date (highest dose <= enrollment date) - VECTORIZED VERSION
    _progress("Computing reference dates...")
    # For people who died before enrollment, use their death date instead of enrollment date
    reference_dates = a_copy['DateOfDeath'].where(
        a_copy['DateOfDeath'].notna() & (a_copy['DateOfDeath'] < enrollment_date),
        enrollment_date
    )
    
    _progress("Freezing cohort moves...")
    # Optionally freeze cohort moves after enrollment by zeroing post-enrollment dose dates
    # Set env BYPASS_FREEZE=1 to treat entire dataset as one variable cohort (no truncation)
    a_var = a_copy.copy()
    _bypass_freeze = str(os.environ.get('BYPASS_FREEZE','')).strip().lower() in ('1','true','yes')
    if not _bypass_freeze:
        for _col in ['Date_FirstDose','Date_SecondDose','Date_ThirdDose','Date_FourthDose']:
            # Freeze transitions on or after enrollment Monday to keep fixed cohorts post-enrollment
            a_var.loc[a_var[_col] >= enrollment_date, _col] = pd.NaT

    _progress("Assigning dose groups...")
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
    if max_dose >= 4:
        a_copy.loc[dose4_valid, 'dose_group'] = 4
    else:
        # For max_dose=3, collapse dose 4+ into dose 3
        a_copy.loc[dose4_valid, 'dose_group'] = 3
    # Collapse any doses above max_dose into max_dose
    if max_dose < 4:
        a_copy.loc[a_copy['dose_group'] > max_dose, 'dose_group'] = max_dose
    
    _progress("Computing population base...")
    # Compute population base: count of people in each (born, sex, dose_group)
    # Freeze dose at enrollment for population base and exclude those who died before enrollment
    alive_at_enroll = a_copy[(a_copy['DateOfDeath'].isna()) | (a_copy['DateOfDeath'] >= enrollment_date)].copy()
    if monte_carlo_mode:
        # For Monte Carlo: aggregate by Dose only
        pop_base = alive_at_enroll.groupby(['dose_group']).size().reset_index(name='pop')
        pop_base['YearOfBirth'] = -2  # Placeholder for compatibility
        pop_base['Sex'] = 'O'  # Placeholder
        pop_base['DCCI'] = 0  # Placeholder
    else:
        pop_base = alive_at_enroll.groupby(['YearOfBirth', 'Sex', 'DCCI', 'dose_group']).size().reset_index(name='pop')

    # Compute deaths per week using two classifications:
    #  - Pre-enrollment weeks: deaths grouped by dose at time of death (variable cohorts)
    #  - Post-enrollment weeks: deaths grouped by frozen enrollment dose (fixed cohorts)
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
    if max_dose >= 4:
        a_copy.loc[d4_at, 'dose_at_death'] = 4  # 4 = 4+
    else:
        a_copy.loc[d4_at, 'dose_at_death'] = max_dose  # Collapse 4+ into max_dose
    # Collapse any doses above max_dose into max_dose
    if max_dose < 4:
        a_copy.loc[a_copy['dose_at_death'] > max_dose, 'dose_at_death'] = max_dose
    
    _progress("Computing week range...")
    # Get all weeks in the study period (from database start to end, including pre-enrollment period)
    # Use all vaccination and death dates to get the full week range
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
    
    _progress(f"Building week list (from {min_year}-{min_week:02d} to {max_year}-{max_week:02d})...")
    # Build all weeks between min and max
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
    
    _progress(f"Filtering weeks for Monte Carlo (total weeks: {len(all_weeks)})...")
    # For Monte Carlo mode: filter to only weeks >= enrollment_date (post-enrollment only)
    # This significantly reduces the output grid size (weeks × YearOfBirth × Sex × DCCI × Dose)
    if monte_carlo_mode:
        enroll_week_monday = pd.to_datetime(enroll_week_str + '-1', format='%G-%V-%u', errors='coerce')
        # Vectorized filtering: convert all weeks to dates at once
        weeks_series = pd.Series(all_weeks)
        week_mondays = pd.to_datetime(weeks_series + '-1', format='%G-%V-%u', errors='coerce')
        mask_post_enroll = week_mondays >= enroll_week_monday
        all_weeks = weeks_series[mask_post_enroll].tolist()
        # Rebuild week_index for filtered weeks (starting from 0)
        week_index = {wk: idx for idx, wk in enumerate(all_weeks)}
        _progress(f"Week filtering complete (post-enrollment weeks: {len(all_weeks)})")
    
    _progress("Classifying deaths...")
    # Single death table aligned to start-of-week conventions
    mask_deaths = a_copy['DateOfDeath'].notnull() & a_copy['WeekOfDeath'].notna()
    # For fixed cohorts: attribute post-enrollment deaths to enrollment dose group
    # For pre-enrollment deaths: use dose at start of death week (variable cohorts)
    # Use vectorized conversion (was using .apply() before)
    week_monday = pd.to_datetime(a_copy['WeekOfDeath'] + '-1', format='%G-%V-%u', errors='coerce')
    is_post_enroll_death = week_monday >= enrollment_date
    
    # Pre-enrollment deaths: use dose at start of death week
    d1_pre = a_var['Date_FirstDose'].notna() & (a_var['Date_FirstDose'] < week_monday) & ~is_post_enroll_death
    d2_pre = a_var['Date_SecondDose'].notna() & (a_var['Date_SecondDose'] < week_monday) & ~is_post_enroll_death
    d3_pre = a_var['Date_ThirdDose'].notna() & (a_var['Date_ThirdDose'] < week_monday) & ~is_post_enroll_death
    d4_pre = a_var['Date_FourthDose'].notna() & (a_var['Date_FourthDose'] < week_monday) & ~is_post_enroll_death
    
    # Post-enrollment deaths: use enrollment dose group (fixed cohorts)
    # For post-enrollment deaths, use the enrollment dose_group directly
    a_copy['dose_at_week'] = 0
    a_copy.loc[d1_pre, 'dose_at_week'] = 1
    a_copy.loc[d2_pre, 'dose_at_week'] = 2
    a_copy.loc[d3_pre, 'dose_at_week'] = 3
    if max_dose >= 4:
        a_copy.loc[d4_pre, 'dose_at_week'] = 4
    else:
        a_copy.loc[d4_pre, 'dose_at_week'] = max_dose  # Collapse 4+ into max_dose
    # For post-enrollment deaths, use enrollment dose_group
    a_copy.loc[is_post_enroll_death, 'dose_at_week'] = a_copy.loc[is_post_enroll_death, 'dose_group']
    # Collapse any doses above max_dose into max_dose
    if max_dose < 4:
        a_copy.loc[a_copy['dose_at_week'] > max_dose, 'dose_at_week'] = max_dose
    
    _progress("Aggregating deaths...")
    # Attribute all deaths by dose (enrollment dose for post-enrollment, dose at death for pre-enrollment)
    # For Monte Carlo mode: aggregate across YearOfBirth, Sex, DCCI early to reduce grid size
    if monte_carlo_mode:
        # Aggregate deaths by (Week, Dose) only - no YearOfBirth, Sex, DCCI
        deaths_week = (
            a_copy[mask_deaths]
                .groupby(['WeekOfDeath','dose_at_week'])
                .size()
                .reset_index(name='dead')
        )
        # COVID-attributed deaths
        covid_flag = a_copy['Date_COVID_death'].notna() & (a_copy['Date_COVID_death'].astype(str).str.strip() != '')
        deaths_week_covid = (
            a_copy[mask_deaths & covid_flag]
                .groupby(['WeekOfDeath','dose_at_week'])
                .size()
                .reset_index(name='dead_covid')
        )
        observed_deaths = deaths_week.rename(columns={'dose_at_week':'Dose','WeekOfDeath':'ISOweekDied'})[
            ['ISOweekDied','Dose','dead']
        ]
        observed_deaths_covid = deaths_week_covid.rename(columns={'dose_at_week':'Dose','WeekOfDeath':'ISOweekDied'})[
            ['ISOweekDied','Dose','dead_covid']
        ]
    else:
        # Normal mode: keep YearOfBirth, Sex, DCCI dimensions
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
    
    _progress("Building output grid...")
    # Build output grid
    # For Monte Carlo mode: only (Week, Dose) - much smaller grid!
    # For normal mode: (Week, YearOfBirth, Sex, DCCI, Dose) - full grid
    if monte_carlo_mode:
        # Simple grid: weeks × doses only
        weeks_df = pd.DataFrame({'ISOweekDied': all_weeks})
        dose_df = pd.DataFrame({'Dose': list(range(max_dose + 1))})
        weeks_df['__k'] = 1
        dose_df['__k'] = 1
        out = weeks_df.merge(dose_df, on='__k', how='left').drop(columns='__k')
        out['WeekIdx'] = out['ISOweekDied'].map(week_index).astype(int)
    else:
        # Build dose-agnostic grid to avoid duplicating transition counts
        weeks_df = pd.DataFrame({'ISOweekDied': all_weeks})
        nodose_combos = a_copy[['YearOfBirth', 'Sex', 'DCCI']].drop_duplicates().copy()
        weeks_df['__k'] = 1
        nodose_combos['__k'] = 1
        grid_nodose = weeks_df.merge(nodose_combos, on='__k', how='left').drop(columns='__k')
        grid_nodose['WeekIdx'] = grid_nodose['ISOweekDied'].map(week_index).astype(int)

        # Cross join with Dose 0..max_dose to build full output grid
        dose_df = pd.DataFrame({'Dose': list(range(max_dose + 1))})
        dose_df['__k'] = 1
        grid_nodose['__k'] = 1
        out = grid_nodose.merge(dose_df, on='__k', how='left').drop(columns='__k')

    # Attach single aligned death series
    if monte_carlo_mode:
        # Merge on (ISOweekDied, Dose) only
        out = out.merge(observed_deaths, on=['ISOweekDied','Dose'], how='left')
        out['dead'] = out['dead'].fillna(0).astype(int)
        # Attach COVID-specific deaths
        out = out.merge(observed_deaths_covid, on=['ISOweekDied','Dose'], how='left')
        out['dead_covid'] = out['dead_covid'].fillna(0).astype(int)
    else:
        # Merge on full keys (ISOweekDied, YearOfBirth, Sex, DCCI, Dose)
        out = out.merge(observed_deaths, on=['ISOweekDied','YearOfBirth','Sex','DCCI','Dose'], how='left')
        out['dead'] = out['dead'].fillna(0).astype(int)
        # Attach COVID-specific deaths aligned to the same keys
        out = out.merge(observed_deaths_covid, on=['ISOweekDied','YearOfBirth','Sex','DCCI','Dose'], how='left')
        out['dead_covid'] = out['dead_covid'].fillna(0).astype(int)

    out['ISOweekDied'] = out['ISOweekDied'].astype(str)
    out['WeekIdx'] = out['ISOweekDied'].map(week_index).astype(int)
    # Optimize dtypes for speed and memory
    out['Dose'] = out['Dose'].astype('int8')
    if not monte_carlo_mode:
        out['Sex'] = out['Sex'].astype('category')
        out['DCCI'] = out['DCCI'].astype('Int8')
    
    # Add DateDied as Timestamp (we'll convert to string just before writing)
    # VECTORIZED: pd.to_datetime() on Series is vectorized, no need for .apply()
    out['DateDied'] = pd.to_datetime(out['ISOweekDied'] + '-1', format='%G-%V-%u', errors='coerce')

    _progress("Sorting output grid...")
    # Overwrite population columns to reflect attrition from deaths (vectorized)
    if monte_carlo_mode:
        out = out.sort_values(['Dose', 'WeekIdx'])
    else:
        out = out.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'WeekIdx'])

    _progress("Computing Alive column...")
    # -------- Fixed-cohort Alive calculation --------
    # For fixed cohorts: initialize each dose group at enrollment, then subtract only deaths
    
    if monte_carlo_mode:
        # For Monte Carlo: aggregate by Dose only, then create columns
        # First, ensure we only have doses 0 through max_dose
        pop_base_filtered = pop_base[pop_base['dose_group'] <= max_dose].copy()
        # Sum across all rows (since we aggregated by dose_group only)
        pop_by_dose = pop_base_filtered.groupby('dose_group')['pop'].sum().reset_index()
        # Create a dictionary mapping dose to population
        pop_dict = {row['dose_group']: row['pop'] for _, row in pop_by_dose.iterrows()}
        # Assign to output grid - ensure all doses 0-max_dose have values
        for d in range(max_dose + 1):
            out[f'pop_dose{d}'] = pop_dict.get(d, 0)
    else:
        # Merge enrollment population counts (pop_base) into output grid
        # pop_base has: YearOfBirth, Sex, DCCI, dose_group, pop
        # We need to pivot it so each dose group becomes a column
        pop_base_pivot = pop_base.pivot_table(
            index=['YearOfBirth', 'Sex', 'DCCI'],
            columns='dose_group',
            values='pop',
            fill_value=0
        ).reset_index()
        # Rename columns to pop_dose0, pop_dose1, etc.
        pop_base_pivot.columns = ['YearOfBirth', 'Sex', 'DCCI'] + [f'pop_dose{d}' for d in range(max_dose + 1)]
        
        # Merge enrollment counts into output grid
        out = out.merge(pop_base_pivot, on=['YearOfBirth', 'Sex', 'DCCI'], how='left')
        # Fill missing values (shouldn't happen, but be safe)
        for d in range(max_dose + 1):
            out[f'pop_dose{d}'] = out[f'pop_dose{d}'].fillna(0).astype(int)
    
    # Build Dead directly from aligned series
    out['Dead'] = out['dead'].astype(int)
    out['Dead_COVID'] = out['dead_covid'].astype(int)
    
    # Get enrollment week index for comparison
    enroll_week_idx = week_index.get(enroll_week_str, 0)
    
    if monte_carlo_mode:
        # For Monte Carlo: all weeks are post-enrollment, simple cumulative deaths by Dose
        out = out.sort_values(['Dose', 'WeekIdx'])
        out['cumDead_total'] = out.groupby(['Dose'])['dead'].cumsum()
        # CRITICAL FIX: shift() must be applied WITHIN each group, not across the entire DataFrame
        # The bug was: out.groupby(['Dose'])['dead'].cumsum().shift(fill_value=0)
        # This shifts across all rows, pulling values from dose 0 into dose 1, etc.
        # The fix: compute cumDead_prev separately for each dose group to ensure shift happens within group
        # IMPORTANT: out is already sorted by ['Dose', 'WeekIdx'], so rows for each dose are contiguous and in order
        out['cumDead_prev'] = 0
        for d in range(max_dose + 1):
            dose_mask = out['Dose'] == d
            if dose_mask.any():
                # Get the dead values for this dose group (already sorted by WeekIdx)
                dose_indices = out.index[dose_mask].tolist()
                dead_vals = out.loc[dose_indices, 'dead'].values
                # Compute cumulative sum
                cumsum_vals = np.cumsum(dead_vals)
                # Shift by 1, filling first value with 0 (cumDead_prev = deaths from previous weeks only)
                shifted_vals = np.concatenate([[0], cumsum_vals[:-1]])
                # Assign back using the same indices to ensure alignment
                out.loc[dose_indices, 'cumDead_prev'] = shifted_vals
    else:
        # For fixed cohorts, we only count deaths AFTER enrollment
        # Compute cumulative deaths per dose group, but reset at enrollment
        # Split into pre-enrollment and post-enrollment, then compute cumsum separately
        out_pre = out[out['WeekIdx'] < enroll_week_idx].copy()
        out_post = out[out['WeekIdx'] >= enroll_week_idx].copy()
        
        # Pre-enrollment: cumDead_total = 0 (not used, but set for completeness)
        out_pre['cumDead_total'] = 0
        
        # Post-enrollment: compute cumulative deaths from enrollment onwards
        out_post = out_post.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'WeekIdx'])
        out_post['cumDead_total'] = (
            out_post.groupby(['YearOfBirth', 'Sex', 'DCCI', 'Dose'])['dead']
              .cumsum()
        )
        
        # Combine back
        out = pd.concat([out_pre, out_post], ignore_index=True)
        out = out.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'WeekIdx'])
        
        # Compute cumulative prior-week deaths for post-enrollment weeks only
        # This represents deaths from enrollment up to (but not including) the current week
        out['cumDead_prev'] = 0
        out_post_sorted = out[out['WeekIdx'] >= enroll_week_idx].copy()
        out_post_sorted = out_post_sorted.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'WeekIdx'])
        cumDead_prev_post = (
            out_post_sorted.groupby(['YearOfBirth', 'Sex', 'DCCI', 'Dose'])['dead']
              .cumsum()
              .shift(fill_value=0)
        )
        # Map back to original index - need to match by position since we're using a subset
        post_enroll_indices = out[out['WeekIdx'] >= enroll_week_idx].index
        out.loc[post_enroll_indices, 'cumDead_prev'] = cumDead_prev_post.values
    
    # Initialize Alive column
    out['Alive'] = 0
    
    if monte_carlo_mode:
        # For Monte Carlo: all weeks are post-enrollment, simple Alive calculation
        # Alive = enrollment population - cumulative deaths from previous weeks
        for d in range(max_dose + 1):
            dose_mask = out['Dose'] == d
            if dose_mask.any():
                pop_dose_vals = out.loc[dose_mask, f'pop_dose{d}'].values
                cumDead_prev_vals = out.loc[dose_mask, 'cumDead_prev'].values
                alive_vals = np.maximum(pop_dose_vals - cumDead_prev_vals, 0).astype(int)
                out.loc[dose_mask, 'Alive'] = alive_vals
    else:
        # -------- Pre-enrollment weeks: Variable cohorts (track transitions) --------
        # For pre-enrollment weeks, compute Alive using transition-based approach
        # Everyone starts in dose 0, then transitions to higher doses as they get vaccinated
        pre_enroll_mask = out['WeekIdx'] < enroll_week_idx
        if pre_enroll_mask.any():
            # Get base total population (everyone starts in dose 0)
            base_total = a_copy.groupby(['YearOfBirth', 'Sex', 'DCCI']).size().rename('base_total').reset_index()
            
            # Build transition frames for pre-enrollment only
            # trans_0: everyone starts in dose 0 at the first week
            first_week_str = all_weeks[0]
            trans_0_pre = base_total.copy()
            trans_0_pre['ISOweekDied'] = first_week_str
            trans_0_pre['trans_0'] = trans_0_pre['base_total']
            
            # Track transitions into doses 1-max_dose (pre-enrollment only, since a_var has post-enrollment frozen)
            trans_frames_pre = [trans_0_pre[['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI', 'trans_0']]]
            dose_cols = [(1, 'Date_FirstDose'), (2, 'Date_SecondDose'), (3, 'Date_ThirdDose'), (4, 'Date_FourthDose')]
            for dose_num, col in dose_cols:
                wk = a_var[col].dt.strftime('%G-%V')
                trans_mask = (wk.notna()) & (pd.to_datetime(wk + '-1', format='%G-%V-%u', errors='coerce') < enrollment_date)
                if trans_mask.any():
                    trans_df = a_var.loc[trans_mask, ['YearOfBirth', 'Sex', 'DCCI']].copy()
                    trans_df['ISOweekDied'] = wk[trans_mask].values
                    # Collapse doses above max_dose into max_dose
                    target_dose = min(dose_num, max_dose)
                    trans_df[f'trans_{target_dose}'] = 1
                    trans_counts = trans_df.groupby(['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'])[f'trans_{target_dose}'].sum().reset_index()
                    trans_frames_pre.append(trans_counts)
            
            # Merge all transitions, summing when multiple transitions map to the same dose
            transitions_pre = trans_frames_pre[0]
            for tf in trans_frames_pre[1:]:
                transitions_pre = transitions_pre.merge(tf, on=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'], how='outer', suffixes=('', '_new'))
                # Sum columns that have _new suffix with original columns
                for d in range(max_dose + 1):
                    col_ = f'trans_{d}'
                    col_new = f'{col_}_new'
                    if col_new in transitions_pre.columns:
                        transitions_pre[col_] = transitions_pre[col_].fillna(0) + transitions_pre[col_new].fillna(0)
                        transitions_pre.drop(columns=[col_new], inplace=True)
            # Ensure all transition columns exist
            for d in range(max_dose + 1):
                col_ = f'trans_{d}'
                if col_ not in transitions_pre.columns:
                    transitions_pre[col_] = 0
            transitions_pre = transitions_pre.fillna(0)
            
            # Merge transitions into pre-enrollment grid (nodose level)
            out_pre_nodose = out[pre_enroll_mask][['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI', 'WeekIdx']].drop_duplicates().copy()
            trans_grid_pre = out_pre_nodose.merge(transitions_pre, on=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'], how='left')
            trans_cols = [f'trans_{d}' for d in range(max_dose + 1)]
            trans_grid_pre[trans_cols] = trans_grid_pre[trans_cols].fillna(0).astype(int)
            
            # Compute cumulative transitions (nodose level)
            trans_grid_pre = trans_grid_pre.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'WeekIdx'])
            cumT_pre = (
                trans_grid_pre.groupby(['YearOfBirth', 'Sex', 'DCCI'])[trans_cols]
                  .cumsum()
                  .shift(fill_value=0)
            )
            cumT_pre.columns = [f'cumT{d}_prev' for d in range(max_dose + 1)]
            trans_grid_pre = pd.concat([trans_grid_pre, cumT_pre], axis=1)
            
            # Merge cumulative transitions into full pre-enrollment grid (with doses)
            out_pre = out[pre_enroll_mask].copy()
            # Also merge base_total for first week initialization
            out_pre = out_pre.merge(base_total, on=['YearOfBirth', 'Sex', 'DCCI'], how='left')
            out_pre['base_total'] = out_pre['base_total'].fillna(0).astype(int)
            merge_cols = ['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'] + [f'cumT{d}_prev' for d in range(max_dose + 1)]
            out_pre = out_pre.merge(trans_grid_pre[merge_cols],
                                    on=['ISOweekDied', 'YearOfBirth', 'Sex', 'DCCI'], how='left')
            cumT_cols = [f'cumT{d}_prev' for d in range(max_dose + 1)]
            out_pre[cumT_cols] = out_pre[cumT_cols].fillna(0).astype(int)
            
            # Compute cumulative deaths (pre-enrollment, variable cohorts)
            out_pre = out_pre.sort_values(['YearOfBirth', 'Sex', 'DCCI', 'Dose', 'WeekIdx'])
            out_pre['cumDead_var_prev'] = (
                out_pre.groupby(['YearOfBirth', 'Sex', 'DCCI', 'Dose'])['dead']
                  .cumsum()
                  .shift(fill_value=0)
            )
            
            # Compute Alive for pre-enrollment using transition formula
            # Dose 0: started in dose 0 - transitions to dose 1 - deaths
            # Dose 1: transitions to dose 1 - transitions to dose 2 - deaths
            # etc.
            # For max_dose: transitions to max_dose - deaths (no transitions out)
            dose_vals_pre = out_pre['Dose'].values
            alive_pre = np.zeros(len(out_pre))
            for d in range(max_dose + 1):
                mask = dose_vals_pre == d
                if mask.any():
                    if d < max_dose:
                        # Transition in - transition out - deaths
                        alive_pre[mask] = (out_pre.loc[mask, f'cumT{d}_prev'] - 
                                          out_pre.loc[mask, f'cumT{d+1}_prev'] - 
                                          out_pre.loc[mask, 'cumDead_var_prev']).values
                    else:
                        # Max dose: transition in - deaths (no transitions out)
                        alive_pre[mask] = (out_pre.loc[mask, f'cumT{d}_prev'] - 
                                          out_pre.loc[mask, 'cumDead_var_prev']).values
            out_pre['Alive'] = np.maximum(alive_pre, 0).astype(int)
            
            # Ensure first week starts correctly (everyone in dose 0)
            # For first week, cumT0_prev is 0 (shifted), but trans_0 has the initial population
            first_week_mask_pre = out_pre['ISOweekDied'] == first_week_str
            # Get base_total for first week dose 0
            first_week_base = out_pre.loc[first_week_mask_pre & (out_pre['Dose'] == 0), 'base_total'].values
            if len(first_week_base) > 0:
                out_pre.loc[first_week_mask_pre & (out_pre['Dose'] == 0), 'Alive'] = first_week_base
            out_pre.loc[first_week_mask_pre & (out_pre['Dose'] != 0), 'Alive'] = 0
            
            # Map back to main dataframe
            out.loc[pre_enroll_mask, 'Alive'] = out_pre['Alive'].values
    
    if not monte_carlo_mode:
        # For enrollment week: Alive = enrollment population count (no deaths yet)
        enroll_mask = out['ISOweekDied'] == enroll_week_str
        for d in range(max_dose + 1):
            out.loc[enroll_mask & (out['Dose'] == d), 'Alive'] = out.loc[enroll_mask & (out['Dose'] == d), f'pop_dose{d}'].values
        
        # For post-enrollment weeks: Alive = enrollment_count - cumulative_deaths from previous weeks
        # Use cumDead_prev (deaths up to but not including this week) since Alive represents start-of-week population
        post_enroll_mask = out['WeekIdx'] > enroll_week_idx
        dose_vals = out.loc[post_enroll_mask, 'Dose'].values
        
        # Get enrollment counts for each row
        enroll_counts = np.zeros(len(out.loc[post_enroll_mask]))
        for d in range(max_dose + 1):
            dose_mask = dose_vals == d
            if dose_mask.any():
                enroll_counts[dose_mask] = out.loc[post_enroll_mask & (out['Dose'] == d), f'pop_dose{d}'].values
        
        # Compute Alive: enrollment_count - cumulative_deaths from previous weeks (not including this week)
        # This represents people alive at the START of the week
        cum_deaths_prev = out.loc[post_enroll_mask, 'cumDead_prev'].values
        out.loc[post_enroll_mask, 'Alive'] = np.maximum(enroll_counts - cum_deaths_prev, 0).astype(int)
    
    _progress("Cleaning up columns...")
    # Drop helper columns
    cols_to_drop = [
        'dead',
        'dead_covid',
        'dead_pre','dead_post',
        'cumDead_total',  # Keep cumDead_prev for compatibility, drop cumDead_total
    ] + [f'pop_dose{d}' for d in range(max_dose + 1)] + [  # Enrollment counts, no longer needed
        'WeekIdx',
        'is_post_enroll'  # Helper column, no longer needed
    ]
    out.drop(columns=[c for c in cols_to_drop if c in out.columns], inplace=True)

    # Ensure Dead_COVID column is placed right after Dead
    if 'Dead_COVID' in out.columns and 'Dead' in out.columns:
        cols = list(out.columns)
        cols.remove('Dead_COVID')
        insert_at = cols.index('Dead') + 1
        cols.insert(insert_at, 'Dead_COVID')
        out = out[cols]

    # Convert DateDied back to string for Excel after computations
    out['DateDied'] = out['DateDied'].dt.strftime('%Y-%m-%d')
    
    return out, alive_at_enroll, all_weeks, week_index, pop_base

def process_monte_carlo_iteration(args):
    """
    Process a single Monte Carlo iteration.
    Writes result to a temporary CSV file and returns (iteration_number, temp_file_path) tuple.
    
    Args:
        args: tuple of (iteration, master_count, enrollment_date, enroll_week_str, enroll_date_str, temp_dir)
    
    Note: Accesses _master_monte_carlo_global via fork() copy-on-write (no pickling!)
    """
    iteration, master_count, enrollment_date, enroll_week_str, enroll_date_str, temp_dir = args
    
    # Access module-level variable via fork() copy-on-write (no pickling!)
    master_monte_carlo = _master_monte_carlo_global
    if master_monte_carlo is None:
        raise RuntimeError("_master_monte_carlo_global is None - fork() may not have worked correctly")
    
    import datetime
    import pandas as pd
    
    print(f"[Iteration {iteration}] Starting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    try:
        # Iteration #1 uses the full dataset without sampling (for validation/comparison)
        # All other iterations sample WITH REPLACEMENT from master_monte_carlo
        if iteration == 1:
            print(f"[Iteration {iteration}] Using full dataset without sampling ({len(master_monte_carlo)} records)...", flush=True)
            a_copy = master_monte_carlo.copy()
        else:
            # Sample WITH REPLACEMENT from master_monte_carlo
            print(f"[Iteration {iteration}] Sampling {master_count} records...", flush=True)
            sampled_records = master_monte_carlo.sample(n=master_count, replace=True, random_state=iteration)
            # Use sampled records as a_copy for this iteration
            a_copy = sampled_records.copy()
        
        print(f"[Iteration {iteration}] Processing enrollment data (input: {len(a_copy)} records)...", flush=True)
        # Process using shared helper function
        # For Monte Carlo mode, cap dose groups to match the selected enrollment cohort
        try:
            out, alive_at_enroll, all_weeks, week_index, pop_base = process_enrollment_data(
                a_copy,
                enrollment_date,
                enroll_week_str,
                max_dose=int(MC_MAX_DOSE_EFFECTIVE) if MC_MAX_DOSE_EFFECTIVE is not None else 3,
                monte_carlo_mode=True,
                iteration=iteration,
            )
            print(f"[Iteration {iteration}] Enrollment data processed (output: {len(out)} rows)", flush=True)
        except Exception as e:
            print(f"[Iteration {iteration}] ERROR in process_enrollment_data: {e}", flush=True)
            raise
        
        print(f"[Iteration {iteration}] Aggregating results...", flush=True)
        
        # Monte Carlo mode: Filter to YearOfBirth=-2 (all ages) and simplify columns
        # Filter to all-ages cohort only (YearOfBirth == -2)
        # Aggregate across all YearOfBirth, Sex, DCCI to create all-ages cohort
        out_mc_agg = out.groupby(['ISOweekDied', 'Dose', 'DateDied']).agg({
            'Alive': 'sum',
            'Dead': 'sum'
        }).reset_index()
        out_mc_agg['YearOfBirth'] = -2  # Mark as all-ages
        
        # Select only required columns
        out_mc_final = out_mc_agg[['ISOweekDied', 'Dose', 'DateDied', 'Dead', 'Alive']].copy()
        
        print(f"[Iteration {iteration}] Writing CSV file...", flush=True)
        # Write to temporary CSV file instead of returning DataFrame
        temp_file = os.path.join(temp_dir, f"mc_iteration_{iteration}.csv")
        out_mc_final.to_csv(temp_file, index=False)
        
        print(f"[Iteration {iteration}] Writing marker file...", flush=True)
        # Write completion marker file (avoids returning values through multiprocessing queue)
        marker_file = os.path.join(temp_dir, f"mc_iteration_{iteration}.done")
        with open(marker_file, 'w') as f:
            f.write(temp_file)
        
        print(f"[Iteration {iteration}] Completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    except Exception as e:
        import traceback
        print(f"[Iteration {iteration}] ERROR: {e}")
        print(f"[Iteration {iteration}] Traceback: {traceback.format_exc()}")
        raise
    
    # Return None to avoid any serialization issues
    return None

# Main enrollment processing loop
# In Monte Carlo mode, use parallel processing for iterations
if MONTE_CARLO_MODE:
    # MC mode: process iterations in parallel
    import datetime
    print(f"\n{'='*60}", flush=True)
    print(f"Starting Monte Carlo processing with {MC_ITERATIONS} iterations using {MC_THREADS} parallel processes", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Prepare arguments for parallel processing
    # We only process one enrollment date in MC mode (selected via MC_ENROLLMENT_DATE)
    enroll_date_str = enrollment_dates[0]
    enrollment_date = pd.to_datetime(enroll_date_str + '-1', format='%G-%V-%u', errors='coerce')
    enroll_week_str = enrollment_date.strftime('%G-%V')
    
    # Create temporary directory for CSV files (avoids serializing DataFrames through multiprocessing queue)
    temp_dir = tempfile.mkdtemp(prefix='kcor_mc_')
    print(f"Using temporary directory: {temp_dir}")
    
    # CRITICAL FIX: Use multiprocessing with 'fork' start method (Linux only)
    # With fork(), child processes get copy-on-write access to parent's memory
    # We access master_monte_carlo via closure (not as argument) to avoid pickling
    # Prepare arguments for each iteration (ONLY pass small args, DataFrame accessed via closure)
    # CRITICAL: Only create args for the requested number of iterations
    iteration_args = [
        (iteration, master_count, enrollment_date, enroll_week_str, enroll_date_str, temp_dir)
        for iteration in range(1, MC_ITERATIONS + 1)
    ]
    
    # Verify we're not exceeding the requested iterations
    if len(iteration_args) != MC_ITERATIONS:
        print(f"ERROR: Created {len(iteration_args)} iteration args but MC_ITERATIONS={MC_ITERATIONS}")
        sys.exit(1)
    
    # Process iterations in parallel using multiprocessing with fork (copy-on-write, no pickling!)
    print(f"Processing {MC_ITERATIONS} iterations in parallel using {MC_THREADS} processes...", flush=True)
    print(f"  Using multiprocessing with 'fork' - DataFrame accessed via copy-on-write (no pickling!)", flush=True)
    start_time = datetime.datetime.now()
    
    # CRITICAL FIX: Use multiprocessing with 'fork' start method (Linux/WSL)
    # With fork(), child processes get copy-on-write access to parent's memory
    # We pass master_monte_carlo via partial/closure to avoid pickling
    temp_files = {}
    completed_count = 0
    
    # Use 'fork' start method on Linux (copy-on-write, no pickling needed!)
    # On Windows this will fail, but user is on WSL/Linux
    try:
        ctx = get_context('fork')
        print(f"  Using 'fork' start method - DataFrame accessed via copy-on-write (no pickling!)")
    except ValueError:
        # Fallback to default if 'fork' not available
        print("  WARNING: 'fork' start method not available, using default (may require pickling)")
        ctx = get_context()
    
    with ctx.Pool(processes=MC_THREADS) as pool:
        # Submit all tasks - master_monte_carlo accessed via module-level variable (copy-on-write, no pickling!)
        async_results = []
        for args in iteration_args:
            iteration = args[0]
            async_result = pool.apply_async(process_monte_carlo_iteration, (args,))
            async_results.append((iteration, async_result))
        
        # Wait for all tasks to complete (process results as they finish, not in order)
        remaining = dict(async_results)  # {iteration: async_result}
        last_progress_time = datetime.datetime.now()
        last_completed_count = 0
        
        while remaining:
            # Check which tasks are ready (non-blocking check)
            ready_iterations = []
            import time as time_module
            for iteration, async_result in list(remaining.items()):
                try:
                    is_ready = async_result.ready()
                    if is_ready:
                        ready_iterations.append(iteration)
                except Exception as e:
                    pass
            
            # Process ready tasks
            for iteration in ready_iterations:
                async_result = remaining.pop(iteration)
                try:
                    # Get result (should be immediate since ready() returned True)
                    # Use timeout to detect if get() hangs (shouldn't happen if ready() is True, but be safe)
                    try:
                        async_result.get(timeout=5)  # Increased timeout to 5s to be safe
                    except Exception as get_error:
                        # Even if get() fails, check if marker file exists (process might have completed)
                        marker_file = os.path.join(temp_dir, f"mc_iteration_{iteration}.done")
                        if os.path.exists(marker_file):
                            # Process completed but get() failed - continue processing
                            pass
                        else:
                            # Process didn't complete - re-raise to handle in outer exception handler
                            raise
                    completed_count += 1
                    
                    # Read the temp file path from marker file
                    marker_file = os.path.join(temp_dir, f"mc_iteration_{iteration}.done")
                    if os.path.exists(marker_file):
                        with open(marker_file, 'r') as f:
                            temp_file = f.read().strip()
                        temp_files[iteration] = temp_file
                        os.remove(marker_file)  # Clean up marker
                    else:
                        # Fallback: construct expected temp file path
                        temp_file = os.path.join(temp_dir, f"mc_iteration_{iteration}.csv")
                        if os.path.exists(temp_file):
                            temp_files[iteration] = temp_file
                    
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    elapsed = (datetime.datetime.now() - start_time).total_seconds()
                    print(f"  [Iteration {iteration}] Completed - {completed_count}/{MC_ITERATIONS} complete at {timestamp} (elapsed: {elapsed:.1f}s)", flush=True)
                    last_progress_time = datetime.datetime.now()
                    last_completed_count = completed_count
                except Exception as e:
                    import traceback
                    print(f"  ERROR: Iteration {iteration} failed: {e}", flush=True)
                    print(f"  Traceback: {traceback.format_exc()}", flush=True)
                    # Try to read partial results if CSV exists
                    temp_file = os.path.join(temp_dir, f"mc_iteration_{iteration}.csv")
                    if os.path.exists(temp_file):
                        temp_files[iteration] = temp_file
            
            # If no tasks are ready, wait a bit before checking again
            if remaining and not ready_iterations:
                import time
                time.sleep(0.5)  # Check every 0.5 seconds
                
                # Warn if no progress for a while
                time_since_progress = (datetime.datetime.now() - last_progress_time).total_seconds()
                if time_since_progress > 300:  # 5 minutes with no progress
                    stuck_iterations = list(remaining.keys())
                    print(f"  WARNING: No progress for {time_since_progress:.0f} seconds. Still waiting for iterations: {stuck_iterations}", flush=True)
                    
                    # After 10 minutes of no progress, terminate stuck processes and exit with error
                    if time_since_progress > 600:  # 10 minutes
                        print(f"  ERROR: Processes stuck for {time_since_progress:.0f} seconds. Terminating pool and exiting.", flush=True)
                        # Break out of loop - pool will be terminated by context manager
                        break
                    
                    last_progress_time = datetime.datetime.now()  # Reset to avoid spam
    
    end_time = datetime.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\nCompleted all {MC_ITERATIONS} iterations in {elapsed:.1f} seconds ({elapsed/MC_ITERATIONS:.2f} seconds per iteration)")
    
    # Now read CSV files and write to Excel in order
    print(f"\nWriting results to Excel...")
    for iteration in sorted(temp_files.keys()):
        temp_file = temp_files[iteration]
        result_df = pd.read_csv(temp_file)
        sheet_name = str(iteration)
        result_df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
        print(f"  Wrote iteration {iteration} to sheet '{sheet_name}' ({len(result_df)} rows)")
        # Clean up temp file immediately
        os.remove(temp_file)
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        # Directory might not be empty if some files weren't cleaned up
        pass
    
    print(f"\n{'='*60}")
    print(f"Monte Carlo processing complete!")
    print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"Monte Carlo processing complete!")
    print(f"{'='*60}")
    
else:
    # Normal mode processing (Monte Carlo mode is handled separately above)
    for enroll_date_str in enrollment_dates:
        # Parse ISO week string as Monday of that week
        import datetime
        print(f"Processing enrollment date {enroll_date_str} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        enrollment_date = pd.to_datetime(enroll_date_str + '-1', format='%G-%V-%u', errors='coerce')
        enroll_week_str = enrollment_date.strftime('%G-%V')
        
        print(f"  Creating copy of dataset ({len(a)} records)...")
        a_copy = a.copy()
        # Restrict processing to birth years within [1920, 2005] inclusive, but keep -1 (unknown)
        before = len(a_copy)
        a_copy = a_copy[((a_copy['YearOfBirth'] >= 1920) & (a_copy['YearOfBirth'] <= 2005)) | (a_copy['YearOfBirth'] == -1)].copy()
        after = len(a_copy)
        print(f"  Filtered YearOfBirth to 1920-2005: kept {after}/{before} records")
        print(f"  Keeping all records including deaths before enrollment date...")
        # Keep all individuals, including those who died before the enrollment date
        # This allows us to see pre-enrollment deaths in their correct dose groups
        print(f"  Records in analysis: {len(a_copy)}")
        
        # Process using shared helper function
        print(f"  Processing enrollment date {enroll_date_str}...")
        out, alive_at_enroll, all_weeks, week_index, pop_base = process_enrollment_data(
            a_copy, enrollment_date, enroll_week_str
        )
        
        # Write to Excel sheet
        sheet_name = enroll_date_str.replace('-', '_')
        print(f"  Writing to Excel sheet...")
        out.to_excel(excel_writer, sheet_name=sheet_name, index=False)
        print(f"  Wrote sheet '{sheet_name}' ({len(out)} rows)")
        
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
                # Get a_var from the helper function's scope - we need to reconstruct it
                # For now, use a_copy and filter by manufacturer
                a_var_local = a_copy.copy()
                _bypass_freeze = str(os.environ.get('BYPASS_FREEZE','')).strip().lower() in ('1','true','yes')
                if not _bypass_freeze:
                    for _col in ['Date_FirstDose','Date_SecondDose','Date_ThirdDose','Date_FourthDose']:
                        a_var_local.loc[a_var_local[_col] >= enrollment_date, _col] = pd.NaT
                
                sub = a_var_local[(a_var_local[mcol] == mfg_label)].copy()
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
        
        print(f"Added sheet {sheet_name} to {excel_out_path}")
        print(f"Completed enrollment date {enroll_date_str} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

# Save the Excel file after all sheets are added
excel_writer.close()
if MONTE_CARLO_MODE:
    print(f"Wrote all {MC_ITERATIONS} Monte Carlo iterations to {excel_out_path}")
else:
    print(f"Wrote all dose group CMRs to {excel_out_path}")

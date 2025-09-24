#!/usr/bin/env python3
"""
KCOR (Kirsch Cumulative Outcomes Ratio) Analysis Script v4.4

This script analyzes mortality data to compute KCOR values, which are ratios of cumulative
hazards between different dose groups, normalized to 1 at a baseline period.

METHODOLOGY OVERVIEW:
====================

1. DATA PREPROCESSING:
   - Filter data by enrollment date (derived from sheet name, e.g., "2021_24" = 2021, week 24)
   - Aggregate mortality data across sexes for each (YearOfBirth, Dose, DateDied) combination
   - Apply 8-week centered moving average to raw mortality rates for smoothing

2. SLOPE CALCULATION (Lookup Table Method):
   - Uses predefined anchor points (e.g., weeks 53 and 114 from enrollment for 2021_24)
   - For each anchor point, creates a ±2 week window (5 points total)
   - Calculates geometric mean of smoothed MR values within each window
   - Slope = ln(B/A) / T where A = geometric mean at first anchor, B = geometric mean at second anchor
   - Same anchor points used for all doses to ensure comparability
   - Anchor dates chosen during "quiet periods" with no differential events (COVID waves, policy changes, etc.)

3. MORTALITY RATE ADJUSTMENT:
   - Applies exponential slope removal: MR_adj = MR × exp(-slope × (t - t0))
   - t0 = baseline week (typically week 4) where KCOR is normalized to 1
   - Each dose-age combination gets its own slope for adjustment

4. KCOR COMPUTATION (v4.1+):
   - KCOR = (CH_num / CH_den) / (CH_num_baseline / CH_den_baseline)
   - CH = slope-corrected cumulative hazard (mathematically exact)
   - Step 1: Apply slope correction to individual MR: MR_adj = MR × exp(-slope × (t - t0))
   - Step 2: Apply discrete cumulative-hazard transform: hazard = -ln(1 - MR_adj)
   - Step 3: Calculate cumulative hazard: CH = cumsum(hazard)
   - Baseline values taken at week 4 (or first available week)
   - Results in KCOR = 1 at baseline, showing relative risk over time

5. UNCERTAINTY QUANTIFICATION:
   - 95% confidence intervals using proper uncertainty propagation
   - Accounts for baseline uncertainty and current time uncertainty
   - Uses binomial variance approximation: Var[D] ≈ D for death counts
   - CI bounds calculated on log scale then exponentiated for proper asymmetry

6. AGE STANDARDIZATION (Option 2+ - v4.2):
   - ASMR pooling across age groups using expected-deaths weights
   - Weights = smoothed mortality rate × person-time in quiet baseline window
   - Formula: w_a ∝ h_a × PT_a(W) where h_a = smoothed mean MR in quiet window W
   - Properly weights elderly age groups who contribute most to death burden
   - Provides population-level KCOR estimates reflecting actual mortality impact
   - FIXED in v4.2: Previous person-time only weighting over-weighted young people

KEY ASSUMPTIONS:
- Mortality rates follow exponential trends during observation period
- No differential events affect dose groups differently during anchor periods
- Baseline period (week 4) represents "normal" conditions
- Person-time = Alive (survivor function approximation)

INPUT WORKBOOK SCHEMA per sheet (e.g., '2021-13', '2021_24', '2022_06', ...):
    ISOweekDied, DateDied, YearOfBirth, Sex, Dose, Alive, Dead

OUTPUTS (two main sheets):
    - "by_dose": Individual dose curves with complete methodology transparency including:
      EnrollmentDate, Date, YearOfBirth, Dose, ISOweek, Dead, Alive, MR, MR_adj, CH, CH_actual, Hazard, 
      Slope, Scale_Factor, Cumu_Adj_Deaths, Cumu_Unadj_Deaths, Cumu_Person_Time, 
      Smoothed_Raw_MR, Smoothed_Adjusted_MR, Time_Index
    - "dose_pairs": KCOR values for all dose comparisons with complete methodology transparency:
      EnrollmentDate, ISOweekDied, Date, YearOfBirth (0 = ASMR pooled, -1 = unknown age), Dose_num, Dose_den,
      KCOR, CI_lower, CI_upper, MR_num, MR_adj_num, CH_num, CH_actual_num, hazard_num, 
      slope_num, scale_factor_num, MR_smooth_num, t_num, MR_den, MR_adj_den, CH_den, 
      CH_actual_den, hazard_den, slope_den, scale_factor_den, MR_smooth_den, t_den

USAGE:
    python KCOR.py KCOR_output.xlsx KCOR_processed_REAL.xlsx
    
DEPENDENCIES:
    pip install pandas numpy openpyxl

This approach provides robust, interpretable estimates of relative mortality risk
between vaccination groups while accounting for underlying time trends. Version 4.2
includes corrected ASMR pooling using expected-deaths weights that properly reflect
actual mortality burden rather than population size.
"""
import sys
import math
import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import logging
from datetime import datetime, timedelta
import tempfile
import shutil

# Dependencies: pandas, numpy, openpyxl

# Sensitivity analysis flag (environment-driven)
def _is_sa_mode() -> bool:
    v = os.environ.get("SENSITIVITY_ANALYSIS", "")
    return str(v).strip().lower() in ("1", "true", "yes")

def setup_dual_output(output_dir, log_filename="KCOR_summary.log"):
    """Set up dual output to both console and file."""
    # Create log file path
    log_file = os.path.join(output_dir, log_filename)
    
    # Open file for writing
    log_file_handle = open(log_file, 'w', encoding='utf-8')
    
    def dual_print(*args, **kwargs):
        """Print to both console and file."""
        message = ' '.join(str(arg) for arg in args)
        print(message, **kwargs)  # Console output
        print(message, file=log_file_handle, **kwargs)  # File output
        log_file_handle.flush()  # Ensure immediate write
    
    return dual_print, log_file_handle

# ---------------- Configuration Parameters ----------------
# Version information
VERSION = "v4.4"                # KCOR version number

# Version History:
# v4.0 - Initial implementation with slope correction applied to individual MRs then cumulated
# v4.1 - Enhanced with discrete cumulative-hazard transform for mathematical exactness
#        - Changed from simple cumsum(MR_adj) to cumsum(-ln(1 - MR_adj))
#        - Removes small-rate approximation limitation
#        - More robust for any mortality rate magnitude
# v4.2 - Fixed ASMR pooling with Option 2+ expected-deaths weights
#        - Changed from person-time weights to expected-deaths weights: w_a ∝ h_a × PT_a(W)
#        - Properly weights elderly age groups who contribute most to death burden
#        - Uses pooled quiet baseline window with smoothed mortality rates
#        - ASMR now reflects actual mortality impact rather than population size
# v4.3 - Added fine-tuning parameters for lowering the baseline value if the final KCOR value is below the minimum
#        - Implements KCOR scaling based on FINAL_KCOR_DATE and FINAL_KCOR_MIN parameters
#        - Corrects for baseline normalization issues where unsafe vaccines create artificially high baseline mortality rates
# v4.4 - Added enrollment cohort 2022_47 with dose comparisons 4 vs 3,2,1,0
#        - Default processing now includes four cohorts: 2021_13, 2021_24, 2022_06, 2022_47
#        - Added slope lookup entry for 2022_47 (legacy; dynamic anchors now default)

# Core KCOR methodology parameters
# KCOR baseline normalization week (KCOR == 1 at this week; weeks counted from enrollment t=0)
KCOR_NORMALIZATION_WEEK = 4     # default week 4 which is the 5th week of cumulated data. See also DYNAMIC_HVE_SKIP_WEEKS.
AGE_RANGE = 10                  # Bucket size for YearOfBirth aggregation (e.g., 10 -> 1920, 1930, ..., 2000)
SLOPE_ANCHOR_T = 0              # Enrollment week index for slope anchoring
EPS = 1e-12                     # Numerical floor to avoid log(0) and division by zero
DYNAMIC_HVE_SKIP_WEEKS = 0      # Start accumulating hazards/statistics from this week index (0 = from enrollment)
MR_DISPLAY_SCALE = 52 * 1e5     # Display-only scaling of MR columns (annualized per 100,000)

# KCOR normalization fine-tuning parameters
FINAL_KCOR_MIN = 0              # Setting to 0 DISABLES scaling based on the final value
FINAL_KCOR_DATE = "4/1/24"      # Date to check for KCOR scaling (MM/DD/YY format)

# Removed MAX_DATE_FOR_SLOPE: dynamic anchors specify calendar dates directly

# SLOPE CALCULATION METHODOLOGY (Dynamic by default):
# Anchors are chosen from QUIET_ANCHOR_ISO_WEEKS subject to gap/separation rules.
SLOPE_WINDOW_SIZE = 2  # Window size: use anchor point ± 2 weeks (5 points total) for geometric mean

# Dynamic anchor selection (preferred over lookup in normal runs)
# Candidates are calendar quiet periods (ISO year-week). The first anchor must be
# at least MIN_ANCHOR_GAP_WEEKS after enrollment; the second must be at least
# MIN_ANCHOR_SEPARATION_WEEKS after the first. Both are configurable.
QUIET_ANCHOR_ISO_WEEKS = ["2022-25", "2023-28", "2024-15"]
MIN_ANCHOR_GAP_WEEKS = 26
MIN_ANCHOR_SEPARATION_WEEKS = 39

# Reporting date lookup for KCOR summary/console per cohort (sheet name)
# For the first three cohorts, use end of 2022; for 2022_47, use one year later (end of 2023)
KCOR_REPORTING_DATE = {
    '2021-13': '2022-12-31',
    '2021_13': '2022-12-31',
    '2021_24': '2022-12-31',
    '2022_06': '2022-12-31',
    '2022_47': '2023-12-31',
}

# DATA SMOOTHING:
# Apply moving average before slope calculation to reduce noise and improve stability
MA_TOTAL_LENGTH = 8  # Total length of centered moving average (8 weeks = 4 weeks on either side)
CENTERED = True      # Use centered MA (4 weeks before + 4 weeks after each point) to minimize lag

# Processing parameters
YEAR_RANGE = (1920, 2000)       # Process age groups from start to end year (inclusive)
ENROLLMENT_DATES = ['2021_13', '2021_24', '2022_06', '2022_47']  # List of enrollment dates (sheet names) to process (set to None to process all)
DEBUG_DOSE_PAIR_ONLY = None  # Only process this dose pair (set to None to process all)
DEBUG_VERBOSE = True            # Print detailed debugging info for each date
# ----------------------------------------------------------

# Optional overrides via environment for sensitivity/plumbing without CLI changes
# SA_COHORTS: comma-separated list of sheet names, e.g., "2021_24,2022_06"
# SA_DOSE_PAIRS: semicolon-separated list of pairs as a,b; e.g., "1,0;2,0"
# SA_YOB: "0" for ASMR only, or range "start,end,step", or list "y1,y2,y3"
OVERRIDE_DOSE_PAIRS = None
OVERRIDE_YOBS = None

def _parse_env_dose_pairs(value: str):
    pairs = []
    for part in value.split(';'):
        part = part.strip()
        if not part:
            continue
        ab = [x.strip() for x in part.split(',')]
        if len(ab) != 2:
            continue
        try:
            pairs.append((int(ab[0]), int(ab[1])))
        except Exception:
            continue
    return pairs if pairs else None

def _parse_sa_yob(value: str):
    try:
        s = value.strip()
        if not s:
            return None
        # Range form: start,end,step
        parts = [p.strip() for p in s.split(',')]
        if len(parts) == 1:
            return [int(parts[0])]
        if len(parts) == 3:
            start, end, step = map(int, parts)
            if step <= 0:
                step = 1
            # inclusive end
            vals = list(range(start, end + 1, step))
            return vals
        # Else treat as list
        return [int(p) for p in parts]
    except Exception:
        return None

# Apply environment overrides if provided
try:
    _env_cohorts = os.environ.get('SA_COHORTS')
    if _env_cohorts:
        ENROLLMENT_DATES = [x.strip() for x in _env_cohorts.split(',') if x.strip()]
    _env_pairs = os.environ.get('SA_DOSE_PAIRS')
    if _env_pairs:
        OVERRIDE_DOSE_PAIRS = _parse_env_dose_pairs(_env_pairs)
        if DEBUG_VERBOSE and OVERRIDE_DOSE_PAIRS:
            print(f"[DEBUG] Overriding dose pairs via SA_DOSE_PAIRS: {OVERRIDE_DOSE_PAIRS}")
    _env_yob = os.environ.get('SA_YOB')
    if _env_yob:
        OVERRIDE_YOBS = _parse_sa_yob(_env_yob)
        if DEBUG_VERBOSE and OVERRIDE_YOBS:
            print(f"[DEBUG] Overriding YOB selection via SA_YOB: {OVERRIDE_YOBS}")
    # Core method knobs
    _env_anchor = os.environ.get('SA_ANCHOR_WEEKS')
    if _env_anchor:
        KCOR_NORMALIZATION_WEEK = int(_env_anchor)
        if DEBUG_VERBOSE:
            print(f"[DEBUG] Overriding KCOR_NORMALIZATION_WEEK via SA_ANCHOR_WEEKS: {KCOR_NORMALIZATION_WEEK}")
    _env_ma = os.environ.get('SA_MA_TOTAL_LENGTH')
    if _env_ma:
        MA_TOTAL_LENGTH = int(_env_ma)
        if DEBUG_VERBOSE:
            print(f"[DEBUG] Overriding MA_TOTAL_LENGTH via SA_MA_TOTAL_LENGTH: {MA_TOTAL_LENGTH}")
    _env_centered = os.environ.get('SA_CENTERED')
    if _env_centered:
        CENTERED = str(_env_centered).strip().lower() in ("1","true","yes")
        if DEBUG_VERBOSE:
            print(f"[DEBUG] Overriding CENTERED via SA_CENTERED: {CENTERED}")
    _env_win = os.environ.get('SA_SLOPE_WINDOW_SIZE')
    if _env_win:
        SLOPE_WINDOW_SIZE = int(_env_win)
        if DEBUG_VERBOSE:
            print(f"[DEBUG] Overriding SLOPE_WINDOW_SIZE via SA_SLOPE_WINDOW_SIZE: {SLOPE_WINDOW_SIZE}")
    _env_final_min = os.environ.get('SA_FINAL_KCOR_MIN')
    if _env_final_min:
        # If a range/list is provided (e.g., "0,1,1"), defer handling to SA loop
        try:
            if ',' not in _env_final_min.strip():
                FINAL_KCOR_MIN = float(_env_final_min)
                if DEBUG_VERBOSE:
                    print(f"[DEBUG] Overriding FINAL_KCOR_MIN via SA_FINAL_KCOR_MIN: {FINAL_KCOR_MIN}")
        except Exception:
            pass
    _env_final_date = os.environ.get('SA_FINAL_KCOR_DATE')
    if _env_final_date:
        FINAL_KCOR_DATE = _env_final_date
        if DEBUG_VERBOSE:
            print(f"[DEBUG] Overriding FINAL_KCOR_DATE via SA_FINAL_KCOR_DATE: {FINAL_KCOR_DATE}")
except Exception:
    pass

# Helpers to parse numeric triplet ranges like "start,end,step"
def _parse_triplet_range(value: str):
    try:
        parts = [p.strip() for p in str(value).split(',') if p.strip()]
        if len(parts) == 3:
            start, end, step = map(int, parts)
            if step <= 0:
                step = 1
            if start <= end:
                return list(range(start, end + 1, step))
            else:
                return list(range(start, end - 1, -step))
        elif len(parts) == 1:
            return [int(parts[0])]
        else:
            # interpret as list of ints
            return [int(p) for p in parts]
    except Exception:
        return None

def safe_log(x, eps=EPS):
    """Safe logarithm with clipping to avoid log(0) or log(negative)."""
    return np.log(np.clip(x, eps, None))

def safe_exp(x, max_val=1e6):
    """Safe exponential with clipping to prevent overflow."""
    # Clip the input to prevent overflow, not the output
    clipped_x = np.clip(x, -np.log(max_val), np.log(max_val))
    return np.exp(clipped_x)


def get_dose_pairs(sheet_name):
    """
    Get dose pairs based on sheet name.
    
    This function implements sheet-specific dose configurations to handle different
    vaccination schedules and data availability across different enrollment periods.
    
    The dose pairs are designed so that the lower dose is always the denominator,
    providing consistent relative risk comparisons across different vaccination levels.
    """
    # Global override for sensitivity: if provided, use it for all sheets
    if OVERRIDE_DOSE_PAIRS is not None:
        return OVERRIDE_DOSE_PAIRS

    if sheet_name in ("2021-13", "2021_13"):
        # Early 2021 sheet: max dose is 2, only doses 0, 1, 2
        return [(1,0), (2,0), (2,1)]
    elif sheet_name == "2021_24":
        # Mid 2021 sheet: max dose is 2, only doses 0, 1, 2
        return [(1,0), (2,0), (2,1)]
    elif sheet_name == "2022_06":
        # 2022 sheet: includes dose 3 comparisons
        return [(1,0), (2,0), (2,1), (3,2), (3,0)]
    elif sheet_name == "2022_47":
        # Late 2022 sheet: focus on 4th dose comparisons per v4.4 requirements
        return [(4,3), (4,2), (4,1), (4,0)]
    else:
        # Default: max dose is 2
        return [(1,0), (2,0), (2,1)]

def compute_group_slopes_lookup(df, sheet_name, logger=None):
    """[Deprecated] Lookup-based slopes removed. Use compute_group_slopes_dynamic instead."""
    slopes = {}
    for (yob, dose), g in df.groupby(["YearOfBirth", "Dose"], sort=False):
        slopes[(yob, dose)] = 0.0
        return slopes
    
def _parse_iso_year_week(s: str):
    from datetime import datetime
    y, w = s.split("-")
    # ISO week to date: Monday of that ISO week
    return datetime.fromisocalendar(int(y), int(w), 1)

def select_dynamic_anchor_offsets(enrollment_date, df_dates):
    """Select offsets (weeks from enrollment) for two anchors based on calendar quiet weeks.

    - First anchor: first candidate date ≥ enrollment_date + MIN_ANCHOR_GAP_WEEKS
    - Second anchor: first candidate ≥ first_anchor + MIN_ANCHOR_SEPARATION_WEEKS
    Returns (offset1, offset2) in weeks. If not found, returns (None, None).
    """
    # 'from datetime import timedelta' already declared at top of function
    candidates = [_parse_iso_year_week(s) for s in QUIET_ANCHOR_ISO_WEEKS]
    # Snap candidate to the nearest available df date on/after candidate
    df_dates_sorted = sorted(set(pd.to_datetime(df_dates).dt.date))
    def next_available(d):
        for dt in df_dates_sorted:
            if pd.to_datetime(dt) >= d:
                return pd.to_datetime(dt)
        return None
    first_earliest = enrollment_date + timedelta(weeks=int(MIN_ANCHOR_GAP_WEEKS))
    first = None
    for c in candidates:
        if c >= first_earliest:
            na = next_available(c)
            if na is not None:
                first = na
                break
    if first is None:
        return (None, None)
    second_earliest = first + timedelta(weeks=int(MIN_ANCHOR_SEPARATION_WEEKS))
    second = None
    for c in candidates:
        if c >= second_earliest:
            na = next_available(c)
            if na is not None:
                second = na
                break
    if second is None:
        return (None, None)
    # Compute week offsets from enrollment (floor to integer weeks)
    offset1 = int((first - enrollment_date).days // 7)
    offset2 = int((second - enrollment_date).days // 7)
    return (offset1, offset2)

def compute_group_slopes_dynamic(df, sheet_name, dual_print_fn=None):
    """Compute slopes using dynamic anchors derived from QUIET_ANCHOR_ISO_WEEKS.

    Falls back to 0.0 slopes if anchors cannot be selected.
    """
    from datetime import datetime, timedelta
    slopes = {}
    # Derive enrollment date from sheet name
    enrollment_date = None
    if "_" in sheet_name:
        year_str, week_str = sheet_name.split("_")
        try:
            year = int(year_str)
            week = int(week_str)
            enrollment_date = datetime.fromisocalendar(year, week, 1)
        except Exception:
            enrollment_date = None
    if enrollment_date is None:
        for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
            slopes[(yob, dose)] = 0.0
        return slopes
    # Choose anchors
    off1, off2 = select_dynamic_anchor_offsets(enrollment_date, df["DateDied"])
    if off1 is None or off2 is None or off2 <= off1:
        if dual_print_fn:
            dual_print_fn(f"[WARN] Dynamic anchors not found for {sheet_name}; using slope 0.0")
        for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
            slopes[(yob, dose)] = 0.0
        return slopes
    T = off2 - off1
    # Log anchors
    if dual_print_fn:
        a1 = (enrollment_date + pd.to_timedelta(off1, unit='W')).date()
        a2 = (enrollment_date + pd.to_timedelta(off2, unit='W')).date()
        dual_print_fn(f"[INFO] {sheet_name} dynamic anchors: t={off1} ({a1}), t={off2} ({a2}), Δt={T} weeks")
    # Compute slopes using smoothed MR within ± window around each offset
    for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        g_sorted = g.sort_values("t")
        w1s, w1e = max(0, off1 - SLOPE_WINDOW_SIZE), off1 + SLOPE_WINDOW_SIZE
        w2s, w2e = max(0, off2 - SLOPE_WINDOW_SIZE), off2 + SLOPE_WINDOW_SIZE
        p1 = g_sorted[(g_sorted["t"] >= w1s) & (g_sorted["t"] <= w1e)]
        p2 = g_sorted[(g_sorted["t"] >= w2s) & (g_sorted["t"] <= w2e)]
        vals1 = p1["MR_smooth"][p1["MR_smooth"] > EPS].values
        vals2 = p2["MR_smooth"][p2["MR_smooth"] > EPS].values
        if len(vals1) == 0 or len(vals2) == 0:
            slopes[(yob, dose)] = 0.0
            continue
        slope = (np.mean(np.log(vals2)) - np.mean(np.log(vals1))) / T
        slopes[(yob, dose)] = float(np.clip(slope, -10.0, 10.0))
    return slopes

def compute_death_slopes_lookup(df, sheet_name, logger=None):
    """[Deprecated] Deaths-based lookup slopes removed. Using 0.0 slopes here."""
    slopes = {}
    for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        slopes[(yob, dose)] = 0.0
    return slopes

def adjust_mr(df, slopes, t0=SLOPE_ANCHOR_T):
    """Multiplicative slope removal on MR with anchoring at week index t0 (enrollment)."""
    def f(row):
        b = slopes.get((row["YearOfBirth"], row["Dose"]), 0.0)
        return row["MR"] * safe_exp(-b * (row["t"] - float(t0)))
    return df.assign(MR_adj=df.apply(f, axis=1))

def safe_sqrt(x, eps=EPS):
    """Safe square root with clipping."""
    return np.sqrt(np.clip(x, eps, None))


def apply_moving_average(df, window=MA_TOTAL_LENGTH, centered=CENTERED):
    """Apply centered moving average smoothing to MR values before quantile regression.
    
    Args:
        window: Total length of moving average (e.g., 8 weeks = 4 weeks on either side)
        centered: If True, use centered MA (4 weeks before + 4 weeks after each point)
    """
    df_smooth = df.copy()
    
    for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        # Sort by time to ensure proper order
        g_sorted = g.sort_values("t")
        
        # Apply centered moving average to MR values
        if centered:
            g_sorted["MR_smooth"] = g_sorted["MR"].rolling(window=window, center=True, min_periods=1).mean()
        else:
            g_sorted["MR_smooth"] = g_sorted["MR"].rolling(window=window, center=False, min_periods=1).mean()
        
        # Update the original dataframe
        df_smooth.loc[g_sorted.index, "MR_smooth"] = g_sorted["MR_smooth"]
    
    return df_smooth

def build_kcor_rows(df, sheet_name, dual_print=None):
    """
    Build per-age KCOR rows for all PAIRS and ASMR pooled rows (YearOfBirth=0).
    Assumptions:
      - Person-time PT = Alive
      - MR = Dead / PT
      - MR_adj slope-removed via QR (for smoothing, not CH calculation)
      - CH = cumsum(-ln(1 - MR_adj)) where MR_adj = MR × exp(-slope × (t - t0))
      - KCOR = (cum_hazard_num / cum_hazard_den), anchored to 1 at week KCOR_NORMALIZATION_WEEK if available
              - 95% CI uses proper uncertainty propagation: Var[KCOR] = KCOR² * [Var[cumD_num]/cumD_num² + Var[cumD_den]/cumD_den² + Var[baseline_num]/baseline_num² + Var[baseline_den]/baseline_den²]
      - ASMR pooling uses fixed baseline weights = sum of PT in the first 4 weeks per age (time-invariant).
    """
    out_rows = []
    # Fast access by (age,dose)
    by_age_dose = {(y,d): g.sort_values("DateDied")
                   for (y,d), g in df.groupby(["YearOfBirth","Dose"], sort=False)}
    
    # Store scale factors for ASMR computation
    scale_factors = {}  # {(age, dose_num, dose_den): scale_factor}

    # -------- per-age KCOR rows --------
    dose_pairs = get_dose_pairs(sheet_name)
    for yob in df["YearOfBirth"].unique():
        for num, den in dose_pairs:
            # Apply debug dose pair filter
            if DEBUG_DOSE_PAIR_ONLY and (num, den) != DEBUG_DOSE_PAIR_ONLY:
                continue
            gv = by_age_dose.get((yob, num))
            gu = by_age_dose.get((yob, den))
            if gv is None or gu is None:
                continue
            # Ensure we have exactly one row per date by taking the first occurrence
            gv_unique = gv[["DateDied","ISOweekDied","MR","MR_adj","CH","CH_actual","cumD_adj","cumD_unadj","hazard","slope","scale_factor","MR_smooth","t"]].drop_duplicates(subset=["DateDied"], keep="first")
            gu_unique = gu[["DateDied","ISOweekDied","MR","MR_adj","CH","CH_actual","cumD_adj","cumD_unadj","hazard","slope","scale_factor","MR_smooth","t"]].drop_duplicates(subset=["DateDied"], keep="first")
            
            merged = pd.merge(
                gv_unique,
                gu_unique,
                on="DateDied", suffixes=("_num","_den"), how="inner"
            ).sort_values("DateDied")
            if merged.empty:
                continue
                
            # Debug: Print detailed info for each date
            # if DEBUG_VERBOSE:
            #     print(f"\n[DEBUG] Age {yob}, Doses {num} vs {den}")
            #     print(f"  Number of merged rows: {len(merged)}")
            #     print(f"  Date range: {merged['DateDied'].min().strftime('%m/%d/%Y')} to {merged['DateDied'].max().strftime('%m/%d/%Y')}")
            #     
            #     # Check for duplicate dates
            #     date_counts = merged['DateDied'].value_counts()
            #     if date_counts.max() > 1:
            #         print(f"  WARNING: Found duplicate dates!")
            #         print(f"    Max rows per date: {date_counts.max()}")
            #         print(f"    Dates with multiple rows: {len(date_counts[date_counts > 1])}")
            #         print(f"    Sample duplicate dates:")
            #         for date, count in date_counts[date_counts > 1].head(3).items():
            #             print(f"      {date.strftime('%m/%d/%Y')}: {count} rows")
            #     
            #     print(f"  Sample data points:")
            #     for i, row in merged.head(5).iterrows():
            #         print(f"    Date: {row['DateDied'].strftime('%m/%d/%Y')}")
            #         print(f"      Dose {num}: MR={row['MR_num']:.6f}, MR_adj={row['MR_adj_num']:.6f}, CH={row['CH_num']:.6f}, cumD_adj={row['cumD_adj_num']:.6f}")
            #         print(f"      Dose {den}: MR={row['MR_den']:.6f}, MR_adj={row['MR_adj_den']:.6f}, CH={row['CH_den']:.6f}, cumD_adj={row['cumD_adj_den']:.6f}")
            #     print()

            # Handle division by zero or very small denominators
            # Note: CH_num and CH_den are cumulative hazards, not mortality rates
            valid_denom = merged["CH_den"] > EPS
            merged["K_raw"] = np.where(valid_denom,
                                      merged["CH_num"] / merged["CH_den"], 
                                      np.nan)
            
            # Get baseline K_raw value at week 4 (usual baseline period)
            t0_idx = KCOR_NORMALIZATION_WEEK if len(merged) > KCOR_NORMALIZATION_WEEK else 0
            baseline_k_raw = merged["K_raw"].iloc[t0_idx]
            if not (np.isfinite(baseline_k_raw) and baseline_k_raw > EPS):
                baseline_k_raw = 1.0
            
            # Check if KCOR at FINAL_KCOR_DATE needs scaling adjustment
            scale_factor = 1.0  # Default: no scaling
            
            # Parse the final KCOR date
            try:
                final_date = pd.to_datetime(FINAL_KCOR_DATE, format="%m/%d/%y")
            except:
                try:
                    final_date = pd.to_datetime(FINAL_KCOR_DATE, format="%m/%d/%Y")
                except:
                    final_date = None
            
            if final_date is not None:
                # Check if we have data for the final date
                final_date_data = merged[merged["DateDied"] == final_date]
                if not final_date_data.empty:
                    # Compute what KCOR would be at final date with current baseline
                    final_k_raw = final_date_data["K_raw"].iloc[0]
                    if np.isfinite(final_k_raw):
                        final_kcor = final_k_raw / baseline_k_raw
                        
                        # If final KCOR is below minimum, adjust scaling factor
                        if final_kcor < FINAL_KCOR_MIN:
                            scale_factor = 1.0 / final_kcor
            
            # Save scale factor for ASMR computation
            scale_factors[(yob, num, den)] = scale_factor
            
            # Compute final KCOR values using adjusted scale factor
            merged["KCOR"] = np.where(np.isfinite(merged["K_raw"]), 
                                     (merged["K_raw"] / baseline_k_raw) * scale_factor, 
                                     np.nan)
            
            # Debug: Check for suspiciously large KCOR values
            # if merged["KCOR"].max() > 10:
            #     print(f"\n[DEBUG] Large KCOR detected in {sheet_name}, Age {yob}, Doses {num} vs {den}")
            #     print(f"  CH_num range: {merged['CH_num'].min():.6f} to {merged['CH_num'].max():.6f}")
            #     print(f"  CH_den range: {merged['CH_den'].min():.6f} to {merged['CH_den'].max():.6f}")
            #     print(f"  K_raw range: {merged['K_raw'].min():.6f} to {merged['K_raw'].max():.6f}")
            #     print(f"  Anchor value: {anchor:.6f}")
            #     print(f"  KCOR range: {merged['KCOR'].min():.6f} to {merged['KCOR'].max():.6f}")
            #     print(f"  Sample data points:")
            #     for i, row in merged.head(3).iterrows():
            #         print(f"    Date: {row['DateDied']}, CH_num: {row['CH_num']:.6f}, CH_den: {row['CH_den']:.6f}, K_raw: {row['K_raw']:.6f}, KCOR: {row['KCOR']:.6f}")
            #     print()

            # Correct KCOR 95% CI calculation based on baseline uncertainty
            # Get baseline death counts at week 4
            t0_idx = KCOR_NORMALIZATION_WEEK if len(merged) > KCOR_NORMALIZATION_WEEK else 0
            baseline_num = merged["cumD_adj_num"].iloc[t0_idx]
            baseline_den = merged["cumD_adj_den"].iloc[t0_idx]
            
            # Note: The baseline uncertainty calculation doesn't need scaling adjustment
            # because the scaling factor affects the normalization but not the underlying
            # death count uncertainties, which are what drive the CI calculation
            
            # Calculate variance components for each time point
            # Var[KCOR] = KCOR² * [Var[cumD_num]/cumD_num² + Var[cumD_den]/cumD_den² + Var[baseline_num]/baseline_num² + Var[baseline_den]/baseline_den²]
            # Using binomial uncertainty: Var[D] ≈ D for death counts
            
            merged["SE_logKCOR"] = safe_sqrt(
                (merged["cumD_adj_num"] + EPS) / (merged["cumD_adj_num"] + EPS)**2 +  # Var[cumD_num]/cumD_num²
                (merged["cumD_adj_den"] + EPS) / (merged["cumD_adj_den"] + EPS)**2 +  # Var[cumD_den]/cumD_den²
                (baseline_num + EPS) / (baseline_num + EPS)**2 +                      # Var[baseline_num]/baseline_num²
                (baseline_den + EPS) / (baseline_den + EPS)**2                        # Var[baseline_den]/baseline_den²
            )
            
            # Calculate 95% CI bounds on log scale, then exponentiate
            # CI = exp(log(KCOR) ± 1.96 * SE_logKCOR)
            merged["CI_lower"] = merged["KCOR"] * safe_exp(-1.96 * merged["SE_logKCOR"])
            merged["CI_upper"] = merged["KCOR"] * safe_exp(1.96 * merged["SE_logKCOR"])
            
            # Clip CI bounds to reasonable values
            merged["CI_lower"] = np.clip(merged["CI_lower"], 0, merged["KCOR"] * 10)
            merged["CI_upper"] = np.clip(merged["CI_upper"], merged["KCOR"] * 0.1, merged["KCOR"] * 10)

            out = merged[["DateDied","ISOweekDied_num","KCOR","CI_lower","CI_upper",
                          "MR_num","MR_adj_num","CH_num","CH_actual_num","hazard_num","slope_num","scale_factor_num","MR_smooth_num","t_num",
                          "MR_den","MR_adj_den","CH_den","CH_actual_den","hazard_den","slope_den","scale_factor_den","MR_smooth_den","t_den"]].copy()

            # Scale MR fields for display (annualized per 100,000)
            for col in ["MR_num","MR_adj_num","MR_den","MR_adj_den","MR_smooth_num","MR_smooth_den"]:
                if col in out.columns:
                    out[col] = out[col] * MR_DISPLAY_SCALE
            out["EnrollmentDate"] = sheet_name
            out["YearOfBirth"] = yob
            out["Dose_num"] = num
            out["Dose_den"] = den
            out.rename(columns={"ISOweekDied_num":"ISOweekDied",
                                "DateDied":"Date"}, inplace=True)
            # CI_95 column removed as it's redundant with CI_lower and CI_upper columns
            # Convert Date to standard pandas date format (same as debug sheet)
            out["Date"] = pd.to_datetime(out["Date"]).apply(lambda x: x.date())
            out_rows.append(out)

    # -------- ASMR pooled rows (YearOfBirth = 0) --------
    # Expected-deaths weights using pooled quiet baseline window
    # w_a ∝ h_a × PT_a(W) where h_a = smoothed mean MR in quiet window W
    weights = {}
    df_sorted = df.sort_values("DateDied")
    
    # Define quiet baseline window W using the agreed range: first 4 distinct weeks from the sheet start
    quiet_window_dates = df_sorted.drop_duplicates(subset=["DateDied"]).head(4)["DateDied"].tolist()
    quiet_data = df_sorted[df_sorted["DateDied"].isin(quiet_window_dates)]
    
    # Calculate expected-deaths weights for each age group
    for yob, g_age in quiet_data.groupby("YearOfBirth", sort=False):
        # Get all doses for this age group in quiet window
        age_quiet_data = g_age
        
        # Calculate smoothed mean mortality rate h_a across all doses in quiet window
        # Use 8-week MA smoothing (already applied to MR_smooth column)
        smoothed_mrs = age_quiet_data["MR_smooth"].values
        valid_mrs = smoothed_mrs[smoothed_mrs > EPS]  # Remove zeros for mean calculation
        
        if len(valid_mrs) > 0:
            h_a = np.mean(valid_mrs)  # Smoothed mean MR in quiet window
        else:
            # Handle zero deaths with shrinkage - use minimum observed MR across all ages
            all_mrs = quiet_data["MR_smooth"][quiet_data["MR_smooth"] > EPS].values
            h_a = np.min(all_mrs) if len(all_mrs) > 0 else EPS
        
        # Calculate person-time PT_a(W) for this age group in quiet window
        PT_a = float(age_quiet_data["PT"].sum())
        
        # Expected-deaths weight: w_a ∝ h_a × PT_a(W)
        weights[yob] = h_a * PT_a
    
    # Normalize weights: w_a = (h_a PT_a(W)) / (Σ_b h_b PT_b(W))
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {yob: w / total_weight for yob, w in weights.items()}
    else:
        # Fallback to equal weights if all weights are zero
        weights = {yob: 1.0 / len(weights) for yob in weights.keys()}
    
    # Debug: Show expected-deaths weights
    if DEBUG_VERBOSE:
        print(f"\n[DEBUG] Expected-deaths weights for ASMR pooling:")
        for yob in sorted(weights.keys()):
            print(f"  Age {yob}: weight = {weights[yob]:.6f}")
        print(f"  Total weight: {sum(weights.values()):.6f}")
        print()

    pooled_rows = []
    all_dates = sorted(df_sorted["DateDied"].unique())

    dose_pairs = get_dose_pairs(sheet_name)
    for num, den in dose_pairs:
        # Per-age anchors at t0 for this (num,den)
        anchors = {}
        for yob, g_age in df_sorted.groupby("YearOfBirth", sort=False):
            gvn = g_age[g_age["Dose"] == num].sort_values("DateDied")
            gdn = g_age[g_age["Dose"] == den].sort_values("DateDied")
            if gvn.empty or gdn.empty:
                continue
            t0_idx = KCOR_NORMALIZATION_WEEK if len(gvn) > KCOR_NORMALIZATION_WEEK and len(gdn) > KCOR_NORMALIZATION_WEEK else 0
            c1 = gvn["CH"].iloc[t0_idx]
            c0 = gdn["CH"].iloc[t0_idx]
            if np.isfinite(c1) and np.isfinite(c0) and c1 > EPS and c0 > EPS:
                anchors[yob] = c1 / c0

        for dt in all_dates:
            logs, wts, var_terms = [], [], []
            for yob, g_age in df_sorted.groupby("YearOfBirth", sort=False):
                if yob not in anchors:
                    continue
                gv = g_age[(g_age["Dose"]==num) & (g_age["DateDied"]==dt)]
                gu = g_age[(g_age["Dose"]==den) & (g_age["DateDied"]==dt)]
                if gv.empty or gu.empty:
                    continue
                # Take first occurrence if multiple rows per date
                gv = gv.iloc[[0]]
                gu = gu.iloc[[0]]
                denom_ch = gu["CH"].values[0]
                if denom_ch > EPS:
                    k = (gv["CH"].values[0]) / denom_ch
                else:
                    continue  # Skip this comparison if denominator is too small
                k0 = anchors[yob]
                if not (np.isfinite(k) and np.isfinite(k0) and k0 > EPS and k > EPS):
                    continue
                kstar = k / k0
                
                # ASMR pooled values should not be scaled; use raw kstar
                
                logs.append(safe_log(kstar))
                wts.append(weights.get(yob, 0.0))
                Dv = float(gv["cumD_adj"].values[0])
                Du = float(gu["cumD_adj"].values[0])
                if Dv > EPS and Du > EPS:
                    var_terms.append((weights.get(yob,0.0)**2) * (1.0/Dv + 1.0/Du))

            # Pool only over valid, finite entries; ignore NaNs/inf entirely
            if logs and sum(wts) > 0:
                logs_arr = np.array(logs, dtype=float)
                wts_arr = np.array(wts, dtype=float)
                valid_mask = np.isfinite(logs_arr) & (wts_arr > 0)
                if np.any(valid_mask):
                    logs_arr = logs_arr[valid_mask]
                    wts_arr = wts_arr[valid_mask]
                    # Re-normalize weights among valid ages to avoid implicit down-weighting
                    w_sum = wts_arr.sum()
                    if w_sum <= 0:
                        continue
                    logK = np.average(logs_arr, weights=wts_arr)
                    Kpool = float(safe_exp(logK))
                else:
                    continue
                
                # Correct pooled KCOR 95% CI calculation based on baseline uncertainty
                # For pooled results, we need to account for baseline uncertainty across all age groups
                baseline_uncertainty = 0.0
                for yob, g_age in df_sorted.groupby("YearOfBirth", sort=False):
                    if yob not in anchors:
                        continue
                    gvn = g_age[g_age["Dose"] == num].sort_values("DateDied")
                    gdn = g_age[g_age["Dose"] == den].sort_values("DateDied")
                    if len(gvn) > KCOR_NORMALIZATION_WEEK and len(gdn) > KCOR_NORMALIZATION_WEEK:
                        baseline_num = gvn["cumD_adj"].iloc[KCOR_NORMALIZATION_WEEK]
                        baseline_den = gdn["cumD_adj"].iloc[KCOR_NORMALIZATION_WEEK]
                        if np.isfinite(baseline_num) and np.isfinite(baseline_den) and baseline_num > EPS and baseline_den > EPS:
                            # Add baseline uncertainty for this age group (weighted)
                            weight = weights.get(yob, 0.0)
                            baseline_uncertainty += (weight**2) * (1.0/baseline_num + 1.0/baseline_den)
                
                # Calculate total uncertainty including baseline
                total_uncertainty = sum(var_terms) + baseline_uncertainty
                SE_total = safe_sqrt(total_uncertainty) / sum(wts)
                
                # Clip SE to prevent overflow (using reasonable bound)
                SE_total = min(SE_total, 10.0)
                
                # Calculate 95% CI bounds on log scale, then exponentiate
                CI_lower = Kpool * safe_exp(-1.96 * SE_total)
                CI_upper = Kpool * safe_exp(1.96 * SE_total)
                
                # Clip CI bounds to reasonable values
                CI_lower = max(0, min(CI_lower, Kpool * 10))
                CI_upper = max(Kpool * 0.1, min(CI_upper, Kpool * 10))
                

                
                # Debug: Check for suspiciously large pooled KCOR values
                if Kpool > 10:
                    print(f"\n[DEBUG] Large pooled KCOR detected: {Kpool:.6f}")
                    print(f"  Dose combination: {num} vs {den}, Date: {dt}")
                    print(f"  Number of age groups: {len(logs)}")
                    print(f"  Log K values: {[f'{x:.6f}' for x in logs[:5]]}...")
                    print(f"  Weights: {[f'{w:.6f}' for w in wts[:5]]}...")
                    print(f"  Final logK: {logK:.6f}")
                    print()
                pooled_rows.append({
                    "EnrollmentDate": sheet_name,
                    "ISOweekDied": df_sorted.loc[df_sorted["DateDied"]==dt, "ISOweekDied"].iloc[0],
                    "Date": pd.to_datetime(dt).date(),  # Convert to standard pandas date format (same as debug sheet)
                    "YearOfBirth": 0,      # ASMR pooled row
                    "Dose_num": num,
                    "Dose_den": den,
                    "KCOR": Kpool,
                    "CI_lower": CI_lower,
                    "CI_upper": CI_upper,
                    "MR_num": np.nan,
                    "MR_adj_num": np.nan,
                    "CH_num": np.nan,
                    "CH_actual_num": np.nan,
                    "hazard_num": np.nan,
                    "slope_num": np.nan,
                    "scale_factor_num": np.nan,
                    "MR_smooth_num": np.nan,
                    "t_num": np.nan,
                    "MR_den": np.nan,
                    "MR_adj_den": np.nan,
                    "CH_den": np.nan,
                    "CH_actual_den": np.nan,
                    "hazard_den": np.nan,
                    "slope_den": np.nan,
                    "scale_factor_den": np.nan,
                    "MR_smooth_den": np.nan,
                    "t_den": np.nan
                })

    if out_rows or pooled_rows:
        return pd.concat(out_rows + [pd.DataFrame(pooled_rows)], ignore_index=True)
    return pd.DataFrame(columns=[
        "EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den",
        "KCOR","CI_lower","CI_upper","MR_num","MR_adj_num","CH_num","CH_actual_num","hazard_num","slope_num","scale_factor_num","MR_smooth_num","t_num",
        "MR_den","MR_adj_den","CH_den","CH_actual_den","hazard_den","slope_den","scale_factor_den","MR_smooth_den","t_den"
    ])

def build_kcor_o_rows(df, sheet_name):
    """Compute KCOR_o based on death-based slope adjustment and cumulative adjusted deaths.
    
    Steps:
      1) Compute death-based slopes via lookup windows.
      2) Adjust weekly deaths using multiplicative slope removal anchored at week KCOR_NORMALIZATION_WEEK.
      3) Compute cumulative adjusted deaths per (YearOfBirth, Dose).
      4) For each dose pair and age, compute the ratio of cumulative deaths, then normalize by its
         value at week KCOR_NORMALIZATION_WEEK to make it 1 at that anchor week.
    """
    # Compute death-based slopes
    death_slopes = compute_death_slopes_lookup(df, sheet_name)
    
    # Time index already exists; ensure sorting
    df = df.sort_values(["YearOfBirth","Dose","DateDied"]).copy()
    
    # Apply slope correction to weekly deaths using death-based slopes
    def apply_death_slope(row):
        b = death_slopes.get((row["YearOfBirth"], row["Dose"]), 0.0)
        # Anchor slope normalization at enrollment week (t0 = 0)
        scale = safe_exp(-np.clip(b, -10.0, 10.0) * row["t"])
        return row["Dead"] * scale
    df["Dead_adj_o"] = df.apply(apply_death_slope, axis=1)
    
    # Cumulative adjusted deaths per group
    df["cumD_o"] = df.groupby(["YearOfBirth","Dose"])['Dead_adj_o'].cumsum()
    
    out_rows = []
    by_age_dose = {(y,d): g.sort_values("DateDied") for (y,d), g in df.groupby(["YearOfBirth","Dose"], sort=False)}
    dose_pairs = get_dose_pairs(sheet_name)
    for yob in df["YearOfBirth"].unique():
        for num, den in dose_pairs:
            gv = by_age_dose.get((yob, num))
            gu = by_age_dose.get((yob, den))
            if gv is None or gu is None:
                continue
            gv_unique = gv[["DateDied","ISOweekDied","cumD_o"]].drop_duplicates(subset=["DateDied"], keep="first")
            gu_unique = gu[["DateDied","ISOweekDied","cumD_o"]].drop_duplicates(subset=["DateDied"], keep="first")
            merged = pd.merge(gv_unique, gu_unique, on="DateDied", suffixes=("_num","_den"), how="inner").sort_values("DateDied")
            if merged.empty:
                continue
            # Raw ratio of cumulative adjusted deaths
            valid = merged["cumD_o_den"] > EPS
            merged["K_raw_o"] = np.where(valid, merged["cumD_o_num"] / merged["cumD_o_den"], np.nan)
            # Normalize at anchor week index (week 4 or first)
            t0_idx = KCOR_NORMALIZATION_WEEK if len(merged) > KCOR_NORMALIZATION_WEEK else 0
            k0 = merged["K_raw_o"].iloc[t0_idx]
            if not (np.isfinite(k0) and k0 > EPS):
                k0 = 1.0
            merged["KCOR_o"] = np.where(np.isfinite(merged["K_raw_o"]), merged["K_raw_o"] / k0, np.nan)
            out = merged[["DateDied","ISOweekDied_num","KCOR_o"]].copy()
            out["EnrollmentDate"] = sheet_name
            out["YearOfBirth"] = yob
            out["Dose_num"] = num
            out["Dose_den"] = den
            out.rename(columns={"ISOweekDied_num":"ISOweekDied","DateDied":"Date"}, inplace=True)
            out["Date"] = pd.to_datetime(out["Date"]).apply(lambda x: x.date())
            out_rows.append(out)
    if out_rows:
        return pd.concat(out_rows, ignore_index=True)
    return pd.DataFrame(columns=["EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den","KCOR_o"])

def build_kcor_o_deaths_details(df, sheet_name):
    """For each dose pair and age, output deaths/week, adjusted deaths/week, cumulative adjusted deaths,
    and normalized cumulative ratio (normalized to 1 at week KCOR_NORMALIZATION_WEEK).

    Columns:
      EnrollmentDate, Date, ISOweekDied, YearOfBirth, Dose_num, Dose_den,
      Dead_num, Dead_adj_num, cumD_num, Dead_den, Dead_adj_den, cumD_den,
      K_raw_o, KCOR_o
    """
    # Compute death-based slopes and adjusted deaths
    death_slopes = compute_death_slopes_lookup(df, sheet_name)
    df = df.sort_values(["YearOfBirth","Dose","DateDied"]).copy()

    def apply_death_slope(row):
        b = death_slopes.get((row["YearOfBirth"], row["Dose"]), 0.0)
        # Anchor slope normalization at enrollment week (t0 = 0)
        scale = safe_exp(-np.clip(b, -10.0, 10.0) * row["t"])
        return row["Dead"] * scale
    df["Dead_adj_o"] = df.apply(apply_death_slope, axis=1)
    df["cumD_o"] = df.groupby(["YearOfBirth","Dose"])['Dead_adj_o'].cumsum()

    # Compute enrollment-week Alive per (YearOfBirth, Dose)
    # alive0_map[(YearOfBirth, Dose)] = Alive at t == 0
    try:
        first_rows = (
            df.sort_values(["YearOfBirth","Dose","t"])\
              .groupby(["YearOfBirth","Dose"], sort=False)
              .first()
              .reset_index()
        )
        alive0_map = {
            (int(r["YearOfBirth"]), int(r["Dose"])): float(r.get("Alive", np.nan))
            for _, r in first_rows.iterrows()
        }
    except Exception:
        alive0_map = {}

    out_rows = []
    by_age_dose = {(y,d): g.sort_values("DateDied") for (y,d), g in df.groupby(["YearOfBirth","Dose"], sort=False)}
    dose_pairs = get_dose_pairs(sheet_name)
    for yob in df["YearOfBirth"].unique():
        for num, den in dose_pairs:
            gv = by_age_dose.get((yob, num))
            gu = by_age_dose.get((yob, den))
            if gv is None or gu is None:
                continue
            gv_unique = gv[["DateDied","ISOweekDied","Dead","Dead_adj_o","cumD_o"]].drop_duplicates(subset=["DateDied"], keep="first")
            gu_unique = gu[["DateDied","ISOweekDied","Dead","Dead_adj_o","cumD_o"]].drop_duplicates(subset=["DateDied"], keep="first")
            merged = pd.merge(
                gv_unique, gu_unique, on="DateDied", suffixes=("_num","_den"), how="inner"
            ).sort_values("DateDied")
            if merged.empty:
                continue
            valid = merged["cumD_o_den"] > EPS
            merged["K_raw_o"] = np.where(valid, merged["cumD_o_num"] / merged["cumD_o_den"], np.nan)
            t0_idx = KCOR_NORMALIZATION_WEEK if len(merged) > KCOR_NORMALIZATION_WEEK else 0
            k0 = merged["K_raw_o"].iloc[t0_idx]
            if not (np.isfinite(k0) and k0 > EPS):
                k0 = 1.0
            merged["KCOR_o"] = np.where(np.isfinite(merged["K_raw_o"]), merged["K_raw_o"] / k0, np.nan)

            # Add CMRR = (cumD_num/cumD_den) * (Alive_den0/Alive_num0)
            alive_num0 = float(alive0_map.get((yob, num), np.nan))
            alive_den0 = float(alive0_map.get((yob, den), np.nan))
            const_ratio = np.nan
            if np.isfinite(alive_num0) and np.isfinite(alive_den0) and alive_num0 > EPS:
                const_ratio = alive_den0 / alive_num0
            merged["CMRR"] = np.where(
                np.isfinite(merged["K_raw_o"]) & np.isfinite(const_ratio),
                merged["K_raw_o"] * const_ratio,
                np.nan,
            )

            out = merged[[
                "DateDied","ISOweekDied_num",
                "Dead_num","Dead_adj_o_num","cumD_o_num",
                "Dead_den","Dead_adj_o_den","cumD_o_den",
                "K_raw_o","KCOR_o","CMRR"
            ]].copy()
            out["EnrollmentDate"] = sheet_name
            out["YearOfBirth"] = yob
            out["Dose_num"] = num
            out["Dose_den"] = den
            out.rename(columns={"ISOweekDied_num":"ISOweekDied","DateDied":"Date",
                                "cumD_o_num":"cumD_num","cumD_o_den":"cumD_den",
                                "Dead_adj_o_num":"Dead_adj_num","Dead_adj_o_den":"Dead_adj_den"}, inplace=True)
            out["Date"] = pd.to_datetime(out["Date"]).apply(lambda x: x.date())
            out_rows.append(out)
    if out_rows:
        cols = [
            "EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den",
            "Dead_num","Dead_adj_num","cumD_num","Dead_den","Dead_adj_den","cumD_den",
            "K_raw_o","KCOR_o","CMRR"
        ]
        return pd.concat(out_rows, ignore_index=True)[cols]
    return pd.DataFrame(columns=[
        "EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den",
        "Dead_num","Dead_adj_num","cumD_num","Dead_den","Dead_adj_den","cumD_den",
        "K_raw_o","KCOR_o","CMRR"
    ])

def process_workbook(src_path: str, out_path: str, log_filename: str = "KCOR_summary.log"):
    
    # Set up dual output (console + file)
    output_dir = os.path.dirname(out_path)
    dual_print, log_file_handle = setup_dual_output(output_dir, log_filename)
    
    # Print professional header
    from datetime import datetime
    log_file_path = os.path.join(output_dir, log_filename)
    # Use forward slashes for display consistency
    log_file_display = log_file_path.replace('\\', '/')
    dual_print("="*80)
    dual_print(f"KCOR {VERSION} - Kirsch Cumulative Outcomes Ratio Analysis")
    dual_print("="*80)
    dual_print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mode_str = os.environ.get('KCOR_MODE', 'Primary Analysis')
    dual_print(f"Mode: {mode_str}")
    if _is_sa_mode():
        out_dir_hdr = os.path.dirname(out_path)
        sa_hdr_path = os.path.join(out_dir_hdr, "KCOR_SA.xlsx").replace('\\', '/')
        dual_print(f"Input File: {src_path}")
        dual_print(f"SA Output File: {sa_hdr_path}")
        dual_print(f"Log File: {log_file_display}")
        dual_print("(Standard multi-sheet workbook suppressed in SA mode)")
    else:
        dual_print(f"Input File: {src_path}")
        dual_print(f"Output File: {out_path}")
        dual_print(f"Log File: {log_file_display}")

    # Configuration parameter dump (always show effective values)
    dual_print("-"*80)
    dual_print("Configuration Parameters (effective):")
    dual_print(f"  FINAL_KCOR_MIN        = {FINAL_KCOR_MIN}")
    dual_print(f"  FINAL_KCOR_DATE       = {FINAL_KCOR_DATE}")
    dual_print(f"  KCOR_NORMALIZATION_WEEK = {KCOR_NORMALIZATION_WEEK}")
    dual_print(f"  SLOPE_ANCHOR_T        = {SLOPE_ANCHOR_T}")
    dual_print(f"  DYNAMIC_HVE_SKIP_WEEKS = {DYNAMIC_HVE_SKIP_WEEKS}")
    dual_print(f"  AGE_RANGE             = {AGE_RANGE}")
    dual_print(f"  SLOPE_WINDOW_SIZE     = {SLOPE_WINDOW_SIZE}")
    dual_print(f"  MA_TOTAL_LENGTH       = {MA_TOTAL_LENGTH}")
    dual_print(f"  CENTERED              = {CENTERED}")
    dual_print(f"  YEAR_RANGE            = {YEAR_RANGE}")
    dual_print(f"  ENROLLMENT_DATES      = {ENROLLMENT_DATES}")
    # MR is computed in annualized per 100k units; no additional display scaling applied
    dual_print(f"  DEBUG_VERBOSE         = {DEBUG_VERBOSE}")
    dual_print(f"  OVERRIDE_DOSE_PAIRS   = {OVERRIDE_DOSE_PAIRS}")
    dual_print(f"  OVERRIDE_YOBS         = {OVERRIDE_YOBS}")
    dual_print(f"  QUIET_ANCHOR_ISO_WEEKS= {QUIET_ANCHOR_ISO_WEEKS}")
    dual_print(f"  MIN_ANCHOR_GAP_WEEKS  = {MIN_ANCHOR_GAP_WEEKS}")
    dual_print(f"  MIN_ANCHOR_SEPARATION_WEEKS = {MIN_ANCHOR_SEPARATION_WEEKS}")
    dual_print(f"  KCOR_REPORTING_DATE   = {KCOR_REPORTING_DATE}")
    dual_print("="*80)
    dual_print("")
    
    xls = pd.ExcelFile(src_path)
    all_out = []
    pair_deaths_all = []
    
    # Apply processing filters
    sheets_to_process = ENROLLMENT_DATES if ENROLLMENT_DATES else xls.sheet_names
    if YEAR_RANGE:
        start_year, end_year = YEAR_RANGE
        dual_print(f"[DEBUG] Limiting to age range: {start_year}-{end_year}")
    if ENROLLMENT_DATES:
        dual_print(f"[DEBUG] Limiting to enrollment dates: {ENROLLMENT_DATES}")
    
    # Initialize debug data collection (will be populated inside sheet loop)
    debug_data = []
    
    for sh in sheets_to_process:
        dual_print(f"[Info] Processing sheet: {sh}")
        df = pd.read_excel(src_path, sheet_name=sh)
        # prep
        df["DateDied"] = pd.to_datetime(df["DateDied"])
        
        # Filter out unreasonably large birth years (keep -1 for "not available")
        df = df[df["YearOfBirth"] <= 2020]
        
        # Filter to start from enrollment date (sheet name format: YYYY_WW)
        if "_" in sh:
            year_str, week_str = sh.split("_")
            enrollment_year = int(year_str)
            enrollment_week = int(week_str)
            # Convert ISO week to date more accurately
            from datetime import datetime, timedelta
            # ISO week 24 of 2021 should be around June 14, 2021
            # Let's use a more precise calculation
            jan1 = datetime(enrollment_year, 1, 1)
            # Find the first Monday of the year (ISO week 1)
            days_to_monday = (7 - jan1.weekday()) % 7
            if days_to_monday == 0 and jan1.weekday() != 0:
                days_to_monday = 7
            first_monday = jan1 + timedelta(days=days_to_monday)
            # Calculate the start of the specified ISO week
            enrollment_date = first_monday + timedelta(weeks=enrollment_week-1)
            df = df[df["DateDied"] >= enrollment_date]
            dual_print(f"[DEBUG] Filtered to start from enrollment date {enrollment_date.strftime('%m/%d/%Y')}: {len(df)} rows")
            # SA pre-check: ensure max(start)+max(length) does not exceed available data window
            if _is_sa_mode():
                try:
                    sa_starts_chk = _parse_triplet_range(os.environ.get('SA_SLOPE_START', ''))
                    sa_lengths_chk = _parse_triplet_range(os.environ.get('SA_SLOPE_LENGTH', ''))
                    if sa_starts_chk and sa_lengths_chk:
                        max_start = max(sa_starts_chk)
                        max_len = max(sa_lengths_chk)
                        last_date = pd.to_datetime(df["DateDied"]).max()
                        allowed_weeks = int((last_date - enrollment_date).days // 7)
                        if (max_start + max_len) > allowed_weeks:
                            dual_print(f"\n❌ SA configuration invalid for sheet {sh}: max(start)+max(length)={max_start}+{max_len}={max_start+max_len} > allowed {allowed_weeks} weeks (to {last_date.date()}).")
                            dual_print("   Please reduce SA_SLOPE_START/SA_SLOPE_LENGTH maxima.")
                            raise SystemExit(2)
                except Exception:
                    pass
        
        # Apply debug age filter
        if YEAR_RANGE:
            start_year, end_year = YEAR_RANGE
            df = df[(df["YearOfBirth"] >= start_year) & (df["YearOfBirth"] <= end_year)]
            dual_print(f"[DEBUG] Filtered to {len(df)} rows for ages {start_year}-{end_year}")
        
        # Apply sheet-specific dose filtering
        dose_pairs = get_dose_pairs(sh)
        max_dose = max(max(pair) for pair in dose_pairs)
        valid_doses = list(range(max_dose + 1))  # Include all doses from 0 to max_dose
        df = df[df["Dose"].isin(valid_doses)]
        dual_print(f"[DEBUG] Filtered to doses {valid_doses} (max dose {max_dose}): {len(df)} rows")
        
        # Aggregate across sexes for each dose/date/age combination
        # Optionally bucket YearOfBirth into AGE_RANGE-year bins (e.g., 10 -> 1920, 1930, ...)
        try:
            if int(AGE_RANGE) and int(AGE_RANGE) > 1:
                df["YearOfBirth"] = (df["YearOfBirth"].astype(int) // int(AGE_RANGE)) * int(AGE_RANGE)
        except Exception:
            pass
        df = df.groupby(["YearOfBirth", "Dose", "DateDied"]).agg({
            "ISOweekDied": "first",
            "Alive": "sum",
            "Dead": "sum"
        }).reset_index()
        dual_print(f"[DEBUG] Aggregated across sexes: {len(df)} rows")
        
        df = df.sort_values(["YearOfBirth","Dose","DateDied"]).reset_index(drop=True)
        # person-time proxy and MR (internal probability per week)
        df["PT"]   = df["Alive"].astype(float).clip(lower=0.0)
        df["Dead"] = df["Dead"].astype(float).clip(lower=0.0)
        df["MR"]   = np.where(df["PT"] > 0, df["Dead"]/(df["PT"] + EPS), np.nan)
        df["t"]    = df.groupby(["YearOfBirth","Dose"]).cumcount().astype(float)

        # Apply centered moving average smoothing before quantile regression
        df = apply_moving_average(df, window=MA_TOTAL_LENGTH, centered=CENTERED)
        
        # Debug: Show effect of moving average
        # if DEBUG_VERBOSE:
        #     print(f"\n[DEBUG] Centered moving average smoothing (total length={MA_TOTAL_LENGTH} weeks, centered={CENTERED}):")
        #     for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        #         if len(g) > 0:
        #             mr_orig = g["MR"].values
        #             mr_smooth = g["MR_smooth"].values
        #             print(f"  Age {yob}, Dose {dose}:")
        #             print(f"    Original MR range: {mr_orig.min():.6f} to {mr_orig.max():.6f}")
        #             print(f"    Smoothed MR range: {mr_smooth.min():.6f} to {mr_smooth.max():.6f}")
        
        # SA: iterate slope ranges if provided; else compute once
        sa_mode = _is_sa_mode()
        sa_starts = _parse_triplet_range(os.environ.get('SA_SLOPE_START', '')) if sa_mode else None
        sa_lengths = _parse_triplet_range(os.environ.get('SA_SLOPE_LENGTH', '')) if sa_mode else None
        if sa_mode and sa_starts and sa_lengths:
            produced_outputs = []
            for start_val in sa_starts:
                skip_remaining_lengths = False
                for length_val in sa_lengths:
                    try:
                        # Build slopes using explicit offsets relative to enrollment
                        # Emulate lookup behavior by using two offsets (start, start+length)
                        # Create a transient slope function using the same geometry as dynamic method
                        def slopes_from_offsets(df_local, start_off, len_off):
                            Tloc = int(len_off)
                            off1loc = int(start_off)
                            off2loc = off1loc + Tloc
                            slopes_loc = {}
                            for (yob, dose), g in df_local.groupby(["YearOfBirth","Dose"], sort=False):
                                g_sorted = g.sort_values("t")
                                w1s, w1e = max(0, off1loc - SLOPE_WINDOW_SIZE), off1loc + SLOPE_WINDOW_SIZE
                                w2s, w2e = max(0, off2loc - SLOPE_WINDOW_SIZE), off2loc + SLOPE_WINDOW_SIZE
                                p1 = g_sorted[(g_sorted["t"] >= w1s) & (g_sorted["t"] <= w1e)]
                                p2 = g_sorted[(g_sorted["t"] >= w2s) & (g_sorted["t"] <= w2e)]
                                v1 = p1["MR_smooth"][p1["MR_smooth"] > EPS].values
                                v2 = p2["MR_smooth"][p2["MR_smooth"] > EPS].values
                                if len(v1) == 0 or len(v2) == 0 or Tloc <= 0:
                                    slopes_loc[(yob,dose)] = 0.0
                                    continue
                                s = (np.mean(np.log(v2)) - np.mean(np.log(v1))) / Tloc
                                slopes_loc[(yob,dose)] = float(np.clip(s, -10.0, 10.0))
                            return slopes_loc
                        slopes = slopes_from_offsets(df, int(start_val), int(length_val))
                    except ValueError as e:
                        # Safety check for invalid slope combo; skip and mark to stop trying larger combos
                        if DEBUG_VERBOSE:
                            print(f"[SA] Skipping invalid slope combo start={start_val}, length={length_val}: {e}")
                        skip_remaining_lengths = True
                        continue
                    # Build a copy pipeline to avoid cross-contamination
                    df2 = df.copy()
                    df2 = adjust_mr(df2, slopes, t0=KCOR_NORMALIZATION_WEEK)
                    # Add slope and scale_factor columns (required by downstream build_kcor_rows)
                    df2["slope"] = df2.apply(lambda row: slopes.get((row["YearOfBirth"], row["Dose"]), 0.0), axis=1)
                    df2["slope"] = np.clip(df2["slope"], -10.0, 10.0)
                    df2["scale_factor"] = df2.apply(lambda row: safe_exp(-df2["slope"].iloc[row.name] * (row["t"] - float(KCOR_NORMALIZATION_WEEK))), axis=1)
                    df2["hazard"] = -np.log(1 - df2["MR_adj"].clip(upper=0.999))
                    # Apply DYNAMIC_HVE_SKIP_WEEKS: start accumulation at this week index
                    df2["hazard_eff"] = np.where(df2["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df2["hazard"], 0.0)
                    df2["MR_eff"] = np.where(df2["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df2["MR"], 0.0)
                    df2["CH"] = df2.groupby(["YearOfBirth","Dose"]) ['hazard_eff'].cumsum()
                    df2["CH_actual"] = df2.groupby(["YearOfBirth","Dose"]) ['MR_eff'].cumsum()
                    df2["cumPT"] = df2.groupby(["YearOfBirth","Dose"]) ['PT'].cumsum()
                    df2["cumD_adj"] = df2["CH"] * df2["cumPT"]
                    df2["cumD_unadj"] = df2.groupby(["YearOfBirth","Dose"]) ['Dead'].cumsum()
                    # Sweep FINAL_KCOR_MIN if provided as range
                    sa_final_mins = _parse_triplet_range(os.environ.get('SA_FINAL_KCOR_MIN', ''))
                    if not sa_final_mins:
                        try:
                            sa_final_mins = [float(FINAL_KCOR_MIN)]
                        except Exception:
                            sa_final_mins = [1.0]
                    prev_final_min = globals().get('FINAL_KCOR_MIN', 1.0)
                    for _final_min in sa_final_mins:
                        try:
                            globals()['FINAL_KCOR_MIN'] = float(_final_min)
                        except Exception:
                            globals()['FINAL_KCOR_MIN'] = prev_final_min
                        out_sh_sa = build_kcor_rows(df2, sh, dual_print)
                        # Compute KCOR_o and merge into SA output
                        kcor_o_sa = build_kcor_o_rows(df2, sh)
                        if not kcor_o_sa.empty and not out_sh_sa.empty:
                            out_sh_sa = pd.merge(
                                out_sh_sa,
                                kcor_o_sa,
                                on=["EnrollmentDate","Date","YearOfBirth","Dose_num","Dose_den"],
                                how="left"
                            )
                        out_sh_sa["param_slope_start"] = int(start_val)
                        out_sh_sa["param_slope_length"] = int(length_val)
                        out_sh_sa["param_final_kcor_min"] = float(globals().get('FINAL_KCOR_MIN', 1.0))
                        produced_outputs.append(out_sh_sa)
                    globals()['FINAL_KCOR_MIN'] = prev_final_min
                if skip_remaining_lengths:
                    # Stop trying larger lengths for this start
                    pass
            # Append and continue to next sheet
            if produced_outputs:
                all_out.append(pd.concat(produced_outputs, ignore_index=True))
            else:
                all_out.append(build_kcor_rows(df, sh, dual_print))
            continue
        else:
            # Dynamic anchor slope calculation (default).
            slopes = compute_group_slopes_dynamic(df, sh, dual_print)
        
        # Debug: Show computed slopes
        if DEBUG_VERBOSE:
            dual_print(f"\n[DEBUG] Computed slopes for sheet {sh}:")
            dose_pairs = get_dose_pairs(sh)
            max_dose = max(max(pair) for pair in dose_pairs)
            for dose in range(max_dose + 1):
                for yob in sorted(df["YearOfBirth"].unique()):
                    if (yob, dose) in slopes:
                        dual_print(f"  YoB {yob}, Dose {dose}: slope = {slopes[(yob, dose)]:.6f}")
            dual_print()
        
        df = adjust_mr(df, slopes, t0=SLOPE_ANCHOR_T)
        
        # Debug: Show MR values week by week, especially weeks with no deaths
        # if DEBUG_VERBOSE:
        #     print(f"\n[DEBUG] MR values by week for sheet {sh}:")
        #     for dose in df["Dose"].unique():
        #         dose_data = df[df["Dose"] == dose].sort_values("DateDied")
        #         print(f"  Dose {dose}:")
        #         print(f"    Original MR range: {dose_data['MR'].min():.6f} to {dose_data['MR'].max():.6f}")
        #         print(f"    Adjusted MR range: {dose_data['MR_adj'].min():.6f} to {dose_data['MR_adj'].max():.6f}")
        #     
        #     # Show weeks with extreme MR_adj values
        #     extreme_mr = dose_data[dose_data["MR_adj"] > 100]
        #     if not extreme_mr.empty:
        #         print(f"    WARNING: {len(extreme_mr)} weeks with MR_adj > 100:")
        #         for _, row in extreme_mr.head(5).iterrows():
        #             print(f"      Date: {row['DateDied']}, Original MR: {row['MR']:.6f}, MR_adj: {row['MR_adj']:.6f}, PT: {row['PT']:.6f}, Dead: {row['Dead']:.6f}")
        #     
        #     # Show weeks with no deaths but high MR_adj
        #     no_deaths_high_mr = dose_data[(dose_data["Dead"] == 0) & (dose_data["MR_adj"] > 10)]
        #     if not no_deaths_high_mr.empty:
        #         print(f"    WARNING: {len(no_deaths_high_mr)} weeks with no deaths but MR_adj > 10:")
        #         for _, row in no_deaths_high_mr.head(5).iterrows():
        #             print(f"      Date: {row['DateDied']}, Original MR: {row['MR']:.6f}, MR_adj: {row['MR_adj']:.6f}, PT: {row['PT']:.6f}, Dead: {row['Dead']:.6f}")
        #     
        #     print()

        # Apply slope correction to each individual MR value
        def apply_slope_correction_to_mr(row):
            slope = slopes.get((row["YearOfBirth"], row["Dose"]), 0.0)
            # Clip slope to prevent overflow
            slope = np.clip(slope, -10.0, 10.0)
            scale_factor = safe_exp(-slope * (row["t"] - float(SLOPE_ANCHOR_T)))
            return row["MR"] * scale_factor
        
        # Add slope and scale factor columns for transparency
        df["slope"] = df.apply(lambda row: slopes.get((row["YearOfBirth"], row["Dose"]), 0.0), axis=1)
        
        # Clip extreme slope values to prevent overflow
        df["slope"] = np.clip(df["slope"], -10.0, 10.0)  # Reasonable bounds for mortality slopes
        
        df["scale_factor"] = df.apply(lambda row: safe_exp(-df["slope"].iloc[row.name] * (row["t"] - float(KCOR_NORMALIZATION_WEEK))), axis=1)
        
        df["MR_adj"] = df.apply(apply_slope_correction_to_mr, axis=1)
        
        # Apply discrete cumulative-hazard transform for mathematical exactness
        # Clip MR_adj to avoid log(0) and ensure numerical stability
        df["hazard"] = -np.log(1 - df["MR_adj"].clip(upper=0.999))
        
        # Apply DYNAMIC_HVE_SKIP_WEEKS to accumulation start
        df["hazard_eff"] = np.where(df["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df["hazard"], 0.0)
        df["MR_eff"] = np.where(df["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df["MR"], 0.0)
        
        # Calculate cumulative hazard (mathematically exact, not approximation)
        df["CH"] = df.groupby(["YearOfBirth","Dose"]) ["hazard_eff"].cumsum()
        
        # Keep unadjusted data for comparison
        df["CH_actual"] = df.groupby(["YearOfBirth","Dose"]) ["MR_eff"].cumsum()
        
        # Keep cumD_adj for backward compatibility (now represents adjusted cumulative deaths)
        df["cumPT"] = df.groupby(["YearOfBirth","Dose"]) ["PT"].cumsum()
        df["cumD_adj"] = df["CH"] * df["cumPT"]
        
        # Keep unadjusted data for comparison
        df["cumD_unadj"] = df.groupby(["YearOfBirth","Dose"]) ["Dead"].cumsum()
        
        # Collect debug data for this sheet (after all columns are created)
        if DEBUG_VERBOSE:
            dose_pairs = get_dose_pairs(sh)
            max_dose = max(max(pair) for pair in dose_pairs)
            for dose in range(max_dose + 1):
                # Get all data for this dose (not aggregated by date yet)
                dose_df = df[df["Dose"] == dose]
                
                # Aggregate across sexes for each dose/date/age combination
                dose_data = dose_df.groupby(["DateDied", "YearOfBirth"]).agg({
                    "ISOweekDied": "first",
                    "Dead": "sum",
                    "Alive": "sum", 
                    "PT": "sum",
                    "MR": "mean",  # Average MR across sexes
                    "MR_adj": "mean",  # Average MR_adj across sexes
                    "CH": "mean",  # Average CH across sexes
                    "CH_actual": "mean",  # Average CH_actual across sexes
                    "hazard": "mean",  # Average hazard across sexes
                    "slope": "mean",  # Average slope across sexes
                    "scale_factor": "mean",  # Average scale factor across sexes
                    "cumD_adj": "sum",  # Sum cumulative deaths
                    "cumD_unadj": "sum",  # Sum unadjusted cumulative deaths
                    "cumPT": "sum",  # Sum cumulative person-time
                    "MR_smooth": "mean",  # Average smoothed MR across sexes
                    "t": "first"  # Time index (should be same for all sexes
                }).reset_index().sort_values(["DateDied", "YearOfBirth"])
                
                for _, row in dose_data.iterrows():
                    # Calculate smoothed adjusted MR using the slope
                    # Use the actual age from the row for slope lookup
                    slope = slopes.get((row["YearOfBirth"], dose), 0.0)
                    smoothed_adj_mr = row["MR_smooth"] * safe_exp(-slope * (row["t"] - float(KCOR_NORMALIZATION_WEEK)))
                    
                    debug_data.append({
                        "EnrollmentDate": sh,  # Add enrollment date column
                        "Date": row["DateDied"].date(),  # Use standard pandas date format (was working perfectly - DON'T TOUCH)
                        "YearOfBirth": row["YearOfBirth"],  # Add year of birth column
                        "Dose": dose,  # Add dose column
                        "ISOweek": row["ISOweekDied"],
                        "Dead": row["Dead"],
                        "Alive": row["Alive"],
                        "MR": row["MR"],
                        "MR_adj": row["MR_adj"],
                        "Cum_MR": row["CH"],
                        "Cum_MR_Actual": row["CH_actual"],
                        "Hazard": row["hazard"],
                        "Slope": row["slope"],
                        "Scale_Factor": row["scale_factor"],
                        "Cumu_Adj_Deaths": row["cumD_adj"],
                        "Cumu_Unadj_Deaths": row["cumD_unadj"],
                        "Cumu_Person_Time": row["cumPT"],
                        "Smoothed_Raw_MR": row["MR_smooth"],
                        "Smoothed_Adjusted_MR": smoothed_adj_mr,
                        "Time_Index": row["t"]
                    })
        
        # Debug: Print detailed CH calculation info
        # if DEBUG_VERBOSE:
        #     print(f"\n[DEBUG] CH calculation details for sheet {sh}:")
        #     for dose in df["Dose"].unique():
        #         dose_data = df[df["Dose"] == dose]
        #         if not dose_data.empty:
        #             print(f"  Dose {dose}:")
        #             print(f"    PT range: {dose_data['PT'].min():.6f} to {dose_data['PT'].max():.6f}")
        #             print(f"    MR_adj range: {dose_data['MR_adj'].min():.6f} to {dose_data['MR_adj'].max():.6f}")
        #             print(f"    D_adj range: {dose_data['D_adj'].min():.6f} to {dose_data['D_adj'].max():.6f}")
        #             print(f"    cumD_adj range: {dose_data['cumD_adj'].min():.6f} to {dose_data['cumD_adj'].max():.6f}")
        #             print(f"    cumPT range: {dose_data['cumPT'].min():.6f} to {dose_data['cumPT'].max():.6f}")
        #             print(f"    CH range: {dose_data['CH'].min():.6f} to {dose_data['CH'].max():.6f}")
        #             print()
        
        # Debug: Check for extreme CH values
        # extreme_ch = df[df["CH"] > 1000]
        # if not extreme_ch.empty:
        #     print(f"\n[DEBUG] Extreme CH values detected in sheet {sh}:")
        #     for _, row in extreme_ch.head(3).iterrows():
        #         print(f"  Age: {row['YearOfBirth']}, Dose: {row['Dose']}, CH: {row['CH']:.6f}")
        #         print(f"    cumD_adj: {row['cumD_adj']:.6f}, cumPT: {row['cumPT']:.6f}")
        #         print(f"    MR_adj: {row['MR_adj']:.6f}, PT: {row['PT']:.6f}")
        #     print()

        out_sh = build_kcor_rows(df, sh, dual_print)
        # Compute KCOR_o on prepared df and merge into output
        kcor_o = build_kcor_o_rows(df, sh)
        if not kcor_o.empty and not out_sh.empty:
            out_sh = pd.merge(
                out_sh,
                kcor_o,
                on=["EnrollmentDate","Date","YearOfBirth","Dose_num","Dose_den"],
                how="left"
            )
        all_out.append(out_sh)
        # Collect per-pair deaths details for new output sheet
        pair_details = build_kcor_o_deaths_details(df, sh)
        if not pair_details.empty:
            pair_deaths_all.append(pair_details)

    # Combine all results
    combined = pd.concat(all_out, ignore_index=True).sort_values(["EnrollmentDate","YearOfBirth","Dose_num","Dose_den","Date"])
    
    # Create debug DataFrame from collected data
    if DEBUG_VERBOSE:
        debug_df = pd.DataFrame(debug_data)
        # print(f"\n[DEBUG] Created debug sheet with {len(debug_df)} rows")
        # print(f"  Date range: {debug_df['Date'].min()} to {debug_df['Date'].max()}")
        # print(f"  Date column dtype: {debug_df['Date'].dtype}")
        # print(f"  Sample Date values: {debug_df['Date'].head(3).tolist()}")
        # print(f"  Doses included: {debug_df['ISOweek'].nunique()} unique ISO weeks")

    # Debug: Show Date column info for main sheets
    # print(f"\n[DEBUG] Main sheets Date column info:")
    # print(f"  Date column dtype: {combined['Date'].dtype}")
    # print(f"  Sample Date values: {combined['Date'].head(3).tolist()}")
    # print(f"  Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    


    # Report KCOR values at reporting dates for each dose combo and age for ALL sheets
    dual_print("\n" + "="*80)
    dual_print("KCOR VALUES AT REPORTING DATES - ALL SHEETS")
    dual_print("="*80)
    
    # Ensure Date is datetime
    combined["Date"] = pd.to_datetime(combined["Date"]) 
    
    # Iterate through each sheet and report at its configured reporting date
    for sheet_name in sorted(combined["EnrollmentDate"].unique()):
        target_str = KCOR_REPORTING_DATE.get(sheet_name)
        if target_str:
            try:
                target_dt = pd.to_datetime(target_str)
            except Exception:
                target_dt = None
        else:
            target_dt = None
        sheet_data_all = combined[combined["EnrollmentDate"] == sheet_name]
        if sheet_data_all.empty:
            continue
        # Choose the closest available date to the configured reporting date; fallback to latest
        if target_dt is not None and not sheet_data_all.empty:
            diffs = (sheet_data_all["Date"] - target_dt).abs()
            idxmin = diffs.idxmin()
            report_date = sheet_data_all.loc[idxmin, "Date"]
        else:
            report_date = sheet_data_all["Date"].max()
        dual_print(f"\nSheet: {sheet_name} — Reporting date: {report_date.strftime('%Y-%m-%d')}")
        dual_print("=" * 60)
        end_data = sheet_data_all[sheet_data_all["Date"] == report_date]
        
        # Get dose pairs for this specific sheet
        dose_pairs = get_dose_pairs(sheet_name)
        
        for (dose_num, dose_den) in dose_pairs:
            dual_print(f"\nDose combination: {dose_num} vs {dose_den} [{sheet_name}]")
            dual_print("-" * 50)
            dual_print(f"{'YoB':>15} | KCOR [95% CI] | KCOR_o")
            dual_print("-" * 50)
            
            # Get data for this dose combination and sheet at reporting date
            dose_data = end_data[
                (end_data["Dose_num"] == dose_num) & 
                (end_data["Dose_den"] == dose_den)
            ]
            
            if dose_data.empty:
                dual_print("  No data available for this dose combination")
                continue
            
            # Show results by age (including ASMR = 0)
            for age in sorted(dose_data["YearOfBirth"].unique()):
                age_data = dose_data[dose_data["YearOfBirth"] == age]
                if not age_data.empty:
                    kcor_val = age_data["KCOR"].iloc[0]
                    ci_lower = age_data["CI_lower"].iloc[0]
                    ci_upper = age_data["CI_upper"].iloc[0]
                    # KCOR_o may be missing for pooled rows; print '-' in that case
                    try:
                        kcor_o_val = age_data.get("KCOR_o", pd.Series([np.nan])).iloc[0]
                    except Exception:
                        kcor_o_val = np.nan
                    kcor_o_str = "-" if not (isinstance(kcor_o_val, (int, float)) and np.isfinite(kcor_o_val)) else f"{kcor_o_val:.4f}"
                    
                    if age == 0:
                        age_label = "ASMR (pooled)"
                    elif age == -1:
                        age_label = "(unknown)"
                    else:
                        age_label = f"{age}"
                    
                    dual_print(f"  {age_label:15} | {kcor_val:8.4f} [{ci_lower:.3f}, {ci_upper:.3f}] | {kcor_o_str}")
    
    dual_print("="*80)

    # write (standard mode handled later with retry block)

    # Write main analysis file with retry logic
    max_retries = 3
    retry_count = 0
    
    if not _is_sa_mode():
        while retry_count < max_retries:
            try:
                # Write to /tmp (when available) to avoid slow/locked Windows target paths, then move atomically
                tmp_dir = "/tmp" if os.path.isdir("/tmp") else os.path.dirname(out_path) or "."
                # Ensure suffix remains .xlsx so engine selection works
                base_no_ext, _ = os.path.splitext(os.path.basename(out_path))
                tmp_base = base_no_ext + ".tmp.xlsx"
                tmp_path = os.path.join(tmp_dir, tmp_base)
                with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                    # Add About sheet first
                    about_data = {
                    "Field": [
                        "KCOR Version",
                        "Analysis Date", 
                        "Input File",
                        "Output File",
                        "",
                        "Configuration Parameters (effective):",
                        "  FINAL_KCOR_MIN",
                        "  FINAL_KCOR_DATE",
                        "  KCOR_NORMALIZATION_WEEK",
                        "  SLOPE_ANCHOR_T",
                        "  SLOPE_WINDOW_SIZE",
                        "  MA_TOTAL_LENGTH",
                        "  CENTERED",
                        "  YEAR_RANGE",
                        "  ENROLLMENT_DATES",
                        "  DEBUG_VERBOSE",
                        "  OVERRIDE_DOSE_PAIRS",
                        "  OVERRIDE_YOBS",
                        "",
                        "Methodology Information:",
                        "",
                        "1. Data Preprocessing:",
                        "   - Enrollment Date Filtering: Data processing starts from enrollment date",
                        "   - Sex Aggregation: Mortality data aggregated across sexes",
                        "   - Smoothing: 8-week centered moving average applied to raw mortality rates",
                        "",
                        "2. Slope Calculation:",
                        "   - Uses predefined anchor points (e.g., weeks 53 and 114)",
                        "   - Geometric mean of smoothed MR values within ±2 week windows",
                        "   - Formula: r = (1/Δt) × ln(B̃/Ã)",
                        "",
                        "3. Mortality Rate Adjustment:",
                        "   - MR_adj = MR × exp(-slope × (t - t0))",
                        "   - t0 = baseline week (typically week 4)",
                        "",
                        "4. KCOR Computation (v4.1):",
                        "   - Step 1: MR_adj = MR × exp(-slope × (t - t0))",
                        "   - Step 2: hazard = -ln(1 - MR_adj) [clipped to 0.999]",
                        "   - Step 3: CH = cumsum(hazard)",
                        "   - KCOR = CH_v / CH_u",
                        "",
                        "5. Uncertainty Quantification:",
                        "   - 95% Confidence intervals using proper uncertainty propagation",
                        "   - Binomial variance approximation for death counts",
                        "",
                        "Output Sheets:",
                        "- dose_pairs: KCOR values for all dose comparisons",
                        "  Columns (dose_pairs):",
                        "    - EnrollmentDate: Enrollment period identifier (e.g., 2021_24)",
                        "    - ISOweekDied: ISO week number of death",
                        "    - Date: Calendar date",
                        "    - YearOfBirth: Birth year (0 indicates ASMR pooled across ages)",
                        "    - Dose_num: Numerator dose group",
                        "    - Dose_den: Denominator dose group",
                        "    - KCOR: Hazard ratio (CH_num/CH_den) normalized to baseline (unitless)",
                        "    - CI_lower: 95% lower confidence bound for KCOR",
                        "    - CI_upper: 95% upper confidence bound for KCOR",
                        "    - MR_num: Weekly mortality rate (annualized per 100,000 for display)",
                        "    - MR_adj_num: Slope-adjusted MR for numerator (annualized per 100,000)",
                        "    - CH_num: Cumulative hazard for numerator (unitless)",
                        "    - CH_actual_num: Cumulative sum of unadjusted weekly MR for numerator (unitless)",
                        "    - hazard_num: Discrete hazard for numerator (unitless)",
                        "    - slope_num: Estimated slope used for numerator",
                        "    - scale_factor_num: exp(-slope*(t - t0)) used for numerator",
                        "    - MR_smooth_num: Smoothed MR for numerator (annualized per 100,000)",
                        "    - t_num: Week index from enrollment for numerator",
                        "    - MR_den: Weekly mortality rate for denominator (annualized per 100,000)",
                        "    - MR_adj_den: Slope-adjusted MR for denominator (annualized per 100,000)",
                        "    - CH_den: Cumulative hazard for denominator (unitless)",
                        "    - CH_actual_den: Cumulative sum of unadjusted weekly MR for denominator (unitless)",
                        "    - hazard_den: Discrete hazard for denominator (unitless)",
                        "    - slope_den: Estimated slope used for denominator",
                        "    - scale_factor_den: exp(-slope*(t - t0)) used for denominator",
                        "    - MR_smooth_den: Smoothed MR for denominator (annualized per 100,000)",
                        "    - t_den: Week index from enrollment for denominator",
                        "- by_dose: Individual dose curves with complete methodology transparency", 
                        "- About: This metadata sheet",
                        "",
                        "For more information, see: https://github.com/skirsch/KCOR"
                    ],
                    "Value": [
                        VERSION,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        src_path,
                        out_path,
                        "",
                        f"",
                        f"{FINAL_KCOR_MIN}",
                        f"{FINAL_KCOR_DATE}",
                        f"{KCOR_NORMALIZATION_WEEK}",
                        f"{SLOPE_ANCHOR_T}",
                        f"{SLOPE_WINDOW_SIZE}",
                        f"{MA_TOTAL_LENGTH}",
                        f"{CENTERED}",
                        f"{YEAR_RANGE}",
                        f"{ENROLLMENT_DATES}",
                        f"{DEBUG_VERBOSE}",
                        f"{OVERRIDE_DOSE_PAIRS}",
                        f"{OVERRIDE_YOBS}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        ""
                    ]
                    }
                    # Write main data first to ensure at least one visible sheet exists
                    all_data = combined.copy()
                    all_data["Date"] = all_data["Date"].apply(lambda x: x.date() if hasattr(x, 'date') else x)
                    all_data.to_excel(writer, index=False, sheet_name="dose_pairs")

                    # Then write About and optional debug sheet
                    # Ensure About 'Field' and 'Value' arrays are the same length
                    # by constructing them explicitly and padding Values as needed
                    fields = about_data["Field"]
                    values = [
                        VERSION,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        src_path,
                        out_path,
                        "",
                        "",
                        f"{FINAL_KCOR_MIN}",
                        f"{FINAL_KCOR_DATE}",
                        f"{KCOR_NORMALIZATION_WEEK}",
                        f"{SLOPE_ANCHOR_T}",
                        f"{SLOPE_WINDOW_SIZE}",
                        f"{MA_TOTAL_LENGTH}",
                        f"{CENTERED}",
                        f"{YEAR_RANGE}",
                        f"{ENROLLMENT_DATES}",
                        f"{DEBUG_VERBOSE}",
                        f"{OVERRIDE_DOSE_PAIRS}",
                        f"{OVERRIDE_YOBS}",
                    ]
                    if len(values) < len(fields):
                        values.extend([""] * (len(fields) - len(values)))
                    elif len(values) > len(fields):
                        values = values[:len(fields)]
                    about_df = pd.DataFrame({"Field": fields, "Value": values})
                    about_df.to_excel(writer, index=False, sheet_name="About")
                    if DEBUG_VERBOSE:
                        debug_df.to_excel(writer, index=False, sheet_name="by_dose")

                    # Add dose_pair_deaths details sheet
                    if pair_deaths_all:
                        pair_deaths_df = pd.concat(pair_deaths_all, ignore_index=True)
                    else:
                        pair_deaths_df = pd.DataFrame(columns=[
                            "EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den",
                            "Dead_num","Dead_adj_num","cumD_num","Dead_den","Dead_adj_den","cumD_den",
                            "K_raw_o","KCOR_o","CMRR"
                        ])
                    pair_deaths_df.to_excel(writer, index=False, sheet_name="dose_pair_deaths")
                
                # Move temp file into place (atomic on POSIX; best-effort on Windows)
                try:
                    os.replace(tmp_path, out_path)
                except OSError as e:
                    # Handle cross-device move (EXDEV) by copy + replace
                    if getattr(e, 'errno', None) == 18 or 'cross-device' in str(e).lower():
                        shutil.copyfile(tmp_path, out_path)
                        os.remove(tmp_path)
                    else:
                        # If target is open in Excel, replacement will fail with PermissionError on Windows
                        raise
                dual_print(f"[Done] Wrote {len(combined)} rows to {out_path}")
                break
                
            except PermissionError as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\n❌ Error: Cannot access output file '{out_path}'")
                    print("   This usually means the file is open in Excel or another program.")
                    print(f"   Attempt {retry_count}/{max_retries}")
                    
                    response = input("   Please close the file and press Enter to retry (or 'q' to quit): ").strip().lower()
                    if response == 'q':
                        print("   Exiting...")
                        return combined
                    print("   Retrying...")
                else:
                    print(f"\n❌ Error: Failed to access '{out_path}' after {max_retries} attempts.")
                    print("   Please ensure the file is not open in Excel or another program.")
                    return combined
            except Exception as e:
                print(f"\n❌ Unexpected error writing main analysis file: {e}")
                return combined
    
    # In SA mode, write a compact SA workbook instead of full summary; else create normal summary file
    if _is_sa_mode():
        try:
            # Filter combined to ASMR rows only (YearOfBirth=0) and specified dose pairs if any
            data = combined.copy()
            # Filter YOBs per override: if SA_YOB provided and not just [0], include those ages; else ASMR only (0)
            if OVERRIDE_YOBS is not None:
                data = data[data["YearOfBirth"].isin(OVERRIDE_YOBS)]
            else:
                data = data[data["YearOfBirth"] == 0]
            if OVERRIDE_DOSE_PAIRS is not None:
                mask = False
                for (a,b) in OVERRIDE_DOSE_PAIRS:
                    mask = mask | ((data["Dose_num"]==a) & (data["Dose_den"]==b))
                data = data[mask]
            # For each cohort and dose pair, pick last 2022 date else latest
            data["Date"] = pd.to_datetime(data["Date"])
            out_records = []
            # Capture parameter settings to memorialize in each SA row
            sa_env_cohorts = os.environ.get('SA_COHORTS', '')
            sa_env_pairs = os.environ.get('SA_DOSE_PAIRS', '')
            sa_env_yob = os.environ.get('SA_YOB', '')
            group_cols = ["EnrollmentDate","Dose_num","Dose_den"]
            if "param_slope_start" in data.columns and "param_slope_length" in data.columns:
                group_cols += ["param_slope_start","param_slope_length"]
            if "param_final_kcor_min" in data.columns:
                group_cols += ["param_final_kcor_min"]
            for keys, g in data.groupby(group_cols, sort=False):
                g2022 = g[g["Date"].dt.year == 2022]
                if not g2022.empty:
                    target = g2022[g2022["Date"] == g2022["Date"].max()].iloc[0]
                else:
                    target = g.iloc[-1]
                # Derive slope anchors used (prefer per-row params if present)
                enr = target["EnrollmentDate"]
                off1 = target.get("param_slope_start", None)
                slope_len = target.get("param_slope_length", None)
                # If explicit SA offsets were not provided, dynamic anchors were used
                if pd.isna(off1) or pd.isna(slope_len):
                            off1, slope_len = (None, None)
                # Use row-level param_final_kcor_min when available
                param_final_min = target.get("param_final_kcor_min", FINAL_KCOR_MIN)
                try:
                    param_final_min = float(param_final_min)
                except Exception:
                    param_final_min = FINAL_KCOR_MIN
                out_records.append({
                    "EnrollmentDate": enr,
                    "Dose_num": int(target["Dose_num"]),
                    "Dose_den": int(target["Dose_den"]),
                    "YearOfBirth": int(target["YearOfBirth"]),
                    "Date": target["Date"].date(),
                    "KCOR": target["KCOR"],
                    "CI_lower": target["CI_lower"],
                    "CI_upper": target["CI_upper"],
                    # Memorialize key parameters
                    "param_normalization_week": KCOR_NORMALIZATION_WEEK,
                    "param_ma_total_length": MA_TOTAL_LENGTH,
                    "param_centered": int(bool(CENTERED)),
                    "param_slope_window_size": SLOPE_WINDOW_SIZE,
                    "param_final_kcor_min": param_final_min,
                    "param_final_kcor_date": FINAL_KCOR_DATE,
                    "param_slope_start": off1,
                    "param_slope_length": slope_len,
                    "param_sa_cohorts": sa_env_cohorts,
                    "param_sa_dose_pairs": sa_env_pairs,
                    "param_sa_yob": sa_env_yob
                })
            sa_df = pd.DataFrame(out_records)
            # Write single-sheet SA workbook named KCOR_SA.xlsx in the same directory as out_path
            out_dir = os.path.dirname(out_path)
            sa_path = os.path.join(out_dir, "KCOR_SA.xlsx")
            with pd.ExcelWriter(sa_path, engine="openpyxl") as writer:
                sa_df.to_excel(writer, index=False, sheet_name="sensitivity")
            dual_print(f"[SA] Wrote {len(sa_df)} rows to {sa_path}")
        except Exception as e:
            print(f"\n❌ Error creating SA workbook: {e}")
    else:
        # Standard summary
        create_summary_file(combined, out_path, dual_print)
    
    # Close log file
    log_file_handle.close()
    
    return combined

def create_summary_file(combined_data, out_path, dual_print):
    """Create KCOR_summary.xlsx with one sheet per enrollment date, formatted like console output."""
    import os
    
    # Determine summary file path
    out_dir = os.path.dirname(out_path)
    # Use SA-specific filename to avoid overwriting normal summary during sensitivity runs
    sa_flag = str(os.environ.get("SENSITIVITY_ANALYSIS", "")).strip().lower() in ("1", "true", "yes")
    summary_name = "KCOR_summary_SA.xlsx" if sa_flag else "KCOR_summary.xlsx"
    summary_path = os.path.join(out_dir, summary_name)
    
    dual_print(f"[Summary] Creating summary file: {summary_path}")
    
    # Group data by EnrollmentDate (enrollment date)
    sheets = combined_data["EnrollmentDate"].unique()
    
    # Write summary file with retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
                for sheet_name in sorted(sheets):
                    sheet_data = combined_data[combined_data["EnrollmentDate"] == sheet_name].copy()
                    
                    # Use reporting date per cohort (sheet) similar to console output
                    sheet_data["Date"] = pd.to_datetime(sheet_data["Date"])
                    target_str = KCOR_REPORTING_DATE.get(sheet_name)
                    if target_str:
                        try:
                            target_dt = pd.to_datetime(target_str)
                        except Exception:
                            target_dt = None
                    else:
                        target_dt = None
                    if target_dt is not None and not sheet_data.empty:
                        diffs = (sheet_data["Date"] - target_dt).abs()
                        idxmin = diffs.idxmin()
                        report_date = sheet_data.loc[idxmin, "Date"]
                    else:
                        report_date = sheet_data["Date"].max()
                    latest_data = sheet_data[sheet_data["Date"] == report_date]
                    
                    # Create summary format similar to console output
                    summary_rows = []
                    # Insert a header noting reporting date for this sheet
                    summary_rows.append({
                        "Dose_Combination": f"Reporting date: {report_date.strftime('%Y-%m-%d')}",
                        "YearOfBirth": "",
                        "KCOR": "",
                        "CI_Lower": "",
                        "CI_Upper": ""
                    })
                    
                    # Get unique dose pairs for this sheet
                    dose_pairs = latest_data[["Dose_num", "Dose_den"]].drop_duplicates().sort_values(["Dose_num", "Dose_den"])
                    
                    for _, dose_pair in dose_pairs.iterrows():
                        dose_num, dose_den = dose_pair["Dose_num"], dose_pair["Dose_den"]
                        
                        # Get data for this dose combination
                        dose_data = latest_data[
                            (latest_data["Dose_num"] == dose_num) & 
                            (latest_data["Dose_den"] == dose_den)
                        ].sort_values("YearOfBirth")
                        
                        # Add header row for this dose combination
                        summary_rows.append({
                            "Dose_Combination": f"{dose_num} vs {dose_den}",
                            "YearOfBirth": "",
                            "KCOR": "",
                            "CI_Lower": "",
                            "CI_Upper": ""
                        })
                        
                        # Add data rows for each age group
                        for _, row in dose_data.iterrows():
                            age = row["YearOfBirth"]
                            if age == 0:
                                age_label = "ASMR (pooled)"
                            elif age == -1:
                                age_label = "(unknown)"
                            else:
                                age_label = f"{age}"
                            
                            summary_rows.append({
                                "Dose_Combination": "",
                                "YearOfBirth": age_label,
                                "KCOR": f"{row['KCOR']:.4f}",
                                "CI_Lower": f"{row['CI_lower']:.3f}",
                                "CI_Upper": f"{row['CI_upper']:.3f}"
                            })
                        
                        # Add empty row for separation
                        summary_rows.append({
                            "Dose_Combination": "",
                            "YearOfBirth": "",
                            "KCOR": "",
                            "CI_Lower": "",
                            "CI_Upper": ""
                        })
                    
                    # Create DataFrame and write to Excel
                    summary_df = pd.DataFrame(summary_rows)
                    summary_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    
                    dual_print(f"  - {sheet_name}: {len(dose_pairs)} dose combinations, {len(summary_rows)} summary rows")
            
            # If we get here, the file was written successfully
            dual_print(f"[Summary] Created summary with {len(sheets)} enrollment periods")
            return summary_path
            
        except PermissionError as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"\n❌ Error: Cannot access summary file '{summary_path}'")
                print("   This usually means the file is open in Excel or another program.")
                print(f"   Attempt {retry_count}/{max_retries}")
                
                response = input("   Please close the file and press Enter to retry (or 'q' to quit): ").strip().lower()
                if response == 'q':
                    print("   Exiting...")
                    return None
                print("   Retrying...")
            else:
                print(f"\n❌ Error: Failed to access '{summary_path}' after {max_retries} attempts.")
                print("   Please ensure the file is not open in Excel or another program.")
                return None
        except Exception as e:
            print(f"\n❌ Unexpected error creating summary file: {e}")
            return None
    
    return None

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python KCOR.py <input_file> <output_file> <mode> [log_filename]")
        print("  mode examples: Primary Analysis | Sensitivity Analysis | Negative Control Test")
        sys.exit(2)
    src = sys.argv[1]
    dst = sys.argv[2]
    mode = sys.argv[3]
    log_filename = sys.argv[4] if len(sys.argv) == 5 else "KCOR_summary.log"
    # Propagate mode to header via env for consistent printing
    os.environ['KCOR_MODE'] = mode
    process_workbook(src, dst, log_filename)

if __name__ == "__main__":
    main()

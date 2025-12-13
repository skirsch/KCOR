#!/usr/bin/env python3s
"""
KCOR (Kirsch Cumulative Outcomes Ratio) Analysis Script v5.0

This script analyzes mortality data to compute KCOR values, which are ratios of cumulative
hazards between different dose groups, normalized to 1 at a baseline period.

METHODOLOGY OVERVIEW:
====================

1. DATA PREPROCESSING:
   - Filter data by enrollment date (derived from sheet name, e.g., "2021_24" = 2021, week 24)
   - Aggregate mortality data across sexes for each (YearOfBirth, Dose, DateDied) combination
   - Apply 8-week centered moving average to raw mortality rates for smoothing

2. SLOPE METRICS (Reporting only):
   - Total slope (diagnostic): For each birth-decade group, Alive-weighted average of dose-specific slopes
     at enrollment (t = 0): \( r_{\text{total}} = \frac{\sum_d A_d \, r_d}{\sum_d A_d} \). Printed for diagnostics only; it does
     not alter KCOR computations.

3. NORMALIZATION AND HAZARDS:
   - Legacy anchor-based slope removal and Czech-specific adjustments have been removed.
   - Slope normalization uses the slope8 method: quantile regression depletion-mode normalization for all cohorts.
     Fit window: enrollment_date to SLOPE_FIT_END_ISO (for highest dose, fit uses data from s >= SLOPE_FIT_DELAY_WEEKS).
     Application: normalization applied from s=0 (enrollment) for all cohorts.
     Fit depletion curve: log h(s) = C + k_∞*s + (k_0 - k_∞)*τ*(1 - e^(-s/τ))
     Normalization: h_norm = h * exp(-kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))  [C term excluded so adjustment is 1 at s=0]
     Apply normalization at the hazard level only. Raw MR values are never modified.
   - Hazard is computed directly from raw MR: \( h = -\ln(1 - \text{MR}) \) with clipping for stability.

4. KCOR COMPUTATION:
   - KCOR = (CH_num / CH_den) normalized to 1 at week KCOR_NORMALIZATION_WEEKS (or first available week).
   - CH is the cumulative sum of weekly hazards.

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
- Baseline period (effective normalization week = KCOR_NORMALIZATION_WEEKS + DYNAMIC_HVE_SKIP_WEEKS) represents "normal" conditions
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
between vaccination groups while accounting for underlying time trends. Version 5.1
uses slope8 (quantile regression depletion-mode normalization) and direct hazard computation from raw MR.
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
import csv
import statsmodels.api as sm
from scipy.optimize import least_squares, minimize

# Dependencies: pandas, numpy, openpyxl, statsmodels, scipy

# Core KCOR methodology parameters
# KCOR baseline normalization week (KCOR == 1 at effective normalization week = KCOR_NORMALIZATION_WEEKS + DYNAMIC_HVE_SKIP_WEEKS)
# Effective normalization week = KCOR_NORMALIZATION_WEEKS + DYNAMIC_HVE_SKIP_WEEKS. See also DYNAMIC_HVE_SKIP_WEEKS.

# this is the number of weeks of data to use in the KCOR normalization baseline computation. 
# the accumulation starts after DYNAMIC_HVE_SKIP_WEEKS. 
KCOR_NORMALIZATION_WEEKS = 4     # Number of weeks of data to use to compute the KCOR normalization baseline. 

AGE_RANGE = 10                  # Bucket size for YearOfBirth aggregation (e.g., 10 -> 1920, 1930, ..., 2000)
SLOPE_ANCHOR_T = 0              # Enrollment week index for slope anchoring
EPS = 1e-12                     # Numerical floor to avoid log(0) and division by zero
# HVE is gone after 2 weeks, so to be safe, start accumulating 
# hazards/statistics from the 3rd week of cumulated data.
DYNAMIC_HVE_SKIP_WEEKS = 2      
MR_DISPLAY_SCALE = 52 * 1e5     # Display-only scaling of MR columns (annualized per 100,000)
NEGATIVE_CONTROL_MODE = 0      # When 1, run negative-control age comparisons and skip normal output

# Slope6 method: Time-centered linear quantile regression normalization
# Uses a single global baseline window for fitting (2022-01 to 2024-12), then applies normalization
# using time-centered approach where t=0 is at the centerpoint of the application window
# (enrollment_date to 2024-16). Fits linear median regression first; if slope is negative,
# Uses slope8 quantile regression depletion-mode normalization to handle depletion-driven curvature.
# Quantile regression estimates baseline slope (median) rather than mean, reducing sensitivity to outliers.

# ---------------- Slope6 Configuration Parameters ----------------
SLOPE6_BASELINE_WINDOW_LENGTH_MIN = 30  # Minimum window length in weeks
SLOPE6_BASELINE_WINDOW_LENGTH_MAX = 60  # Maximum window length in weeks
SLOPE6_BASELINE_START_YEAR = 2023       # Focus on late calendar time (2023+)
SLOPE6_MIN_DATA_POINTS = 5              # Minimum data points required for quantile regression fit
SLOPE6_QUANTILE_TAU = 0.5               # Quantile level for quantile regression (0.5 = 50th percentile/median)
SLOPE_FIT_END_ISO = "2024-16"           # Single constant for fit window end (used for both fit and application windows)
SLOPE6_APPLICATION_END_ISO = SLOPE_FIT_END_ISO  # Rightmost endpoint for determining centerpoint (ISO week format)
# SLOPE7_END_ISO removed - no longer used (slope8 only)
ENABLE_NEGATIVE_SLOPE_FIT = 1           # Legacy flag, not used (slope8 always used)
SLOPE8_QUANTILE_TAU = 0.5               # Quantile level for slope8 quantile regression (0.5 = median)
SLOPE_FIT_DELAY_WEEKS = 15              # Delay in weeks for highest dose fit start (slope8 only)
SLOPE8_MAX_YOB = 1940                   # Maximum YearOfBirth for which slope8 is applied (cohorts >= 1940 use linear fit)

# Optional detailed debug logging for slope8 depletion-mode fits.
# When enabled, each attempted slope8 fit (per cohort) is logged as a CSV row to SLOPE_DEBUG_FILE,
# including summary statistics of the inputs and the fitted parameters.
SLOPE_DEBUG_ENABLED = True
# SLOPE_DEBUG_FILE is set in process_workbook() to the full path in the output directory

# Global dictionary to store ASMR weights per sheet (computed in build_kcor_rows)
# Key: sheet_name, Value: dict mapping YearOfBirth to weight
ASMR_WEIGHTS_BY_SHEET = {}


def format_initial_params(C_init, ka_init, delta_k_init, tau_init):
    """
    Format initial parameter values for logging, converting to float or None.
    Handles None, NaN, and non-finite values consistently.
    
    Parameters
    ----------
    C_init, ka_init, delta_k_init, tau_init : float or None
        Initial parameter estimates
    
    Returns
    -------
    dict
        Dictionary with keys: C_init, ka_init, delta_k_init, tau_init
    """
    def _format_param(val):
        if val is None or (isinstance(val, (int, float, np.number)) and (np.isnan(val) or not np.isfinite(val))):
            return None
        return float(val)
    
    return {
        "C_init": _format_param(C_init),
        "ka_init": _format_param(ka_init),
        "delta_k_init": _format_param(delta_k_init),
        "tau_init": _format_param(tau_init),
    }

def make_slope8_initial_guess(s_valid, logh_valid):
    """
    Compute robust initial parameter guesses for slope8 depletion-mode fit using windowed OLS.
    
    Uses small OLS fits over early and late windows instead of single finite differences,
    dramatically reducing sensitivity to noise and outliers. 
    
    Conditional constraint: When k_0 < 0 (depletion case), enforces k_∞ ≥ k_0 (delta_k ≥ 0).
    When k_0 >= 0, allows k_inf < k_0 (delta_k can be negative).
    Gracefully falls back to global linear slope for non-depletion cases.
    
    Parameters
    ----------
    s_valid : array-like
        Time values (weeks since enrollment), already filtered for valid values
    logh_valid : array-like
        Log-hazard values aligned with s_valid, already filtered for valid values
    
    Returns
    -------
    tuple
        (C_init, k_0_init, delta_k_init, tau_init) - initial parameter estimates
    """
    s_valid = np.asarray(s_valid, dtype=float)
    logh_valid = np.asarray(logh_valid, dtype=float)
    
    # Step 0: Sort data to ensure proper ordering
    order = np.argsort(s_valid)
    s_valid = s_valid[order]
    logh_valid = logh_valid[order]
    
    n = len(s_valid)
    
    # Step 1: Global linear fit as baseline/fallback
    A = np.vstack([s_valid, np.ones_like(s_valid)]).T
    b_global, a_global = np.linalg.lstsq(A, logh_valid, rcond=None)[0]
    
    if n < 5:
        # Too few points: almost linear fallback
        C_init = float(a_global)
        k_0_init = float(b_global)
        delta_k_init = 1e-4
        tau_init = 10.0
        return C_init, k_0_init, delta_k_init, tau_init
    
    # Step 2: Early and late window OLS fits
    # Window size: ~1/3 of data but at least 3 points
    w = max(n // 3, 3)
    
    # Early window: first w points
    s_early = s_valid[:w]
    h_early = logh_valid[:w]
    b0, a0 = np.linalg.lstsq(
        np.vstack([s_early, np.ones_like(s_early)]).T,
        h_early,
        rcond=None
    )[0]
    
    # Late window: last w points
    s_late = s_valid[-w:]
    h_late = logh_valid[-w:]
    b_inf, a_inf = np.linalg.lstsq(
        np.vstack([s_late, np.ones_like(s_late)]).T,
        h_late,
        rcond=None
    )[0]
    
    # Step 3: Derive k_0_init and delta_k_init
    k_0_init = float(b0)
    k_inf_init = float(b_inf)
    
    # Conditional constraint: only enforce k_∞ ≥ k_0 when k_0 < 0 (depletion case)
    # When k_0 >= 0, allow k_inf < k_0 (delta_k can be negative)
    if k_0_init < 0:
        # Depletion case: enforce k_∞ ≥ k_0
        # If pattern is violated (slope more negative late than early), collapse to global slope
        if k_inf_init < k_0_init:
            # Use global slope as both and tiny curvature
            k_0_init = float(b_global)
            k_inf_init = float(b_global)
        delta_k_init = max(k_inf_init - k_0_init, 1e-4)  # small but >= 0 for depletion case
    else:
        # Non-depletion case (k_0 >= 0): allow delta_k to be negative
        delta_k_init = k_inf_init - k_0_init  # Can be negative
    
    # Step 4: Tau based on data span, clamped to reasonable range
    span = max(s_valid) - min(s_valid)
    if span <= 0:
        span = 10.0  # arbitrary small default
    tau_init = span / 3.0
    tau_init = min(max(tau_init, 2.0), 52.0)  # between 2 weeks and 1 year
    
    # Step 5: C from early intercept (evaluated at s=0)
    C_init = float(a0)
    
    return C_init, k_0_init, delta_k_init, tau_init

def log_slope7_fit_debug(record: dict) -> None:
    """
    Append a CSV-formatted debug record for a slope8 (or related) fit to SLOPE_DEBUG_FILE.
    Note: Function name kept as log_slope7_fit_debug for backward compatibility.
    We log summary stats of s_values and logh_values (min/max/mean) plus key fit parameters.
    Failures in logging are silently ignored so as not to affect the main pipeline.
    """
    if not SLOPE_DEBUG_ENABLED:
        return
    try:
        # Extract raw sequences if present and compute simple summaries
        s_vals = record.pop("s_values", None)
        logh_vals = record.pop("logh_values", None)

        def _summaries(arr):
            if arr is None:
                return (None, None, None)
            a = np.asarray(list(arr), dtype=float)
            if a.size == 0 or not np.isfinite(a).any():
                return (None, None, None)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return (None, None, None)
            return (float(a.min()), float(a.max()), float(a.mean()))

        s_min, s_max, s_mean = _summaries(s_vals)
        lh_min, lh_max, lh_mean = _summaries(logh_vals)

        # Standardized column order for easy comparison
        columns = [
            "enrollment_date",
            "mode",
            "YearOfBirth",
            "Dose",
            "n_points",
            "b_original",
            "C",
            "ka",
            "kb",
            "tau",
            "rms_error",
            "mad_error",
            "medae",
            "rmedae",
            "fit_quality",
            "note",
            "error",
            "C_init",
            "ka_init",
            "delta_k_init",
            "tau_init",
            "s_min",
            "s_max",
            "s_mean",
            "logh_min",
            "logh_max",
            "logh_mean",
            "h_first1",
            "h_first2",
            "h_last2",
            "h_last1",
            "t_first1",
            "t_first2",
            "t_last2",
            "t_last1",
            "iso_week_first",
            "iso_week_last",
            "optimizer_success",
            "optimizer_fun",
            "optimizer_nfev",
            "optimizer_nit",
            "optimizer_message",
            "optimizer_status",
            "optimizer_status_meaning",
            "optimizer_warnflag",
            "optimizer_grad_norm",
            "failure_detail",
            "param_C_valid",
            "param_ka_valid",
            "param_kb_valid",
            "param_tau_valid",
        ]

        # Prepare row dict with defaults
        row = {col: None for col in columns}
        for k, v in record.items():
            if k not in row:
                continue
            if isinstance(v, np.generic):
                row[k] = v.item()
            elif isinstance(v, datetime):
                row[k] = v.isoformat()
            else:
                row[k] = v

        # Inject summaries
        row["s_min"] = s_min
        row["s_max"] = s_max
        row["s_mean"] = s_mean
        row["logh_min"] = lh_min
        row["logh_max"] = lh_max
        row["logh_mean"] = lh_mean
        
        # Extract and log first/last few h(t) values for debugging
        h_vals = record.pop("h_values", None)
        if h_vals is not None and len(h_vals) > 0:
            h_arr = np.asarray(list(h_vals), dtype=float)
            h_arr = h_arr[np.isfinite(h_arr)]
            if len(h_arr) >= 2:
                # First 2 and last 2 h(t) values
                row["h_first1"] = float(h_arr[0]) if len(h_arr) > 0 else None
                row["h_first2"] = float(h_arr[1]) if len(h_arr) > 1 else None
                row["h_last2"] = float(h_arr[-2]) if len(h_arr) >= 2 else None
                row["h_last1"] = float(h_arr[-1]) if len(h_arr) >= 1 else None
            else:
                row["h_first1"] = float(h_arr[0]) if len(h_arr) > 0 else None
                row["h_first2"] = None
                row["h_last2"] = None
                row["h_last1"] = float(h_arr[-1]) if len(h_arr) > 0 else None
        else:
            row["h_first1"] = None
            row["h_first2"] = None
            row["h_last2"] = None
            row["h_last1"] = None
        
        # Extract and log first/last few time values for debugging
        if s_vals is not None and len(s_vals) > 0:
            s_arr = np.asarray(list(s_vals), dtype=float)
            s_arr = s_arr[np.isfinite(s_arr)]
            if len(s_arr) >= 2:
                row["t_first1"] = float(s_arr[0]) if len(s_arr) > 0 else None
                row["t_first2"] = float(s_arr[1]) if len(s_arr) > 1 else None
                row["t_last2"] = float(s_arr[-2]) if len(s_arr) >= 2 else None
                row["t_last1"] = float(s_arr[-1]) if len(s_arr) >= 1 else None
            else:
                row["t_first1"] = float(s_arr[0]) if len(s_arr) > 0 else None
                row["t_first2"] = None
                row["t_last2"] = None
                row["t_last1"] = float(s_arr[-1]) if len(s_arr) > 0 else None
        else:
            row["t_first1"] = None
            row["t_first2"] = None
            row["t_last2"] = None
            row["t_last1"] = None
        
        # Extract and log first/last ISO weeks used in fit
        iso_weeks_used = record.pop("iso_weeks_used", None)
        if iso_weeks_used is not None and len(iso_weeks_used) > 0:
            row["iso_week_first"] = str(iso_weeks_used[0]) if len(iso_weeks_used) > 0 else None
            row["iso_week_last"] = str(iso_weeks_used[-1]) if len(iso_weeks_used) > 0 else None
        else:
            row["iso_week_first"] = None
            row["iso_week_last"] = None

        # Append to CSV, writing header if file is new/empty or if header needs updating
        file_exists = Path(SLOPE_DEBUG_FILE).is_file()
        write_header = (not file_exists) or (Path(SLOPE_DEBUG_FILE).stat().st_size == 0)
        
        # Check if header needs updating (if file exists but header doesn't match)
        if file_exists and not write_header:
            try:
                with open(SLOPE_DEBUG_FILE, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    first_line = next(reader, None)
                    if first_line is not None:
                        existing_columns = first_line
                        # Check if all required columns are present
                        missing_columns = [col for col in columns if col not in existing_columns]
                        if missing_columns:
                            # Header is outdated, need to rewrite file with new header
                            # Read all existing data
                            f.seek(0)
                            existing_data = list(csv.DictReader(f))
                            # Rewrite file with new header
                            with open(SLOPE_DEBUG_FILE, "w", newline="", encoding="utf-8") as fw:
                                writer = csv.DictWriter(fw, fieldnames=columns)
                                writer.writeheader()
                                # Write existing data (only columns that exist in both)
                                for old_row in existing_data:
                                    new_row = {col: old_row.get(col, None) for col in columns}
                                    writer.writerow(new_row)
                            write_header = False  # Don't write header again, we just rewrote it
            except Exception:
                # If we can't read/update the header, just append (best effort)
                pass

        with open(SLOPE_DEBUG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        # Debug logging must never break the main computation
        return

KCOR_REPORTING_DATE = {
    '2021-13': '2022-12-31',
    '2021_13': '2022-12-31',
    '2021-20': '2022-12-31',
    '2021_20': '2022-12-31',
    '2021_24': '2022-12-31',
    '2022_06': '2022-12-31',
    '2022_47': '2023-12-31',
}

# Check for Monte Carlo mode
MONTE_CARLO_MODE = str(os.environ.get('MONTE_CARLO', '')).strip().lower() in ('1', 'true', 'yes')

# DATA SMOOTHING:
# Moving-average smoothing parameters removed

# Processing parameters
# NOTE: the 2009 cutoff is because for 10 year processing of the 2000 age group, if we set it to 
# 2000, the 2000 group would NOT include the 2005 cohort. This way, we get 10 year age groups for all the cohorts.
YEAR_RANGE = (1920, 2009)       # Process age groups from start to end year (inclusive)
ENROLLMENT_DATES = None  # List of enrollment dates (sheet names) to process. If None, will be auto-derived from Excel file sheets (excluding _summary and _MFG_ sheets)
DEBUG_DOSE_PAIR_ONLY = None  # Only process this dose pair (set to None to process all)
DEBUG_VERBOSE = True            # Print detailed debugging info for each date
# Slope normalization uses slope8 method (quantile regression depletion-mode normalization for all cohorts)
# removed legacy Czech unvaccinated MR adjustment toggle

# ----------------------------------------------------------

# Optional overrides via environment for sensitivity/plumbing without CLI changes
# SA_COHORTS: comma-separated list of sheet names, e.g., "2021_24,2022_06"
# SA_DOSE_PAIRS: semicolon-separated list of pairs as a,b; e.g., "1,0;2,0"
# SA_YOB: "0" for ASMR only, or range "start,end,step", or list "y1,y2,y3"
OVERRIDE_DOSE_PAIRS = None
OVERRIDE_YOBS = None


# ---------------- Configuration Parameters ----------------
# Version information
VERSION = "v5.4"                # KCOR version number

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
#        - Implements KCOR scaling based on FINAL_KCOR_DATE and FINAL_KOR_MIN parameters
#        - Corrects for baseline normalization issues where unsafe vaccines create artificially high baseline mortality rates
# v4.4 - Added enrollment cohort 2022_47 with dose comparisons 4 vs 3,2,1,0
#        - Default processing now includes four cohorts: 2021_13, 2021_24, 2022_06, 2022_47
#        - Slope calculation simplified to single-window method (no dynamic anchors)
# v4.5 - Removed legacy anchor-based slope adjustments and Czech-specific corrections in favor of
#        - slope3 hazard-level normalization (slope2 windows with lowest-N averaging) and direct hazard computation from raw MR.
#        - (Note: slope3 was replaced by slope4 in v4.8)
# v4.6 - Added enrollment cohort 2021-W20 with dose comparisons 2 vs 1,0
#        - Default processing now includes five cohorts: 2021-13, 2021-W20, 2021-24, 2022-06, 2022-47
#        - Slope calculation simplified to single-window method (no dynamic anchors)
#        - Changed DYNAMIC_HVE_SKIP_WEEKS to 3 to start accumulating hazards/statistics from the 4th week of cumulated data.
# v4.7 - Implemented Slope3 method for improved slope estimation
#        - Instead of averaging all values in each window, now averages the lowest N values (default: 5)
#        - More robust to outliers and noise in the data
#        - Configurable via SLOPE3_MIN_VALUES parameter
#        - Added "All Ages" calculation (YearOfBirth = -2) that aggregates all ages into a single cohort
#        - Different from ASMR pooling: All Ages treats all ages as one cohort, while ASMR weights across age groups
#        - All Ages calculation displayed right after ASMR (direct) in console and summary outputs
# v4.8 - Replaced Slope3 with Slope4 method (2024-12-XX)
#        - Changed from averaging lowest N values to using geometric mean of all values in each window
#        - Geometric mean provides better representation of central tendency for hazard values
#        - More mathematically sound approach that naturally handles the multiplicative nature of hazard rates
#        - Removed SLOPE3_MIN_VALUES parameter (no longer needed)
# v4.9 - Replaced Slope4 with Slope5 independent flat-slope normalization method
#        - Single global baseline window automatically selected
#        - Each cohort normalized independently to achieve zero log-hazard slope using Quantile Regression (median)
#        - Each cohort's own drift slope β_c is estimated and removed, centered at pivot time t_0
#        - Normalization formula: h_c^norm(t) = e^{a_c} * e^{b_c*t} * h_c(t)
#        - Provides mathematically precise, reproducible method per KCOR_slope5_RMS.md specification
#        - Removed BASE_W1/BASE_W2/BASE_W3/BASE_W4 fixed windows (replaced with automatic selection)
# v5.0 - Replaced Slope5 with Slope6 time-centered linear/quadratic quantile regression normalization
#        - Fit window: 2022-01 to 2024-12 (same as slope5) for regression fitting
#        - Application window: enrollment_date to 2024-16 for determining centerpoint
#        - Time-centered approach: t_mean = mean(t) over application window, t_c = t - t_mean
#        - Linear median regression if b_lin >= 0, quadratic with c >= 0 if b_lin < 0
#        - Normalization: linear mode uses h_norm = h * exp(-b_lin * t_c)
#        - Normalization: quadratic mode uses h_norm = h * exp(-(b * t_c + c * t_c^2))
#        - Provides robust handling of depletion-driven curvature while preserving frailty model constraints
#        - Per kcor_slope6_spec.md and kcor_slope6_helpers.md specification
# v5.1 - Replaced quadratic mode with Slope7 depletion-mode normalization for b < 0 cohorts
#        - Uses Levenberg-Marquardt nonlinear least squares to fit exponential relaxation depletion curve
#        - Fit window = deployment window (enrollment to slope7_end_ISO)
#        - Time axis s = weeks since enrollment (no centering)
#        - Parameters: C, ka (k_0), kb (k_∞), tau (τ)
#        - Provides robust handling of depletion-driven curvature while preserving frailty model constraints
# v5.2 - Added Slope8 quantile regression method as diagnostic tool
#        - Uses quantile regression with check loss instead of L2 loss for robustness to outliers
#        - Uses scipy.optimize.minimize with L-BFGS-B method and finite bounds
#        - Fit window = deployment window (enrollment to SLOPE_FIT_END_ISO) for all doses
#        - Special case: For highest dose, fit uses data from s >= SLOPE_FIT_DELAY_WEEKS (default 15 weeks)
#        - Normalization applies from s=0 (enrollment) for all cohorts, including highest dose
#        - Results logged to debug CSV but not yet applied for normalization
#        - Provides alternative diagnostic method alongside linear, slope7 (TRF), and slope7 (LM)
# v5.3 - Switched to Slope8 as primary normalization method for all cohorts
#        - Replaces Slope6/Slope7 decision logic with Slope8 for all cohorts
#        - Tracks abnormal fits via optimizer diagnostics (status=5 or not success)
#        - Flags KCOR results affected by abnormal fits with asterisk (*) in console/log and summary spreadsheet
#        - Slope8 fit window starts later for highest dose, but normalization applied from s=0 for all cohorts

# latest change was setting DYNAMIC_HVE_SKIP_WEEKS to 3 to start accumulating hazards/statistics from the 4th week of cumulated data.



# KCOR normalization fine-tuning parameters
# Removed FINAL_KCOR_MIN/FINAL_KOR_DATE scaling

# Dynamic anchors removed

# Legacy simple window-based slope calculation removed (SLOPE_WINDOW_SIZE obsolete)

# Reporting date lookup for KCOR summary/console per cohort (sheet name)
# For the first three cohorts, use end of 2022; for 2022_47, use one year later (end of 2023)

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
        KCOR_NORMALIZATION_WEEKS = int(_env_anchor)
        if DEBUG_VERBOSE:
            print(f"[DEBUG] Overriding KCOR_NORMALIZATION_WEEKS via SA_ANCHOR_WEEKS: {KCOR_NORMALIZATION_WEEKS}")
    # Effective normalization week accounting for skip weeks: normalization happens KCOR_NORMALIZATION_WEEKS weeks after accumulation starts
    # Subtract 1 because accumulation starts at DYNAMIC_HVE_SKIP_WEEKS, so the Nth week of accumulation is at offset (DYNAMIC_HVE_SKIP_WEEKS + N - 1)
    KCOR_NORMALIZATION_WEEKS_EFFECTIVE = KCOR_NORMALIZATION_WEEKS + DYNAMIC_HVE_SKIP_WEEKS - 1
    # removed MA smoothing env overrides
    # removed SA_SLOPE_WINDOW_SIZE override
    # removed FINAL_KCOR env overrides
    # Legacy CZECH_UNVACCINATED_MR_ADJUSTMENT removed
    # removed SA_SLOPE_NORMALIZE_YOB_LE
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
    elif sheet_name in ("2021-20", "2021_20"):
        # Mid 2021 sheet: max dose is 2, only doses 0, 1, 2
        return [(1,0), (2,0), (2,1)]
    elif sheet_name == "2021_24":
        # Mid 2021 sheet: max dose is 2, only doses 0, 1, 2
        return [(1,0), (2,0), (2,1)]
    elif sheet_name == "2022_06":
        # 2022 sheet: includes dose 3 comparisons (add 3 vs 1)
        return [(1,0), (2,0), (2,1), (3,2), (3,1), (3,0)]
    elif sheet_name in ("2022-26", "2022_26"):
        # 2022 sheet: "5 months after booster" cohort - includes dose 3 comparisons like 2022_06
        return [(1,0), (2,0), (2,1), (3,2), (3,1), (3,0)]
    elif sheet_name == "2022_47":
        # Late 2022 sheet: include all combinations from 2022_06 plus 4 vs lower doses
        return [
            (1,0), (2,0), (2,1), (3,2), (3,1), (3,0),
            (4,3), (4,2), (4,1), (4,0)
        ]
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
    # Note: datetime is already imported at module level
    y, w = s.split("-")
    # ISO week to date: Monday of that ISO week
    return datetime.fromisocalendar(int(y), int(w), 1)

def select_dynamic_anchor_offsets(enrollment_date, df_dates):
    """Deprecated; dynamic anchors removed. Return (None, None)."""
    return (None, None)

def compute_group_slopes_dynamic(df, sheet_name, dual_print_fn=None):
    """Deprecated: return zero slopes; slope2 handles normalization at hazard level."""
    slopes = {}
    for (yob, dose), _ in df.groupby(["YearOfBirth","Dose"], sort=False):
        slopes[(yob, dose)] = 0.0
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


def apply_moving_average(df, window=None, centered=None):
    """Apply centered moving average smoothing to MR values before quantile regression.
    
    Args:
        window: Total length of moving average (e.g., 8 weeks = 4 weeks on either side)
        centered: If True, use centered MA (4 weeks before + 4 weeks after each point)
    """
    df_smooth = df.copy()
    
    # Deprecated: return df unchanged, with MR_smooth equal to MR for compatibility
    df_smooth["MR_smooth"] = df_smooth.get("MR", np.nan)
    
    return df_smooth

def _iso_to_date_slope6(isostr: str):
    """Convert ISO week string (YYYY-WW) to datetime (Monday of that week)."""
    y, w = isostr.split("-")
    return datetime.fromisocalendar(int(y), int(w), 1)

def _iso_week_list_slope6(start_iso: str, end_iso: str):
    """Generate list of ISO week strings from start to end (inclusive)."""
    start_dt = _iso_to_date_slope6(start_iso)
    end_dt = _iso_to_date_slope6(end_iso)
    if end_dt < start_dt:
        raise RuntimeError("slope6 window end before start")
    weeks = []
    cur = start_dt
    while cur <= end_dt:
        iso = cur.isocalendar()
        weeks.append(f"{iso.year}-{int(iso.week):02d}")
        cur = cur + timedelta(weeks=1)
    return weeks

def select_slope6_baseline_window(df_all_sheets_list, dual_print_fn=None):
    """
    Select a single global baseline window for Slope6 normalization (fit window).
    
    Uses a fixed window: 2022-01 to SLOPE_FIT_END_ISO (week 1 of 2022 through the configured end week).
    
    Args:
        df_all_sheets_list: List of (sheet_name, df) tuples containing all enrollment sheets' data
        dual_print_fn: Optional logging function
        
    Returns:
        Tuple (start_iso_week, end_iso_week) in format ("YYYY-WW", "YYYY-WW")
    """
    def _print(msg):
        if dual_print_fn:
            dual_print_fn(msg)
        else:
            print(msg)
    
    # Fixed window: 2022-01 to SLOPE_FIT_END_ISO
    baseline_window = ("2022-01", SLOPE_FIT_END_ISO)
    
    _print(f"SLOPE6_FIT_WINDOW,selected={baseline_window[0]}..{baseline_window[1]} (week 1 of 2022 through {SLOPE_FIT_END_ISO}, fixed)")
    return baseline_window

def fit_linear_median(t, logh, tau=0.5):
    """
    Fit logh ≈ a_lin + b_lin * t_c using median quantile regression.
    
    Parameters
    ----------
    t : array-like
        Time values (1D array).
    logh : array-like
        log(hazard) values aligned with t.
    tau : float
        Quantile level, default 0.5 (median).
    
    Returns
    -------
    a_lin, b_lin, t_mean : floats
        Fitted intercept, slope, and the time centering constant t_mean.
    """
    t = np.asarray(t)
    logh = np.asarray(logh)
    t_mean = t.mean()
    t_c = t - t_mean
    
    # Use statsmodels QuantReg for quantile regression
    X = sm.add_constant(np.array(t_c))  # [1, t_c]
    y = np.array(logh)
    
    qr_model = sm.QuantReg(y, X)
    qr_res = qr_model.fit(q=tau)
    
    a_lin = float(qr_res.params[0])  # intercept
    b_lin = float(qr_res.params[1])   # slope
    
    return a_lin, b_lin, float(t_mean)

def fit_slope7_depletion(s, logh):
    """
    Fit depletion-mode normalization using Trust Region Reflective nonlinear least squares.
    
    Models log-hazard as: log h(s) = C + (k_0 + Δk)*s - Δk*τ*(1 - e^(-s/τ))
    where k_∞ = k_0 + Δk.
    
    Conditional constraint: When k_0 < 0 (depletion case), k_∞ ≥ k_0 (delta_k ≥ 0) is enforced.
    When k_0 ≥ 0, delta_k can be negative (no depletion constraint).
    
    Parameters
    ----------
    s : array-like
        Time values in weeks since enrollment (NOT centered, s=0 at enrollment).
    logh : array-like
        log(hazard) values aligned with s.
    
    Returns
    -------
    (C, k_inf, k_0, tau) : tuple of floats
        Fitted parameters:
        - C: intercept
        - k_inf (kb): long-run background slope = k_0 + Δk
        - k_0 (ka): slope at enrollment (may be negative or positive)
        - tau: depletion timescale in weeks (must be > 0)
        When k_0 < 0, the constraint k_inf ≥ k_0 ensures the slope b(s) is monotonically increasing.
        When k_0 ≥ 0, no such constraint is applied.
        Returns (np.nan, np.nan, np.nan, np.nan) on failure.
    initial_params : tuple
        (C_init, k_0_init, delta_k_init, tau_init) - initial parameter estimates
    """
    s = np.asarray(s, dtype=float)
    logh = np.asarray(logh, dtype=float)
    
    # Remove invalid values
    valid_mask = np.isfinite(s) & np.isfinite(logh)
    if valid_mask.sum() < SLOPE6_MIN_DATA_POINTS:
        return (np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan)
    
    s_valid = s[valid_mask]
    logh_valid = logh[valid_mask]
    
    # Initial parameter estimates
    # C: intercept from first few points
    if len(logh_valid) >= 3:
        C_init = np.mean(logh_valid[:3])
    else:
        C_init = logh_valid[0] if len(logh_valid) > 0 else 0.0
    
    # k_0: initial slope (can be negative)
    if len(s_valid) >= 2:
        k_0_init = (logh_valid[1] - logh_valid[0]) / (s_valid[1] - s_valid[0] + EPS)
    else:
        k_0_init = 0.0
    
    # Δk: difference between long-run and initial slope
    # Conditional constraint: delta_k >= 0 only when k_0 < 0 (depletion case)
    # When k_0 >= 0, delta_k can be negative (no depletion constraint)
    MIN_DELTA_K = -0.1  # Allow negative delta_k when k_0 >= 0
    MIN_TAU = 1e-3      # Minimum value for tau bound (weeks)
    
    if len(logh_valid) >= 5:
        later_slope = (logh_valid[-1] - logh_valid[-3]) / (s_valid[-1] - s_valid[-3] + EPS)
        # Initial guess: if k_0 < 0, ensure delta_k >= 0; otherwise allow negative
        if k_0_init < 0:
            delta_k_init = max(later_slope - k_0_init, 0.001)  # Ensure Δk ≥ 0 for depletion case
        else:
            delta_k_init = later_slope - k_0_init  # Can be negative when k_0 >= 0
    else:
        # If we can't estimate later slope, use small positive for depletion case, zero otherwise
        delta_k_init = 0.001 if k_0_init < 0 else 0.0
    
    # tau: depletion timescale (weeks), initial guess based on data span
    tau_init = max((s_valid.max() - s_valid.min()) / 3.0, max(MIN_TAU * 10, 1.0))  # Ensure well above bound
    
    # Store initial estimates for return (before clipping)
    initial_params = (float(C_init), float(k_0_init), float(delta_k_init), float(tau_init))
    
    # Parameter vector: [C, k_0, Δk, tau]
    p0 = np.array([C_init, k_0_init, delta_k_init, tau_init], dtype=float)
    
    # Bounds: delta_k can be negative when k_0 >= 0, tau > MIN_TAU, C and k_0 unbounded
    lower_bounds = np.array([-np.inf, -np.inf, MIN_DELTA_K, MIN_TAU], dtype=float)
    upper_bounds = np.array([ np.inf,  np.inf,        np.inf,    np.inf], dtype=float)
    
    # Ensure initial guess is within bounds before calling least_squares
    p0 = np.clip(p0, lower_bounds, upper_bounds)
    bounds = (lower_bounds, upper_bounds)
    
    def residual_func(p):
        """Residual function for least squares with conditional constraint."""
        C, k_0, delta_k, tau = p
        
        # Conditional constraint: if k_0 < 0, delta_k must be >= 0 (depletion constraint)
        # If k_0 >= 0, delta_k can be negative (no constraint)
        # Add penalty to residuals if constraint is violated
        penalty = 0.0
        if k_0 < 0 and delta_k < 0:
            # Violation of depletion constraint: add large penalty
            penalty = 1e6 * (abs(delta_k) + 1.0)
        
        # Model: log h(s) = C + (k_0 + Δk)*s - Δk*τ*(1 - exp(-s/τ))
        # This is equivalent to: C + k_∞*s + (k_0 - k_∞)*τ*(1 - exp(-s/τ)) where k_∞ = k_0 + Δk
        k_inf = k_0 + delta_k
        predicted = C + k_inf * s_valid - delta_k * tau * (1.0 - np.exp(-s_valid / (tau + EPS)))
        residuals = logh_valid - predicted
        
        # Add penalty as additional residual component
        if penalty > 0:
            residuals = np.append(residuals, np.sqrt(penalty))
        
        return residuals
    
    try:
        result = least_squares(
            residual_func,
            p0,
            method='trf',  # Trust Region Reflective (supports bounds)
            bounds=bounds,
            max_nfev=1000,
            ftol=1e-8,
            xtol=1e-8
        )
        
        if result.success:
            C, k_0, delta_k, tau = result.x
            # Conditional constraint: only enforce delta_k >= 0 when k_0 < 0 (depletion case)
            # When k_0 >= 0, allow delta_k to be negative
            if k_0 < 0:
                # Depletion case: enforce k_inf >= k_0 (delta_k >= 0)
                delta_k = max(delta_k, 0.0)
            # else: k_0 >= 0, delta_k can be negative (no constraint)
            
            tau = max(tau, MIN_TAU)
            # Compute k_∞ = k_0 + Δk
            k_inf = k_0 + delta_k
            # Verify all parameters are finite
            if np.isfinite(C) and np.isfinite(k_inf) and np.isfinite(k_0) and np.isfinite(tau):
                return (float(C), float(k_inf), float(k_0), float(tau)), initial_params
            else:
                # Parameters are not finite
                return (np.nan, np.nan, np.nan, np.nan), initial_params
        else:
            # Fit did not converge
            return (np.nan, np.nan, np.nan, np.nan), initial_params
    except Exception as e:
        # Log the exception for debugging
        import warnings
        warnings.warn(f"fit_slope7_depletion failed: {str(e)}", RuntimeWarning)
        return (np.nan, np.nan, np.nan, np.nan), initial_params


def fit_slope7_depletion_lm(s, logh):
    """
    Comparator fit: depletion-mode normalization using unbounded Levenberg–Marquardt (method='lm').
    
    Uses the same functional form as fit_slope7_depletion, but WITHOUT bounds and with method='lm'
    for nonlinear least squares. Intended for diagnostics/comparison only; results are logged but
    not currently used for normalization.
    
    Returns
    -------
    (C, k_inf, k_0, tau) : tuple of floats
        Fitted parameters (same as fit_slope7_depletion)
    initial_params : tuple
        (C_init, k_0_init, delta_k_init, tau_init) - initial parameter estimates
    """
    s = np.asarray(s, dtype=float)
    logh = np.asarray(logh, dtype=float)
    
    # Remove invalid values
    valid_mask = np.isfinite(s) & np.isfinite(logh)
    if valid_mask.sum() < SLOPE6_MIN_DATA_POINTS:
        return (np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan)
    
    s_valid = s[valid_mask]
    logh_valid = logh[valid_mask]
    
    # Initial parameter estimates (same strategy as bounded fit, but no clipping)
    if len(logh_valid) >= 3:
        C_init = np.mean(logh_valid[:3])
    else:
        C_init = logh_valid[0] if len(logh_valid) > 0 else 0.0
    
    if len(s_valid) >= 2:
        k_0_init = (logh_valid[1] - logh_valid[0]) / (s_valid[1] - s_valid[0] + EPS)
    else:
        k_0_init = 0.0
    
    if len(logh_valid) >= 5:
        later_slope = (logh_valid[-1] - logh_valid[-3]) / (s_valid[-1] - s_valid[-3] + EPS)
        # Initial guess: if k_0 < 0, ensure delta_k >= 0; otherwise allow negative
        if k_0_init < 0:
            delta_k_init = max(later_slope - k_0_init, 0.001)  # Ensure Δk ≥ 0 for depletion case
        else:
            delta_k_init = later_slope - k_0_init  # Can be negative when k_0 >= 0
    else:
        # If we can't estimate later slope, use small positive for depletion case, zero otherwise
        delta_k_init = 0.001 if k_0_init < 0 else 0.0
    
    tau_init = max((s_valid.max() - s_valid.min()) / 3.0, 1.0)
    
    # Store initial estimates for return
    initial_params = (float(C_init), float(k_0_init), float(delta_k_init), float(tau_init))
    
    p0 = np.array([C_init, k_0_init, delta_k_init, tau_init], dtype=float)
    
    def residual_func(p):
        """Residual function with conditional constraint."""
        C, k_0, delta_k, tau = p
        
        # Conditional constraint: if k_0 < 0, delta_k must be >= 0 (depletion constraint)
        # If k_0 >= 0, delta_k can be negative (no constraint)
        # Add penalty to residuals if constraint is violated
        penalty = 0.0
        if k_0 < 0 and delta_k < 0:
            # Violation of depletion constraint: add large penalty
            penalty = 1e6 * (abs(delta_k) + 1.0)
        
        k_inf = k_0 + delta_k
        predicted = C + k_inf * s_valid - delta_k * tau * (1.0 - np.exp(-s_valid / (tau + EPS)))
        residuals = logh_valid - predicted
        
        # Add penalty as additional residual component
        if penalty > 0:
            residuals = np.append(residuals, np.sqrt(penalty))
        
        return residuals
    
    try:
        result = least_squares(
            residual_func,
            p0,
            method='lm',  # Unbounded Levenberg–Marquardt for comparison
            max_nfev=1000,
            ftol=1e-8,
            xtol=1e-8
        )
        if result.success:
            C, k_0, delta_k, tau = result.x
            # Conditional constraint: only enforce delta_k >= 0 when k_0 < 0 (depletion case)
            # When k_0 >= 0, allow delta_k to be negative
            if k_0 < 0:
                # Depletion case: enforce k_inf >= k_0 (delta_k >= 0)
                delta_k = max(delta_k, 0.0)
            # else: k_0 >= 0, delta_k can be negative (no constraint)
            
            k_inf = k_0 + delta_k
            if np.isfinite(C) and np.isfinite(k_inf) and np.isfinite(k_0) and np.isfinite(tau):
                return (float(C), float(k_inf), float(k_0), float(tau)), initial_params
            else:
                return (np.nan, np.nan, np.nan, np.nan), initial_params
        else:
            return (np.nan, np.nan, np.nan, np.nan), initial_params
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan), initial_params

def fit_slope8_depletion(s, logh):
    """
    Fit depletion-mode normalization using quantile regression with L-BFGS-B optimization.
    
    Models log-hazard as: log h(s) = C + (k_0 + Δk)*s - Δk*τ*(1 - e^(-s/τ))
    where k_∞ = k_0 + Δk.
    
    Conditional constraint: When k_0 < 0 (depletion case), k_∞ ≥ k_0 (delta_k ≥ 0) is enforced.
    When k_0 ≥ 0, delta_k can be negative (no depletion constraint).
    
    Uses quantile (check) loss instead of L2 loss for robustness to outliers.
    Uses improved windowed OLS initial guess algorithm for robust parameter initialization.
    
    Parameters
    ----------
    s : array-like
        Time values in weeks since enrollment (NOT centered, s=0 at enrollment).
    logh : array-like
        log(hazard) values aligned with s.
    
    Returns
    -------
    (C, k_inf, k_0, tau) : tuple of floats
        Fitted parameters:
        - C: intercept
        - k_inf (kb): long-run background slope = k_0 + Δk
        - k_0 (ka): slope at enrollment (may be negative or positive)
        - tau: depletion timescale in weeks (must be > 0)
        When k_0 < 0, the constraint k_inf ≥ k_0 ensures the slope b(s) is monotonically increasing.
        When k_0 ≥ 0, no such constraint is applied.
        Returns (np.nan, np.nan, np.nan, np.nan) on failure.
    initial_params : tuple
        (C_init, k_0_init, delta_k_init, tau_init) - initial parameter estimates
    diagnostics : dict
        Dictionary with optimizer diagnostics:
        - success: bool - whether optimizer reported success
        - fun: float - final loss value
        - nfev: int - number of function evaluations
        - nit: int - number of iterations
        - message: str - optimizer message
        - C_valid: bool - whether C parameter is valid (finite)
        - ka_valid: bool - whether ka parameter is valid (finite)
        - kb_valid: bool - whether kb parameter is valid (finite)
        - tau_valid: bool - whether tau parameter is valid (finite and > EPS)
    """
    s = np.asarray(s, dtype=float)
    logh = np.asarray(logh, dtype=float)
    
    # Remove invalid values
    valid_mask = np.isfinite(s) & np.isfinite(logh)
    if valid_mask.sum() < SLOPE6_MIN_DATA_POINTS:
        diagnostics = {
            "success": False,
            "fun": None,
            "nfev": 0,
            "nit": 0,
            "message": "insufficient_data_points",
            "optimizer_status": None,
            "optimizer_warnflag": None,
            "optimizer_grad_norm": None,
            "failure_detail": "insufficient_data_points",
            "C_valid": False,
            "ka_valid": False,
            "kb_valid": False,
            "tau_valid": False,
        }
        return (np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan), diagnostics
    
    s_valid = s[valid_mask]
    logh_valid = logh[valid_mask]
    
    # Use improved initial guess algorithm (windowed OLS approach)
    C_init, k_0_init, delta_k_init, tau_init = make_slope8_initial_guess(s_valid, logh_valid)
    
    # Bounds for the quantile fit
    # Note: delta_k constraint is conditional: delta_k >= 0 only when k_0 < 0 (depletion case)
    # When k_0 >= 0, delta_k can be negative (no depletion constraint)
    MIN_DELTA_K = -0.1  # Allow negative delta_k when k_0 >= 0
    MAX_DELTA_K = 0.1   # long-run minus initial slope
    MIN_TAU = 1e-3      # Minimum value for tau bound (weeks)
    MAX_TAU = 260.0     # e.g., 5 years; avoids 600-year degeneracy
    
    # Store initial estimates for return (before clipping)
    initial_params = (float(C_init), float(k_0_init), float(delta_k_init), float(tau_init))
    
    # Parameter vector: [C, k_0, Δk, tau]
    p0 = np.array([C_init, k_0_init, delta_k_init, tau_init], dtype=float)
    
    # Bounds for the quantile fit
    # C ~ log(h). Hazards are ~1e-5–1e-3, so C is safely around [-20, -5].
    C_MIN, C_MAX = -25.0, 0.0
    
    # Weekly log-slope is small (~±0.01). Keep a generous but finite box.
    K_MIN, K_MAX = -0.1, 0.1
    
    bounds = [
        (C_MIN, C_MAX),                 # C
        (K_MIN, K_MAX),                 # k_0
        (MIN_DELTA_K, MAX_DELTA_K),     # Δk (can be negative when k_0 >= 0)
        (MIN_TAU, MAX_TAU),             # tau
    ]
    
    # Make sure starting point is inside the box
    p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
    
    tau_q = SLOPE8_QUANTILE_TAU  # quantile for the check loss
    
    def quantile_loss(p):
        """
        Quantile (check) loss on log h(s):
        
        ρ_τ(u) = τ*u      if u >= 0
               = (τ-1)*u  if u < 0
        
        Here u = logh_valid - predicted.
        
        Conditional constraint: When k_0 < 0 (depletion case), enforce delta_k >= 0 (k_inf >= k_0).
        When k_0 >= 0, delta_k can be negative (no depletion constraint).
        """
        C, k_0, delta_k, tau = p
        
        # Just in case the optimizer wanders slightly outside bounds numerically:
        if not np.isfinite(tau) or tau <= MIN_TAU:
            return 1e9
        
        # Conditional constraint: if k_0 < 0, delta_k must be >= 0 (depletion constraint)
        # If k_0 >= 0, delta_k can be negative (no constraint)
        # Use smooth penalty that increases gradually as constraint is violated
        # Allow small tolerance around delta_k = 0 to handle numerical precision and k_0 = k_inf case
        DELTA_K_TOLERANCE = 1e-6  # Small tolerance for numerical precision
        penalty = 0.0
        if k_0 < 0 and delta_k < -DELTA_K_TOLERANCE:
            # Violation of depletion constraint: add smooth quadratic penalty
            # Penalty increases quadratically with violation magnitude
            # Only penalize if violation is significant (beyond tolerance)
            violation = abs(delta_k + DELTA_K_TOLERANCE)  # Measure violation beyond tolerance
            penalty = 1e4 * violation * violation  # Quadratic penalty, smoother than hard cutoff
        
        k_inf = k_0 + delta_k
        
        # When delta_k is very close to 0, the depletion term becomes negligible
        # Handle this case to avoid numerical issues with tau when delta_k ≈ 0
        if abs(delta_k) < DELTA_K_TOLERANCE:
            # Linear model: log h(s) = C + k_inf * s (depletion term is negligible)
            predicted = C + k_inf * s_valid
            # When delta_k ≈ 0, tau is unidentifiable, so add small regularization
            # to prevent tau from wandering to extreme values
            tau_regularization = 1e-6 * (tau - 52.0)**2  # Encourage tau near typical value
        else:
            # Full depletion model
            predicted = C + k_inf * s_valid - delta_k * tau * (
                1.0 - np.exp(-s_valid / (tau + EPS))
            )
            tau_regularization = 0.0  # No regularization needed when tau is identifiable
        
        resid = logh_valid - predicted  # u in the formula above
        pos = resid >= 0.0
        loss = np.where(pos, tau_q * resid, (tau_q - 1.0) * resid)
        
        # Optional very small ridge to help numerics:
        ridge = 1e-4 * np.sum(resid**2)
        
        # Add penalty for constraint violation (if any) and tau regularization
        total_loss = float(np.sum(loss) + ridge + penalty + tau_regularization)
        
        return total_loss
    
    try:
        # Adjust convergence tolerances based on number of data points
        # With very few points, use looser tolerances to avoid excessive iterations
        n_data = len(s_valid)
        if n_data < 15:
            # Very few data points: use looser tolerances
            ftol = 1e-4
            gtol = 1e-3
        elif n_data < 30:
            # Few data points: moderate tolerances
            ftol = 1e-5
            gtol = 1e-4
        else:
            # Normal case: standard tolerances
            ftol = 1e-6
            gtol = 1e-5
        
        result = minimize(
            quantile_loss,
            p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 5000,  # Increased from 2000 to handle slow convergence cases
                "ftol": ftol,     # Function tolerance (adjusted by data size)
                "gtol": gtol,     # Gradient tolerance (adjusted by data size)
                "maxfun": 15000,  # Maximum function evaluations
            }
        )
        
        # Extract fitted parameters (always extract, even if invalid)
        C, k_0, delta_k, tau = result.x
        
        # Conditional constraint: only enforce delta_k >= 0 when k_0 < 0 (depletion case)
        # When k_0 >= 0, allow delta_k to be negative
        if k_0 < 0:
            # Depletion case: enforce k_inf >= k_0 (delta_k >= 0)
            delta_k = max(delta_k, 0.0)
        # else: k_0 >= 0, delta_k can be negative (no constraint)
        
        tau = min(max(tau, MIN_TAU), MAX_TAU)
        k_inf = k_0 + delta_k
        
        # Check validity of each parameter
        C_valid = np.isfinite(C)
        ka_valid = np.isfinite(k_0)
        kb_valid = np.isfinite(k_inf)
        tau_valid = np.isfinite(tau) and tau > EPS
        
        # Extract message with proper handling of bytes/string and all possible message sources
        def extract_message(msg):
            """Extract full message string, handling bytes and ensuring full content."""
            if msg is None:
                return None
            if isinstance(msg, bytes):
                try:
                    decoded = msg.decode('utf-8', errors='replace')
                    # Check if decoded message is just "ABNORMAL: " - might be incomplete
                    if decoded.strip() == "ABNORMAL:" or decoded.strip().startswith("ABNORMAL:"):
                        # Try to get more info from the raw bytes
                        try:
                            # Sometimes the message might have null bytes or other encoding issues
                            full_bytes = bytes(msg)
                            # Look for printable characters after "ABNORMAL:"
                            if b'ABNORMAL:' in full_bytes:
                                idx = full_bytes.index(b'ABNORMAL:') + len(b'ABNORMAL:')
                                remainder = full_bytes[idx:]
                                if remainder:
                                    remainder_str = remainder.decode('utf-8', errors='replace').strip()
                                    if remainder_str:
                                        decoded = decoded + " " + remainder_str
                        except Exception:
                            pass
                    return decoded
                except Exception:
                    return str(msg)
            msg_str = str(msg)
            # Check if message appears truncated
            if msg_str.strip() == "ABNORMAL:" or (msg_str.strip().startswith("ABNORMAL:") and len(msg_str.strip()) <= 10):
                # Message might be incomplete - try to get more from result object
                pass
            return msg_str
        
        # Get full message string - check multiple possible sources
        full_message = None
        if hasattr(result, 'message'):
            full_message = extract_message(result.message)
        
        # Also check for additional message information in other attributes
        # L-BFGS-B sometimes stores additional info in different places
        additional_info = []
        if hasattr(result, 'task'):
            task_msg = str(result.task) if result.task else None
            if task_msg and task_msg != full_message:
                additional_info.append(f"task={task_msg}")
        
        # Check if message is incomplete and try to reconstruct
        if full_message and (full_message.strip() == "ABNORMAL:" or full_message.strip().startswith("ABNORMAL:")):
            # Message appears incomplete - try to get more context
            if additional_info:
                full_message = full_message + " " + " ".join(additional_info)
        
        # Status code interpretation for L-BFGS-B
        status_code = None
        status_meaning = None
        if hasattr(result, 'status'):
            status_code = int(result.status)
            # L-BFGS-B status codes:
            status_meanings = {
                0: "success",
                1: "too_many_function_evaluations",
                2: "too_many_iterations", 
                3: "lower_bound_greater_than_upper_bound",
                4: "line_search_failed",
                5: "abnormal_termination",
                6: "rounding_errors_prevent_progress",
                7: "user_callback_requested_stop",
            }
            status_meaning = status_meanings.get(status_code, f"unknown_status_{status_code}")
        
        # Build diagnostics dict with comprehensive optimizer information
        diagnostics = {
            "success": bool(result.success),
            "fun": float(result.fun) if np.isfinite(result.fun) else None,
            "nfev": int(result.nfev) if hasattr(result, 'nfev') else None,
            "nit": int(result.nit) if hasattr(result, 'nit') else None,
            "message": full_message,
            "C_valid": C_valid,
            "ka_valid": ka_valid,
            "kb_valid": kb_valid,
            "tau_valid": tau_valid,
        }
        
        # Add additional optimizer details if available
        if status_code is not None:
            diagnostics["optimizer_status"] = status_code
        if status_meaning is not None:
            diagnostics["optimizer_status_meaning"] = status_meaning
        if hasattr(result, 'warnflag'):
            diagnostics["optimizer_warnflag"] = int(result.warnflag)
        if hasattr(result, 'allvecs'):
            diagnostics["optimizer_allvecs_count"] = len(result.allvecs) if result.allvecs else 0
        if hasattr(result, 'grad'):
            grad_norm = np.linalg.norm(result.grad) if result.grad is not None and np.isfinite(result.grad).any() else None
            diagnostics["optimizer_grad_norm"] = float(grad_norm) if grad_norm is not None else None
        
        # Build detailed message explaining what went wrong if optimizer didn't succeed
        if not result.success:
            detail_parts = []
            detail_parts.append(f"optimizer_success=False")
            if status_code is not None:
                detail_parts.append(f"status={status_code}({status_meaning})")
            if full_message:
                # Include full message, escaping quotes for CSV
                msg_escaped = full_message.replace('"', '""').replace('\n', ' ').replace('\r', ' ')
                detail_parts.append(f"message=\"{msg_escaped}\"")
            if not C_valid:
                detail_parts.append(f"C={C:.6e}(invalid)")
            if not ka_valid:
                detail_parts.append(f"ka={k_0:.6e}(invalid)")
            if not kb_valid:
                detail_parts.append(f"kb={k_inf:.6e}(invalid)")
            if not tau_valid:
                detail_parts.append(f"tau={tau:.6e}(invalid)")
            # Add parameter values for debugging abnormal terminations
            if status_code == 5:  # abnormal termination
                detail_parts.append(f"final_params:C={C:.6e},ka={k_0:.6e},kb={k_inf:.6e},tau={tau:.6e}")
                detail_parts.append(f"final_loss={result.fun:.6e}" if np.isfinite(result.fun) else "final_loss=invalid")
                if hasattr(result, 'nfev'):
                    detail_parts.append(f"nfev={result.nfev}")
                if hasattr(result, 'nit'):
                    detail_parts.append(f"nit={result.nit}")
                
                # Check for common patterns that cause abnormal termination
                pattern_parts = []
                # Check if tau is at or near lower bound (MIN_TAU = 1e-3 weeks)
                tau_at_lower_bound = abs(tau - MIN_TAU) < MIN_TAU * 10 or tau < MIN_TAU * 2
                if tau_at_lower_bound:
                    pattern_parts.append(f"tau_at_lower_bound(tau={tau:.6e},MIN_TAU={MIN_TAU})")
                
                # Check if tau is at or near upper bound (MAX_TAU = 260 weeks)
                tau_at_upper_bound = abs(tau - MAX_TAU) < 1.0 or tau > MAX_TAU * 0.99
                if tau_at_upper_bound:
                    pattern_parts.append(f"tau_at_upper_bound(tau={tau:.6e},MAX_TAU={MAX_TAU})")
                
                # Check if delta_k is very small (model collapsing to linear)
                delta_k_small = abs(delta_k) < 1e-5
                if delta_k_small:
                    pattern_parts.append(f"delta_k_very_small(delta_k={delta_k:.6e},ka≈kb)")
                
                # Check if delta_k is at upper bound
                delta_k_at_upper = abs(delta_k - MAX_DELTA_K) < 1e-5
                if delta_k_at_upper:
                    pattern_parts.append(f"delta_k_at_upper_bound(delta_k={delta_k:.6e},MAX_DELTA_K={MAX_DELTA_K})")
                
                # Check if we have very few data points
                if len(s_valid) < 10:
                    pattern_parts.append(f"few_data_points(n={len(s_valid)})")
                
                # Check if gradient norm is very large (numerical issues)
                if hasattr(result, 'grad') and result.grad is not None:
                    grad_norm = np.linalg.norm(result.grad)
                    if np.isfinite(grad_norm) and grad_norm > 1e6:
                        pattern_parts.append(f"large_gradient_norm({grad_norm:.2e})")
                
                if pattern_parts:
                    detail_parts.append("patterns:" + ",".join(pattern_parts))
            
            diagnostics["failure_detail"] = "; ".join(detail_parts)
        else:
            diagnostics["failure_detail"] = None
        
        # Return fitted parameters (even if invalid) along with diagnostics
        if result.success and C_valid and ka_valid and kb_valid and tau_valid:
            return (float(C), float(k_inf), float(k_0), float(tau)), initial_params, diagnostics
        else:
            # Return invalid parameters (as actual values, not NaN) so they can be logged
            return (float(C), float(k_inf), float(k_0), float(tau)), initial_params, diagnostics
    
    except Exception as e:
        import warnings
        warnings.warn(f"fit_slope8_depletion (quantile) failed: {str(e)}", RuntimeWarning)
        diagnostics = {
            "success": False,
            "fun": None,
            "nfev": None,
            "nit": None,
            "message": str(e),
            "optimizer_status": None,
            "optimizer_warnflag": None,
            "optimizer_grad_norm": None,
            "failure_detail": f"exception: {str(e)}",
            "C_valid": False,
            "ka_valid": False,
            "kb_valid": False,
            "tau_valid": False,
        }
        return (np.nan, np.nan, np.nan, np.nan), initial_params, diagnostics

def compute_slope6_normalization(df, baseline_window, enrollment_date_str, dual_print_fn=None, force_linear_mode=False):
    """
    Compute Slope8 normalization parameters for each cohort independently.
    
    For each cohort c (including dose 0 and YearOfBirth=-2 for all-ages):
    - Uses Slope8 quantile regression depletion-mode normalization as primary method
    - Fit window: enrollment_date to SLOPE_FIT_END_ISO
      - For highest dose: fit uses data from s >= SLOPE_FIT_DELAY_WEEKS (default 15 weeks)
      - For other doses: fit uses all data from enrollment
    - Application: normalization applied from s=0 (enrollment) for all cohorts
    - Fit depletion curve: log h(s) = C + k_∞*s + (k_0 - k_∞)*τ*(1 - e^(-s/τ))
    - Normalization: h_norm = h * exp(-kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))  [C term excluded so adjustment is 1 at s=0]
    - Falls back to linear mode if Slope8 fit fails or insufficient data
    - Tracks abnormal fits via optimizer diagnostics (status=5 or not success)
    - Returns dict with mode, parameters, and abnormal_fit flag
    
    This normalizes each cohort independently using Slope8 quantile regression,
    which provides robust handling of depletion-driven curvature.
    
    Args:
        df: DataFrame for one enrollment sheet with columns YearOfBirth, Dose, DateDied, MR, etc.
        baseline_window: Tuple (start_iso_week, end_iso_week) - kept for compatibility, not used for Slope8
        enrollment_date_str: Enrollment date string (e.g., "2021_24") for fit window start
        dual_print_fn: Optional logging function
        force_linear_mode: If True, skip slope8 attempts and use linear fits only (for MC mode)
        
    Returns:
        Dict mapping (YearOfBirth, Dose) -> params dict with keys:
        {
            "mode": "slope8", "linear", or "none",
            "C": intercept (slope8 mode),
            "ka": k_0 starting slope (slope8 mode),
            "kb": k_∞ final slope (slope8 mode),
            "tau": depletion timescale (slope8 mode),
            "b_original": original b_lin value from linear fit (slope8 mode),
            "abnormal_fit": bool - True if optimizer reported abnormal termination or failure,
            "a": intercept (linear fallback mode),
            "b": linear slope (linear fallback mode),
            "t_mean": time centering constant (linear fallback mode) or 0.0 (slope8 mode),
            "c": quadratic coefficient (always 0.0),
            "tau": quantile level (linear fallback mode)
        }
        For cohorts with insufficient data: dict with mode="linear", abnormal_fit=False, all params 0.0
    """
    def _print(msg):
        if dual_print_fn:
            dual_print_fn(msg)
    
    normalization_params = {}
    
    # Compute hazards
    df = df.copy()
    df["hazard"] = hazard_from_mr_improved(df["MR"].clip(lower=0.0, upper=0.999))
    
    # Create all-ages cohort (YearOfBirth=-2) by aggregating across all YearOfBirth values
    # This is done before processing so it's treated like any other cohort
    df_sorted = df.sort_values("DateDied")
    all_ages_agg = df_sorted.groupby(["Dose", "DateDied"]).agg({
        "ISOweekDied": "first",
        "Alive": "sum",
        "Dead": "sum",
        "PT": "sum"
    }).reset_index()
    all_ages_agg["MR"] = np.where(all_ages_agg["PT"] > 0,
                                  all_ages_agg["Dead"] / (all_ages_agg["PT"] + EPS),
                                  np.nan)
    all_ages_agg["hazard"] = hazard_from_mr_improved(all_ages_agg["MR"].clip(lower=0.0, upper=0.999))
    all_ages_agg = all_ages_agg.sort_values(["Dose", "DateDied"])
    all_ages_agg["t"] = all_ages_agg.groupby("Dose").cumcount().astype(float)
    all_ages_agg["YearOfBirth"] = -2  # Mark as all-ages cohort
    
    # Append all-ages cohort to main dataframe (iso_label will be added for all rows below)
    df = pd.concat([df, all_ages_agg[["YearOfBirth", "Dose", "DateDied", "MR", "hazard", "t", "ISOweekDied", "Alive", "Dead", "PT"]]], ignore_index=True)
    
    # Annotate ISO week labels
    iso_parts = df["DateDied"].dt.isocalendar()
    df["iso_label"] = iso_parts.year.astype(str) + "-" + iso_parts.week.astype(str).str.zfill(2)
    
    # Get fit window weeks (for regression)
    fit_weeks = _iso_week_list_slope6(*baseline_window)
    
    # Determine application window: enrollment_date to 2024-16
    # Parse enrollment date to get start of application window
    enrollment_dt = _parse_enrollment_date(enrollment_date_str)
    application_end_dt = _iso_to_date_slope6(SLOPE6_APPLICATION_END_ISO)
    
    # Determine highest dose for this enrollment date (for slope8 delayed fit)
    sheet_name = df['sheet_name'].iloc[0] if 'sheet_name' in df.columns and len(df) > 0 else enrollment_date_str
    dose_pairs = get_dose_pairs(sheet_name)
    max_dose = max(max(pair) for pair in dose_pairs) if dose_pairs else 0
    
    # Determine all expected cohorts: all YearOfBirth values × all doses that appear in dose_pairs
    # This ensures we log all cohorts, even if they have no data
    all_doses = set()
    for pair in dose_pairs:
        all_doses.add(pair[0])
        all_doses.add(pair[1])
    all_yobs = set(df["YearOfBirth"].unique()) if len(df) > 0 else set()
    # Also include -2 (all-ages) if it's not already there
    if len(df) > 0:
        all_yobs.add(-2)
    
    # Track which cohorts we've processed
    processed_cohorts = set()
    
    # Process each cohort independently
    for (yob, dose), g in df.groupby(["YearOfBirth", "Dose"], sort=False):
        processed_cohorts.add((yob, dose))
        # Sort by DateDied to ensure t values are in order
        g_sorted = g.sort_values("DateDied").reset_index(drop=True)
        
        # Extract cohort hazards and time indices by ISO week
        # Group by ISO week and get both hazard and t (time since enrollment)
        cohort_data = g_sorted.groupby("iso_label").agg({
            "hazard": "mean",
            "t": "first",  # Use first t value for each ISO week
            "DateDied": "first"  # Need date for application window filtering
        }).to_dict(orient="index")
        
        # Build application window time values (for determining t_mean)
        application_t_values = []
        for week, week_data in cohort_data.items():
            date_died = week_data.get("DateDied")
            if date_died is not None and enrollment_dt <= date_died <= application_end_dt:
                t_actual = week_data.get("t")
                if t_actual is not None and pd.notna(t_actual):
                    application_t_values.append(float(t_actual))
        
        # Compute t_mean from application window
        if len(application_t_values) == 0:
            # Fallback: use all available t values
            application_t_values = [float(week_data.get("t")) for week_data in cohort_data.values() 
                                   if week_data.get("t") is not None and pd.notna(week_data.get("t"))]
        
        if len(application_t_values) == 0:
            sheet_name = df['sheet_name'].iloc[0] if 'sheet_name' in df.columns and len(df) > 0 else 'unknown'
            _print(f"SLOPE6_FIT,EnrollmentDate={sheet_name},YoB={int(yob)},Dose={int(dose)},status=no_application_window_data")
            normalization_params[(yob, dose)] = {
                "mode": "none",
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
                "t_mean": 0.0,
                "tau": None  # tau only valid for slope8 mode
            }
            continue
        
        t_mean = np.mean(application_t_values)
        
        # Determine fit window: enrollment_date + skip weeks to SLOPE_FIT_END_ISO
        # Skip weeks: DYNAMIC_HVE_SKIP_WEEKS (2 weeks) for all doses, plus SLOPE_FIT_DELAY_WEEKS (15 weeks) for highest dose
        slope8_end_dt = _iso_to_date_slope6(SLOPE_FIT_END_ISO)
        is_highest_dose = (dose == max_dose)
        fit_start_weeks = DYNAMIC_HVE_SKIP_WEEKS + (SLOPE_FIT_DELAY_WEEKS if is_highest_dose else 0)
        
        # Build log h_c(t) for t in fit_window, using actual t values (time since enrollment)
        # Fit window: enrollment_date + fit_start_weeks to SLOPE_FIT_END_ISO
        log_h_values = []
        h_values = []  # Store raw hazard values for debugging
        t_values = []
        iso_weeks_used = []  # Track ISO weeks actually used in fit
        
        for week, week_data in cohort_data.items():
            date_died = week_data.get("DateDied")
            if date_died is not None and enrollment_dt <= date_died <= slope8_end_dt:
                hc = week_data.get("hazard")
                t_actual = week_data.get("t")
                
                # Calculate weeks since enrollment
                weeks_since_enrollment = (date_died - enrollment_dt).days / 7.0
                
                # Only include data from fit_start_weeks onwards
                if weeks_since_enrollment >= fit_start_weeks:
                    # Must be valid and positive
                    if hc is not None and t_actual is not None and hc > EPS and pd.notna(t_actual):
                        try:
                            log_h_val = np.log(hc)
                            if np.isfinite(log_h_val) and np.isfinite(t_actual):
                                log_h_values.append(log_h_val)
                                h_values.append(float(hc))  # Store raw hazard for debugging
                                t_values.append(float(t_actual))  # Actual time since enrollment
                                iso_weeks_used.append(week)  # Track which ISO week was used
                        except Exception:
                            continue
        
        # For linear fit, compute t_mean from fit window (not application window)
        # This ensures time-centering is based on the actual fit window
        if len(t_values) > 0:
            t_mean_fit_window = np.mean(t_values)
        else:
            t_mean_fit_window = t_mean  # Fallback to application window mean
        
        # Initialize slope8_attempted before any try blocks that might fail
        slope8_attempted = False
        
        # Fit linear median regression first
        sheet_name = df['sheet_name'].iloc[0] if 'sheet_name' in df.columns and len(df) > 0 else 'unknown'
        if len(log_h_values) >= SLOPE6_MIN_DATA_POINTS:
            try:
                # Use time-centered approach: t_c = t - t_mean_fit_window
                t_c_fit = np.array(t_values) - t_mean_fit_window
                a_lin, b_lin, _ = fit_linear_median(t_c_fit, np.array(log_h_values), tau=SLOPE6_QUANTILE_TAU)
                
                # Use t_mean from fit window for normalization
                t_mean = t_mean_fit_window
                predicted_lin = a_lin + b_lin * t_c_fit
                residuals_lin = np.array(log_h_values) - predicted_lin
                rms_error_lin = np.sqrt(np.mean(residuals_lin**2))
                # Mean Absolute Deviation (MAD) - independent of quantile tau
                mad_error_lin = np.mean(np.abs(residuals_lin))
                # Median Absolute Error (MedAE) - robust to outliers
                medae_lin = np.median(np.abs(residuals_lin))
                # Median Absolute Deviation of target variable (MAD(y))
                logh_median_lin = np.median(log_h_values)
                mad_y_lin = np.median(np.abs(log_h_values - logh_median_lin))
                # Relative Median Absolute Error (rMedAE) = MedAE / MAD(y)
                rmedae_lin = medae_lin / mad_y_lin if mad_y_lin > EPS else None
                # Classify fit quality based on rMedAE
                if rmedae_lin is not None and np.isfinite(rmedae_lin):
                    if rmedae_lin <= 0.1:
                        fit_quality_lin = "excellent"
                    elif rmedae_lin <= 0.3:
                        fit_quality_lin = "very_good"
                    elif rmedae_lin <= 0.5:
                        fit_quality_lin = "good"
                    else:
                        fit_quality_lin = "poor"
                else:
                    fit_quality_lin = None

                # Always log linear fit (first of three methods)
                try:
                    log_slope7_fit_debug({
                        "enrollment_date": sheet_name,
                        "mode": "linear",
                        "YearOfBirth": int(yob),
                        "Dose": int(dose),
                        "n_points": len(log_h_values),
                        "b_original": float(b_lin),
                        "C": float(a_lin),
                        "ka": None,
                        "kb": None,
                        "tau": None,
                        "rms_error": float(rms_error_lin),
                        "mad_error": float(mad_error_lin) if np.isfinite(mad_error_lin) else None,
                        "medae": float(medae_lin) if np.isfinite(medae_lin) else None,
                        "rmedae": float(rmedae_lin) if rmedae_lin is not None and np.isfinite(rmedae_lin) else None,
                        "fit_quality": fit_quality_lin,
                        "note": f"linear_fit_t_mean={t_mean:.6f},skip_weeks={fit_start_weeks}",
                        "error": None,
                        "s_values": t_values,
                        "logh_values": log_h_values,
                        "h_values": h_values,  # Include raw hazard values for debugging
                        "iso_weeks_used": iso_weeks_used,  # Track ISO weeks actually used
                    })
                except Exception:
                    pass
            except Exception as e:
                # Log linear fit failure
                try:
                    log_slope7_fit_debug({
                        "enrollment_date": sheet_name,
                        "mode": "linear",
                        "YearOfBirth": int(yob),
                        "Dose": int(dose),
                        "n_points": len(log_h_values),
                        "error": str(e),
                        "note": "linear_fit_failed",
                        "s_values": t_values,
                        "logh_values": log_h_values,
                        "h_values": h_values,
                        "iso_weeks_used": iso_weeks_used,
                    })
                except Exception:
                    pass
                a_lin = None
                b_lin = None
                rms_error_lin = None
                mad_error_lin = None
                medae_lin = None
                rmedae_lin = None
                fit_quality_lin = None
        else:
            # Insufficient data for linear fit - log this case
            _print(f"SLOPE6_FIT,EnrollmentDate={sheet_name},YoB={int(yob)},Dose={int(dose)},status=insufficient_data,points={len(log_h_values)}")
            try:
                log_slope7_fit_debug({
                    "enrollment_date": sheet_name,
                    "mode": "linear",
                    "YearOfBirth": int(yob),
                    "Dose": int(dose),
                    "n_points": len(log_h_values),
                    "error": None,
                    "note": f"insufficient_data_for_fit (need {SLOPE6_MIN_DATA_POINTS}, have {len(log_h_values)})",
                    "s_values": t_values if len(t_values) > 0 else None,
                    "logh_values": log_h_values if len(log_h_values) > 0 else None,
                    "h_values": h_values if len(h_values) > 0 else None,
                    "iso_weeks_used": iso_weeks_used if len(iso_weeks_used) > 0 else None,
                })
            except Exception:
                pass
            normalization_params[(yob, dose)] = {
                "mode": "none",
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
                "t_mean": t_mean,
                "tau": None  # tau only valid for slope8 mode
            }
            # Also log that slope8 wasn't attempted
            try:
                log_slope7_fit_debug({
                    "enrollment_date": sheet_name,
                    "mode": "slope8",
                    "YearOfBirth": int(yob),
                    "Dose": int(dose),
                    "n_points": len(log_h_values),
                    "error": None,
                    "note": f"slope8_not_attempted_insufficient_data (need {SLOPE6_MIN_DATA_POINTS}, have {len(log_h_values)})",
                    "optimizer_success": False,
                    "param_C_valid": False,
                    "param_ka_valid": False,
                    "param_kb_valid": False,
                    "param_tau_valid": False,
                })
            except Exception:
                pass
            continue
        
        # Need at least SLOPE6_MIN_DATA_POINTS points for quantile regression fit
        # (This check is now redundant since we already checked above, but keeping for clarity)
        if len(log_h_values) < SLOPE6_MIN_DATA_POINTS:
            # This should not happen since we already handled this case above, but just in case:
            continue
        
        # --- Slope8 quantile regression fit ---
        # Only attempt slope8 for cohorts born before SLOPE8_MAX_YOB (1940)
        # Cohorts >= 1940 and all-ages cohort (YOB=-2) will use linear fit (already computed above)
        # In MC mode (force_linear_mode=True), always use linear fits
        use_slope8 = (yob < SLOPE8_MAX_YOB and yob != -2) and not force_linear_mode
        
        if use_slope8:
            # Build slope8 deployment window: enrollment_date to SLOPE_FIT_END_ISO
            # Skip weeks: DYNAMIC_HVE_SKIP_WEEKS (2 weeks) for all doses, plus SLOPE_FIT_DELAY_WEEKS (15 weeks) for highest dose
            # Fit region = adjustment region (same window for both fitting and applying normalization)
            # Note: slope8_end_dt, is_highest_dose, and fit_start_weeks are already defined above
            
            # Build full deployment window data (s=0 at enrollment_date for all doses)
            # This will be used for logging predictions
            s_values_slope8_full = []
            log_h_slope8_values_full = []
            h_slope8_values_full = []
            iso_weeks_slope8_full = []
            
            # Build fit window data (with skip weeks applied)
            s_values_slope8_fit = []
            log_h_slope8_values_fit = []
            
            # Build s_values: s=0 corresponds to enrollment_date
            for week, week_data in cohort_data.items():
                    date_died = week_data.get("DateDied")
                    if date_died is not None and enrollment_dt <= date_died <= slope8_end_dt:
                        hc = week_data.get("hazard")
                        if hc is not None and hc > EPS:
                            try:
                                log_h_val = np.log(hc)
                                if np.isfinite(log_h_val):
                                    # Calculate s as weeks since enrollment_date
                                    weeks_since_enrollment = (date_died - enrollment_dt).days / 7.0
                                    
                                    # Always add to full dataset (for logging)
                                    s_values_slope8_full.append(weeks_since_enrollment)
                                    log_h_slope8_values_full.append(log_h_val)
                                    h_slope8_values_full.append(float(hc))
                                    iso_weeks_slope8_full.append(week)
                                    
                                    # Only add to fit dataset if weeks_since_enrollment >= fit_start_weeks
                                    # This skips DYNAMIC_HVE_SKIP_WEEKS for all doses, plus SLOPE_FIT_DELAY_WEEKS for highest dose
                                    if weeks_since_enrollment >= fit_start_weeks:
                                        s_values_slope8_fit.append(weeks_since_enrollment)
                                        log_h_slope8_values_fit.append(log_h_val)
                            except Exception:
                                continue
        else:
            # For cohorts >= 1940, skip slope8 and use linear fit
            s_values_slope8_fit = []
            log_h_slope8_values_fit = []
            s_values_slope8_full = []
            log_h_slope8_values_full = []
            h_slope8_values_full = []
            iso_weeks_slope8_full = []
        
        # Fit slope8 using only the fit window data (only for cohorts < 1940)
        # Store results for later use in normalization
        # ALWAYS use slope8 parameters if fit was attempted, even if unreliable
        # Note: slope8_attempted was initialized earlier, but reset here for clarity
        slope8_attempted = False
        abnormal_fit_flag = False
        C_slope8_norm = None
        kb_slope8_norm = None
        ka_slope8_norm = None
        tau_slope8_norm = None
        rms_error_slope8_norm = None
        mad_error_slope8_norm = None
        medae_slope8_norm = None
        rmedae_slope8_norm = None
        fit_quality_slope8_norm = None
        mode_str = None
        note_str = None
        
        if use_slope8 and len(s_values_slope8_fit) >= SLOPE6_MIN_DATA_POINTS:
            try:
                (C_slope8, kb_slope8, ka_slope8, tau_slope8), (C_init_slope8, ka_init_slope8, delta_k_init_slope8, tau_init_slope8), diagnostics_slope8 = fit_slope8_depletion(
                    np.array(s_values_slope8_fit), np.array(log_h_slope8_values_fit)
                )
                slope8_attempted = True
                
                # Always try to compute RMS error and MAD, even if parameters are invalid
                # This helps diagnose how bad the fit actually is
                rms_error_slope8 = None
                mad_error_slope8 = None
                try:
                    # Try to compute prediction even if some parameters are invalid
                    # Use np.nan_to_num to handle invalid values gracefully
                    C_safe = np.nan_to_num(C_slope8, nan=0.0, posinf=0.0, neginf=0.0)
                    kb_safe = np.nan_to_num(kb_slope8, nan=0.0, posinf=0.0, neginf=0.0)
                    ka_safe = np.nan_to_num(ka_slope8, nan=0.0, posinf=0.0, neginf=0.0)
                    tau_safe = np.nan_to_num(tau_slope8, nan=1.0, posinf=1.0, neginf=1.0)
                    tau_safe = max(tau_safe, EPS)  # Ensure tau > 0 for exp calculation
                    
                    predicted_slope8_fit = C_safe + kb_safe * np.array(s_values_slope8_fit) + (ka_safe - kb_safe) * tau_safe * (1.0 - np.exp(-np.array(s_values_slope8_fit) / (tau_safe + EPS)))
                    residuals_slope8_fit = np.array(log_h_slope8_values_fit) - predicted_slope8_fit
                    rms_error_slope8 = np.sqrt(np.mean(residuals_slope8_fit**2))
                    # Mean Absolute Deviation (MAD) - independent of quantile tau
                    mad_error_slope8 = np.mean(np.abs(residuals_slope8_fit))
                    # Median Absolute Error (MedAE) - robust to outliers
                    medae_slope8 = np.median(np.abs(residuals_slope8_fit))
                    # Median Absolute Deviation of target variable (MAD(y))
                    logh_median = np.median(log_h_slope8_values_fit)
                    mad_y_slope8 = np.median(np.abs(log_h_slope8_values_fit - logh_median))
                    # Relative Median Absolute Error (rMedAE) = MedAE / MAD(y)
                    rmedae_slope8 = medae_slope8 / mad_y_slope8 if mad_y_slope8 > EPS else None
                    # Classify fit quality based on rMedAE
                    if rmedae_slope8 is not None and np.isfinite(rmedae_slope8):
                        if rmedae_slope8 <= 0.1:
                            fit_quality_slope8 = "excellent"
                        elif rmedae_slope8 <= 0.3:
                            fit_quality_slope8 = "very_good"
                        elif rmedae_slope8 <= 0.5:
                            fit_quality_slope8 = "good"
                        else:
                            fit_quality_slope8 = "poor"
                    else:
                        fit_quality_slope8 = None
                    
                    # If RMS error is not finite, set to None
                    if not np.isfinite(rms_error_slope8):
                        rms_error_slope8 = None
                    # If MAD error is not finite, set to None
                    if not np.isfinite(mad_error_slope8):
                        mad_error_slope8 = None
                    # If MedAE is not finite, set to None
                    if not np.isfinite(medae_slope8):
                        medae_slope8 = None
                    # If rMedAE is not finite, set to None
                    if rmedae_slope8 is not None and not np.isfinite(rmedae_slope8):
                        rmedae_slope8 = None
                except Exception:
                    # If prediction fails completely, leave RMS error as None
                    pass
                
                # ALWAYS use slope8 parameters, even if invalid or abnormal
                # Replace NaN/inf values with safe defaults for normalization
                C_slope8_norm = C_slope8 if np.isfinite(C_slope8) else 0.0
                kb_slope8_norm = kb_slope8 if np.isfinite(kb_slope8) else 0.0
                ka_slope8_norm = ka_slope8 if np.isfinite(ka_slope8) else 0.0
                tau_slope8_norm = tau_slope8 if (np.isfinite(tau_slope8) and tau_slope8 > EPS) else 1.0
                rms_error_slope8_norm = rms_error_slope8
                mad_error_slope8_norm = mad_error_slope8
                medae_slope8_norm = medae_slope8
                rmedae_slope8_norm = rmedae_slope8
                fit_quality_slope8_norm = fit_quality_slope8
                
                # Determine if fit was abnormal/unreliable:
                # - Optimizer not successful
                # - Abnormal termination (status=5)
                # - Invalid parameters
                abnormal_fit_flag = (
                    not diagnostics_slope8.get("success", False) or
                    diagnostics_slope8.get("optimizer_status") == 5 or
                    not (diagnostics_slope8["C_valid"] and diagnostics_slope8["ka_valid"] and 
                         diagnostics_slope8["kb_valid"] and diagnostics_slope8["tau_valid"])
                )
                
                # Build note describing issues if any
                if abnormal_fit_flag:
                    note_parts = []
                    if not diagnostics_slope8.get("success", False):
                        note_parts.append("optimizer_not_successful")
                    if diagnostics_slope8.get("optimizer_status") == 5:
                        note_parts.append("abnormal_termination")
                    failed_params = []
                    if not diagnostics_slope8["C_valid"]:
                        failed_params.append("C")
                    if not diagnostics_slope8["ka_valid"]:
                        failed_params.append("ka")
                    if not diagnostics_slope8["kb_valid"]:
                        failed_params.append("kb")
                    if not diagnostics_slope8["tau_valid"]:
                        failed_params.append("tau")
                    if failed_params:
                        note_parts.append(f"invalid_params:{','.join(failed_params)}")
                    note_str = "; ".join(note_parts) if note_parts else "unreliable_fit"
                
                mode_str = "slope8"
                
                # Log with full deployment window data (s=0 onwards for all doses)
                # Always log fitted parameters (even if invalid) and diagnostics
                log_slope7_fit_debug({
                    "enrollment_date": sheet_name,
                    "mode": mode_str,
                    "YearOfBirth": int(yob),
                    "Dose": int(dose),
                    "n_points": len(s_values_slope8_fit),  # Number of points used in fit
                    "s_values": list(map(float, s_values_slope8_full)),  # Full range for logging
                    "logh_values": list(map(float, log_h_slope8_values_full)),  # Full range for logging
                    "h_values": h_slope8_values_full,  # Full range for logging
                    "iso_weeks_used": iso_weeks_slope8_full,  # Full range for logging
                    "C": float(C_slope8) if np.isfinite(C_slope8) else None,
                    "ka": float(ka_slope8) if np.isfinite(ka_slope8) else None,
                    "kb": float(kb_slope8) if np.isfinite(kb_slope8) else None,
                    "tau": float(tau_slope8) if np.isfinite(tau_slope8) else None,
                    **format_initial_params(C_init_slope8, ka_init_slope8, delta_k_init_slope8, tau_init_slope8),
                    "b_original": float(b_lin),
                    "rms_error": float(rms_error_slope8) if rms_error_slope8 is not None else None,
                    "mad_error": float(mad_error_slope8) if mad_error_slope8 is not None else None,
                    "medae": float(medae_slope8) if medae_slope8 is not None else None,
                    "rmedae": float(rmedae_slope8) if rmedae_slope8 is not None else None,
                    "fit_quality": fit_quality_slope8,
                    "note": note_str,
                    "optimizer_success": diagnostics_slope8["success"],
                    "optimizer_fun": diagnostics_slope8["fun"],
                    "optimizer_nfev": diagnostics_slope8["nfev"],
                    "optimizer_nit": diagnostics_slope8["nit"],
                    "optimizer_message": diagnostics_slope8["message"],
                    "optimizer_status": diagnostics_slope8.get("optimizer_status"),
                    "optimizer_status_meaning": diagnostics_slope8.get("optimizer_status_meaning"),
                    "optimizer_warnflag": diagnostics_slope8.get("optimizer_warnflag"),
                    "optimizer_grad_norm": diagnostics_slope8.get("optimizer_grad_norm"),
                    "failure_detail": diagnostics_slope8.get("failure_detail"),
                    "param_C_valid": diagnostics_slope8["C_valid"],
                    "param_ka_valid": diagnostics_slope8["ka_valid"],
                    "param_kb_valid": diagnostics_slope8["kb_valid"],
                    "param_tau_valid": diagnostics_slope8["tau_valid"],
                })
            except Exception as e:
                # Even on exception, use slope8 with default parameters (mark as abnormal)
                slope8_attempted = True
                # Use safe default parameters (no normalization effect, but still slope8 mode)
                C_slope8_norm = 0.0
                kb_slope8_norm = 0.0
                ka_slope8_norm = 0.0
                tau_slope8_norm = 1.0
                rms_error_slope8_norm = None
                mad_error_slope8_norm = None
                medae_slope8_norm = None
                rmedae_slope8_norm = None
                fit_quality_slope8_norm = None
                abnormal_fit_flag = True
                mode_str = "slope8_exception"
                note_str = f"exception during slope8 fit: {str(e)}"
                
                # Log exception case
                log_slope7_fit_debug({
                    "enrollment_date": sheet_name,
                    "mode": "slope8_exception",
                    "YearOfBirth": int(yob),
                    "Dose": int(dose),
                    "n_points": len(s_values_slope8_fit),
                    "s_values": list(map(float, s_values_slope8_full)),
                    "logh_values": list(map(float, log_h_slope8_values_full)),
                    "h_values": h_slope8_values_full,
                    "iso_weeks_used": iso_weeks_slope8_full,
                    "error": str(e),
                    "b_original": float(b_lin),
                    "note": note_str,
                    "optimizer_success": False,
                    "optimizer_fun": None,
                    "optimizer_nfev": None,
                    "optimizer_nit": None,
                    "optimizer_message": str(e),
                    "param_C_valid": False,
                    "param_ka_valid": False,
                    "param_kb_valid": False,
                    "param_tau_valid": False,
                })
        
        # Determine which mode to use based on YOB threshold
        if use_slope8 and slope8_attempted and C_slope8_norm is not None:
            # Use slope8 for cohorts < 1940
            params = {
                "mode": "slope8",
                "C": C_slope8_norm,
                "ka": ka_slope8_norm,  # k_0 starting slope
                "kb": kb_slope8_norm,  # k_∞ final slope
                "tau": tau_slope8_norm,
                "b_original": b_lin,
                "abnormal_fit": abnormal_fit_flag,
                "t_mean": 0.0,  # No centering for slope8
                "a": 0.0,  # Not used in slope8
                "b": 0.0,  # Not used in slope8 (use b_original instead)
                "c": 0.0,  # Not used in slope8
            }
            normalization_params[(yob, dose)] = params
            abnormal_str = " (abnormal/unreliable)" if abnormal_fit_flag else ""
            note_str_display = f", note={note_str}" if note_str else ""
            rms_error_str = f"{rms_error_slope8_norm:.6e}" if rms_error_slope8_norm is not None else "nan"
            _print(f"SLOPE8_FIT,EnrollmentDate={sheet_name},YoB={int(yob)},Dose={int(dose)},mode=slope8,C={C_slope8_norm:.6e},ka={ka_slope8_norm:.6e},kb={kb_slope8_norm:.6e},tau={tau_slope8_norm:.6e},b_original={b_lin:.6e},rms_error={rms_error_str},points={len(s_values_slope8_fit)}{abnormal_str}{note_str_display}")
        elif not use_slope8:
            # Use linear mode for cohorts >= 1940 (YOB threshold) or all-ages cohort (YOB=-2)
            if a_lin is not None and b_lin is not None:
                params = {
                    "mode": "linear",
                    "a": a_lin,
                    "b": b_lin,
                    "c": 0.0,
                    "t_mean": t_mean,
                    "tau": None,  # tau only valid for slope8 mode
                    "abnormal_fit": False,  # Normal case for cohorts >= 1940
                }
                normalization_params[(yob, dose)] = params
                # Suppress SLOPE8_FIT messages in MC mode (force_linear_mode=True)
                if not force_linear_mode:
                    rms_error_str = f"{rms_error_lin:.6e}" if rms_error_lin is not None else "nan"
                    _print(f"SLOPE8_FIT,EnrollmentDate={sheet_name},YoB={int(yob)},Dose={int(dose)},mode=linear(yob_threshold),a={a_lin:.6e},b={b_lin:.6e},c=0.000000e+00,t_mean={t_mean:.6e},rms_error={rms_error_str},points={len(log_h_values)},skip_weeks={fit_start_weeks}")
            else:
                # Linear fit failed - mark as abnormal
                params = {
                    "mode": "linear",
                    "a": 0.0,
                    "b": 0.0,
                    "c": 0.0,
                    "t_mean": t_mean,
                    "tau": None,
                    "abnormal_fit": True,
                }
                normalization_params[(yob, dose)] = params
                # Suppress SLOPE8_FIT messages in MC mode (force_linear_mode=True)
                if not force_linear_mode:
                    _print(f"SLOPE8_FIT,EnrollmentDate={sheet_name},YoB={int(yob)},Dose={int(dose)},mode=linear(yob_threshold_failed),a=0.0,b=0.0,c=0.0,t_mean={t_mean:.6e},points={len(log_h_values)} (abnormal - linear fit failed)")
        else:
            # Fallback: slope8 was attempted but failed (insufficient data or exception)
            # In this case, use linear mode but mark as abnormal since slope8 should have been used
            if a_lin is not None and b_lin is not None:
                params = {
                    "mode": "linear",
                    "a": a_lin,
                    "b": b_lin,
                    "c": 0.0,
                    "t_mean": t_mean,
                    "tau": None,  # tau only valid for slope8 mode
                    "abnormal_fit": True,  # Mark as abnormal since slope8 wasn't attempted
                }
                normalization_params[(yob, dose)] = params
                # Suppress SLOPE8_FIT messages in MC mode (force_linear_mode=True)
                if not force_linear_mode:
                    rms_error_str = f"{rms_error_lin:.6e}" if rms_error_lin is not None else "nan"
                    _print(f"SLOPE8_FIT,EnrollmentDate={sheet_name},YoB={int(yob)},Dose={int(dose)},mode=linear(fallback_slope8_not_attempted),a={a_lin:.6e},b={b_lin:.6e},c=0.000000e+00,t_mean={t_mean:.6e},rms_error={rms_error_str},points={len(log_h_values)} (abnormal - slope8 should have been used)")
            else:
                # Both failed - use defaults
                params = {
                    "mode": "none",
                    "a": 0.0,
                    "b": 0.0,
                    "c": 0.0,
                    "t_mean": t_mean,
                    "tau": None,
                    "abnormal_fit": True,
                }
                normalization_params[(yob, dose)] = params
                _print(f"SLOPE8_FIT,EnrollmentDate={sheet_name},YoB={int(yob)},Dose={int(dose)},mode=none,points={len(log_h_values)} (abnormal - both slope8 and linear failed)")
    
    # ------------------------------------------------------------------
    # All-ages (YoB = -2) cohort is now processed as a regular cohort in compute_slope6_normalization
    # No special handling needed here - it's included in the main processing loop
    
    # After processing all cohorts that exist in dataframe, log missing cohorts
    # This ensures all expected cohorts are logged, even if they have no data
    for yob in all_yobs:
        for dose in all_doses:
            if (yob, dose) not in processed_cohorts:
                # Log missing cohort
                try:
                    log_slope7_fit_debug({
                        "enrollment_date": sheet_name,
                        "mode": "linear",
                        "YearOfBirth": int(yob),
                        "Dose": int(dose),
                        "n_points": 0,
                        "error": None,
                        "note": "cohort_not_present_in_data",
                        "s_values": None,
                        "logh_values": None,
                        "h_values": None,
                        "iso_weeks_used": None,
                    })
                    log_slope7_fit_debug({
                        "enrollment_date": sheet_name,
                        "mode": "slope8",
                        "YearOfBirth": int(yob),
                        "Dose": int(dose),
                        "n_points": 0,
                        "error": None,
                        "note": "cohort_not_present_in_data",
                        "optimizer_success": False,
                        "param_C_valid": False,
                        "param_ka_valid": False,
                        "param_kb_valid": False,
                        "param_tau_valid": False,
                    })
                except Exception:
                    pass
                normalization_params[(yob, dose)] = {
                    "mode": "none",
                    "a": 0.0,
                    "b": 0.0,
                    "c": 0.0,
                    "t_mean": 0.0,
                    "tau": None
                }
    
    return normalization_params

def _parse_enrollment_date(enrollment_date_str):
    """Parse enrollment date string (e.g., '2021_24' or '2021-24') to datetime."""
    # Handle both underscore and hyphen formats
    if '_' in enrollment_date_str:
        year_str, week_str = enrollment_date_str.split('_')
    elif '-' in enrollment_date_str and len(enrollment_date_str.split('-')) == 2:
        parts = enrollment_date_str.split('-')
        if len(parts[1]) <= 2:  # Week number
            year_str, week_str = parts
        else:
            # Might be ISO format, try parsing directly
            try:
                return _iso_to_date_slope6(enrollment_date_str)
            except:
                raise ValueError(f"Cannot parse enrollment date: {enrollment_date_str}")
    else:
        raise ValueError(f"Cannot parse enrollment date: {enrollment_date_str}")
    
    year = int(year_str)
    week = int(week_str)
    
    # Get first Monday of the year
    jan1 = datetime(year, 1, 1)
    # ISO week starts on Monday, so find first Monday
    days_until_monday = (7 - jan1.weekday()) % 7
    if days_until_monday == 0 and jan1.weekday() != 0:
        days_until_monday = 7
    first_monday = jan1 + timedelta(days=days_until_monday)
    
    # Add weeks
    enrollment_date = first_monday + timedelta(weeks=week-1)
    return enrollment_date

def build_kcor_rows(df, sheet_name, dual_print=None, slope6_params_map=None):
    """
    Build per-age KCOR rows for all PAIRS and ASMR pooled rows (YearOfBirth=0).
    Assumptions:
      - Person-time PT = Alive
      - MR = Dead / PT
      - MR_adj slope-removed via QR (for smoothing, not CH calculation)
      - CH = cumsum(-ln(1 - MR_adj)) where MR_adj = MR × exp(-slope × (t - t0))
      - KCOR = (cum_hazard_num / cum_hazard_den), anchored to 1 at week KCOR_NORMALIZATION_WEEKS if available
      - 95% CI uses proper uncertainty propagation: Var[KCOR] = KCOR² * [Var[cumD_num]/cumD_num² + Var[cumD_den]/cumD_den² + Var[baseline_num]/baseline_num² + Var[baseline_den]/baseline_den²]
      - ASMR pooling uses fixed baseline weights = sum of PT in the first 4 weeks per age (time-invariant).
    """
    out_rows = []
    # Fast access by (age,dose)
    by_age_dose = {(y,d): g.sort_values("DateDied")
                   for (y,d), g in df.groupby(["YearOfBirth","Dose"], sort=False)}
    
    # Scale factors logic removed (FINAL_KCOR_* deprecated)

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
            gv_unique = gv[["DateDied","ISOweekDied","MR","MR_adj","CH","CH_actual","cumD_adj","cumD_unadj","hazard_raw","hazard_adj","slope","scale_factor","MR_smooth","t","Alive","Dead"]].drop_duplicates(subset=["DateDied"], keep="first")
            gu_unique = gu[["DateDied","ISOweekDied","MR","MR_adj","CH","CH_actual","cumD_adj","cumD_unadj","hazard_raw","hazard_adj","slope","scale_factor","MR_smooth","t","Alive","Dead"]].drop_duplicates(subset=["DateDied"], keep="first")
            
            merged = pd.merge(
                gv_unique,
                gu_unique,
                on="DateDied", suffixes=("_num","_den"), how="inner"
            ).sort_values("DateDied")
            if merged.empty:
                continue
            # Ensure standalone frame (not a slice) to avoid chained-assignment warnings downstream
            merged = merged.reset_index(drop=True).copy(deep=True)
                
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
            
            # Get baseline K_raw value at effective normalization week (KCOR_NORMALIZATION_WEEKS weeks after accumulation starts)
            t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
            baseline_k_raw = merged["K_raw"].iloc[t0_idx]
            if not (np.isfinite(baseline_k_raw) and baseline_k_raw > EPS):
                baseline_k_raw = 1.0
            
            # Compute final KCOR values normalized to baseline (no extra scaling)
            merged["KCOR"] = np.where(np.isfinite(merged["K_raw"]), 
                                      merged["K_raw"] / baseline_k_raw,
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

            # KCOR 95% CI using post-anchor increments (Nelson–Aalen), adjusted for slope-normalization
            # Anchor index
            t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
            # Post-anchor cumulative hazard increments
            dCH_num = merged["CH_num"] - float(merged["CH_num"].iloc[t0_idx])
            dCH_den = merged["CH_den"] - float(merged["CH_den"].iloc[t0_idx])
            # Slope-normalization scale per week: s = hazard_adj / hazard_raw
            s_num = (merged.get("hazard_adj_num", np.nan) / (merged.get("hazard_raw_num", np.nan) + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            s_den = (merged.get("hazard_adj_den", np.nan) / (merged.get("hazard_raw_den", np.nan) + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Nelson–Aalen incremental variances, scaled by s^2
            var_inc_num = (merged.get("Dead_num", 0.0).astype(float) / (merged.get("Alive_num", 0.0).astype(float) + EPS)**2) * (s_num.astype(float)**2)
            var_inc_den = (merged.get("Dead_den", 0.0).astype(float) / (merged.get("Alive_den", 0.0).astype(float) + EPS)**2) * (s_den.astype(float)**2)
            # Cumulative from anchor forward (exclude anchor point)
            var_cum_num = var_inc_num.cumsum() - float(var_inc_num.iloc[t0_idx])
            var_cum_den = var_inc_den.cumsum() - float(var_inc_den.iloc[t0_idx])
            var_cum_num = np.clip(var_cum_num.replace([np.inf, -np.inf], np.nan).fillna(0.0), 0.0, np.inf)
            var_cum_den = np.clip(var_cum_den.replace([np.inf, -np.inf], np.nan).fillna(0.0), 0.0, np.inf)
            # SE on log scale
            denom_num = np.maximum(np.abs(dCH_num.values), EPS)
            denom_den = np.maximum(np.abs(dCH_den.values), EPS)
            se_log_sq = (var_cum_num.values / (denom_num**2)) + (var_cum_den.values / (denom_den**2))
            merged["SE_logKCOR"] = np.sqrt(np.clip(se_log_sq, 0.0, np.inf))
            
            # Calculate 95% CI bounds on log scale, then exponentiate, with clipping applied at creation
            # CI = exp(log(KCOR) ± 1.96 * SE_logKCOR)
            _ci_lower_raw = merged["KCOR"] * safe_exp(-1.96 * merged["SE_logKCOR"])
            _ci_upper_raw = merged["KCOR"] * safe_exp(1.96 * merged["SE_logKCOR"])
            merged["CI_lower"] = np.clip(_ci_lower_raw, 0, merged["KCOR"] * 10)
            merged["CI_upper"] = np.clip(_ci_upper_raw, merged["KCOR"] * 0.1, merged["KCOR"] * 10)
            # Blank CI for all weeks up to and including baseline (t <= t0)
            merged.loc[merged.index <= t0_idx, ["CI_lower", "CI_upper"]] = np.nan

            # Build explicit hazard columns: unadjusted from hazard_raw_*, adjusted from hazard_adj_*
            merged["hazard_num"] = merged.get("hazard_raw_num", np.nan)
            merged["hazard_den"] = merged.get("hazard_raw_den", np.nan)
            merged["hazard_adj_num"] = merged.get("hazard_adj_num", np.nan)
            merged["hazard_adj_den"] = merged.get("hazard_adj_den", np.nan)
            
            # Check if either numerator or denominator cohort had abnormal fit
            abnormal_fit_num = False
            abnormal_fit_den = False
            if slope6_params_map is not None:
                params_num = slope6_params_map.get((sheet_name, int(yob), int(num)), {})
                params_den = slope6_params_map.get((sheet_name, int(yob), int(den)), {})
                if isinstance(params_num, dict):
                    abnormal_fit_num = params_num.get("abnormal_fit", False)
                if isinstance(params_den, dict):
                    abnormal_fit_den = params_den.get("abnormal_fit", False)
            merged["abnormal_fit"] = abnormal_fit_num or abnormal_fit_den
            
            # Order: unadjusted first, then adjusted
            out = merged[["DateDied","ISOweekDied_num","KCOR","CI_lower","CI_upper",
                          "CH_num","hazard_num","hazard_adj_num","t_num",
                          "CH_den","hazard_den","hazard_adj_den","t_den","abnormal_fit"]].copy()

            # MR fields are written raw; no display scaling
            # Remove redundant merged suffix columns like *_den copies of ISOweek
            if "ISOweekDied_den" in out.columns and "ISOweekDied" not in out.columns:
                out.rename(columns={"ISOweekDied_num":"ISOweekDied"}, inplace=True)
                out.drop(columns=[c for c in out.columns if c.endswith("_den") and c.startswith("ISOweekDied")], inplace=True)
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
    
    # Store weights globally for use in KCOR_ns calculation and display
    global ASMR_WEIGHTS_BY_SHEET
    ASMR_WEIGHTS_BY_SHEET[sheet_name] = weights.copy()
    
    # Debug: Show expected-deaths weights (suppress in MC mode)
    if DEBUG_VERBOSE and not MONTE_CARLO_MODE:
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
            t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(gvn) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE and len(gdn) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
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
                
                # Pooled CI via weighted aggregation of per-age log-variance using post-anchor ΔCH (Nelson–Aalen)
                var_terms = []
                for yob, g_age in df_sorted.groupby("YearOfBirth", sort=False):
                    if yob not in anchors:
                        continue
                    gvn = g_age[g_age["Dose"] == num].sort_values("DateDied")
                    gdn = g_age[g_age["Dose"] == den].sort_values("DateDied")
                    if gvn.empty or gdn.empty:
                        continue
                    gvn_upto = gvn[gvn["DateDied"] <= dt]
                    gdn_upto = gdn[gdn["DateDied"] <= dt]
                    if len(gvn_upto) == 0 or len(gdn_upto) == 0:
                        continue
                    t0_idx_age = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(gvn_upto) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE and len(gdn_upto) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
                    # ΔCH at dt
                    dCH_num_age = float(gvn_upto["CH"].iloc[-1]) - float(gvn_upto["CH"].iloc[t0_idx_age])
                    dCH_den_age = float(gdn_upto["CH"].iloc[-1]) - float(gdn_upto["CH"].iloc[t0_idx_age])
                    # Scale factors s = hazard_adj / hazard_raw
                    s_num_age = (gvn_upto["hazard_adj"] / (gvn_upto["hazard_raw"] + EPS)).to_numpy()
                    s_den_age = (gdn_upto["hazard_adj"] / (gdn_upto["hazard_raw"] + EPS)).to_numpy()
                    # NA variance increments
                    var_inc_num_age = (gvn_upto["Dead"].astype(float) / (gvn_upto["Alive"].astype(float) + EPS)**2).to_numpy() * (s_num_age**2)
                    var_inc_den_age = (gdn_upto["Dead"].astype(float) / (gdn_upto["Alive"].astype(float) + EPS)**2).to_numpy() * (s_den_age**2)
                    # Cumulative from anchor forward
                    csum_num = np.cumsum(var_inc_num_age)
                    csum_den = np.cumsum(var_inc_den_age)
                    var_cum_num_age = float(csum_num[-1] - csum_num[t0_idx_age]) if csum_num.size > 0 else 0.0
                    var_cum_den_age = float(csum_den[-1] - csum_den[t0_idx_age]) if csum_den.size > 0 else 0.0
                    # Per-age log-variance contribution
                    denom_num_age = max(abs(dCH_num_age), EPS)
                    denom_den_age = max(abs(dCH_den_age), EPS)
                    var_log_age = (var_cum_num_age / (denom_num_age**2)) + (var_cum_den_age / (denom_den_age**2))
                    w = weights.get(yob, 0.0)
                    var_terms.append((w**2) * var_log_age)
                # Final pooled SE on log scale
                total_uncertainty = sum(var_terms)
                SE_total = safe_sqrt(total_uncertainty) / sum(wts)
                
                # Clip SE to prevent overflow (using reasonable bound)
                SE_total = min(SE_total, 10.0)
                
                # Calculate 95% CI bounds on log scale, then exponentiate
                CI_lower = Kpool * safe_exp(-1.96 * SE_total)
                CI_upper = Kpool * safe_exp(1.96 * SE_total)

                # Clip CI bounds to reasonable values
                CI_lower = max(0, min(CI_lower, Kpool * 10))
                CI_upper = max(Kpool * 0.1, min(CI_upper, Kpool * 10))

                # After clipping: blank CI at baseline and earlier dates (t <= t0)
                if dt <= all_dates[min(KCOR_NORMALIZATION_WEEKS_EFFECTIVE, len(all_dates)-1)]:
                    CI_lower = np.nan
                    CI_upper = np.nan
                

                
                # Check if any contributing age group had abnormal fit
                abnormal_fit_pooled = False
                if slope6_params_map is not None:
                    for yob_check in df_sorted["YearOfBirth"].unique():
                        if yob_check not in anchors:
                            continue
                        params_num_check = slope6_params_map.get((sheet_name, int(yob_check), int(num)), {})
                        params_den_check = slope6_params_map.get((sheet_name, int(yob_check), int(den)), {})
                        if isinstance(params_num_check, dict) and params_num_check.get("abnormal_fit", False):
                            abnormal_fit_pooled = True
                            break
                        if isinstance(params_den_check, dict) and params_den_check.get("abnormal_fit", False):
                            abnormal_fit_pooled = True
                            break
                
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
                    # Keep schema consistent with dose_pairs rows; pooled ASMR has no per-age hazards/CH
                    "CH_num": np.nan,
                    "hazard_adj_num": np.nan,
                    "hazard_num": np.nan,
                    "t_num": np.nan,
                    "CH_den": np.nan,
                    "hazard_adj_den": np.nan,
                    "hazard_den": np.nan,
                    "t_den": np.nan,
                    "abnormal_fit": abnormal_fit_pooled
                })

    # -------- All Ages rows (YearOfBirth = -2) --------
    # Aggregate all ages together as a single cohort (no age grouping)
    # This is different from ASMR pooling which weights across age groups
    all_ages_rows = []
    df_sorted = df.sort_values("DateDied")
    
    # Aggregate across all YearOfBirth values for each Dose and DateDied
    # First, aggregate basic counts
    all_ages_agg = df_sorted.groupby(["Dose", "DateDied"]).agg({
        "ISOweekDied": "first",
        "Alive": "sum",
        "Dead": "sum",
        "PT": "sum"
    }).reset_index()
    
    # Recompute MR from aggregated counts
    all_ages_agg["MR"] = np.where(all_ages_agg["PT"] > 0, all_ages_agg["Dead"] / (all_ages_agg["PT"] + EPS), np.nan)
    
    # Sort and compute time index
    all_ages_agg = all_ages_agg.sort_values(["Dose", "DateDied"])
    all_ages_agg["t"] = all_ages_agg.groupby("Dose").cumcount().astype(float)
    
    # Compute hazard from aggregated MR (hazard-level normalization, not MR-level)
    all_ages_agg["MR_smooth"] = all_ages_agg["MR"]  # Use raw MR for smoothing
    all_ages_agg["hazard_raw"] = hazard_from_mr_improved(all_ages_agg["MR"])
    
    # Use slope8 normalization parameters from slope6_params_map (already computed in compute_slope6_normalization)
    # Do NOT overwrite the parameters - they were correctly computed with slope8 for cohort -2
    # Apply normalization using the same logic as regular cohorts
    
    def apply_slope8_norm_all_ages(row):
        """Apply slope8 normalization to all-ages cohort using parameters from slope6_params_map."""
        if slope6_params_map is None:
            return row["hazard_raw"]
        
        # Get normalization parameters for this dose (cohort -2 was already computed in compute_slope6_normalization)
        params = slope6_params_map.get((sheet_name, -2, int(row["Dose"])), None)
        if params is None:
            # Fallback: try without int conversion
            params = slope6_params_map.get((sheet_name, -2, row["Dose"]), None)
        
        if params is None or not isinstance(params, dict) or params.get("mode") == "none":
            return row["hazard_raw"]
        
        mode = params.get("mode", "linear")
        
        if mode == "slope8":
            # Slope8 mode: h_norm = h * exp(-kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))
            # C term excluded so adjustment is 1 at s=0
            ka = params.get("ka", 0.0)
            kb = params.get("kb", 0.0)
            tau = params.get("tau", 1.0)
            # Use s = t (time since enrollment, NOT centered)
            s = row["t"]
            # Ensure tau is positive and finite
            if not np.isfinite(tau) or tau <= EPS:
                tau = 1.0
            norm_factor = np.exp(-kb * s - (ka - kb) * tau * (1.0 - np.exp(-s / (tau + EPS))))
            if not np.isfinite(norm_factor):
                norm_factor = 1.0
            return row["hazard_raw"] * norm_factor
        elif mode == "linear":
            # Linear mode: h_norm = h * exp(-b * t_c) where t_c = t - t_mean
            b = params.get("b", 0.0)
            t_mean = params.get("t_mean", 0.0)
            t_c = row["t"] - t_mean
            return row["hazard_raw"] * np.exp(-b * t_c)
        else:
            # Unknown mode (shouldn't happen with slope8, but be safe)
            return row["hazard_raw"]
    
    all_ages_agg["hazard_adj"] = all_ages_agg.apply(apply_slope8_norm_all_ages, axis=1)
    all_ages_agg["MR_adj"] = all_ages_agg["MR"]  # Keep MR_adj = MR (normalization is at hazard level)
    all_ages_agg["hazard_eff"] = np.where(all_ages_agg["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), all_ages_agg["hazard_adj"], 0.0)
    all_ages_agg["CH"] = all_ages_agg.groupby("Dose")["hazard_eff"].cumsum()
    all_ages_agg["CH_actual"] = all_ages_agg.groupby("Dose")["MR_adj"].cumsum()
    all_ages_agg["cumPT"] = all_ages_agg.groupby("Dose")["PT"].cumsum()
    all_ages_agg["cumD_adj"] = all_ages_agg["CH"] * all_ages_agg["cumPT"]
    all_ages_agg["cumD_unadj"] = all_ages_agg.groupby("Dose")["Dead"].cumsum()
    
    # Add slope and scale_factor for consistency (using parameters from slope6_params_map)
    def get_slope_for_all_ages(row):
        """Get slope value from slope6_params_map for display purposes."""
        if slope6_params_map is None:
            return 0.0
        params = slope6_params_map.get((sheet_name, -2, int(row["Dose"])), None)
        if params is None:
            params = slope6_params_map.get((sheet_name, -2, row["Dose"]), None)
        if params is None or not isinstance(params, dict):
            return 0.0
        mode = params.get("mode", "none")
        if mode == "slope8":
            # For slope8, use b_original (the original linear slope) for display
            return params.get("b_original", 0.0)
        elif mode == "linear":
            return params.get("b", 0.0)
        else:
            return 0.0
    
    def get_scale_factor_for_all_ages(row):
        """Compute scale_factor from slope8 parameters for display purposes."""
        if slope6_params_map is None:
            return 1.0
        params = slope6_params_map.get((sheet_name, -2, int(row["Dose"])), None)
        if params is None:
            params = slope6_params_map.get((sheet_name, -2, row["Dose"]), None)
        if params is None or not isinstance(params, dict):
            return 1.0
        mode = params.get("mode", "none")
        s = row["t"]
        if mode == "slope8":
            # C term excluded so adjustment is 1 at s=0
            ka = params.get("ka", 0.0)
            kb = params.get("kb", 0.0)
            tau = params.get("tau", 1.0)
            if not np.isfinite(tau) or tau <= EPS:
                tau = 1.0
            norm_factor = np.exp(-kb * s - (ka - kb) * tau * (1.0 - np.exp(-s / (tau + EPS))))
            return norm_factor if np.isfinite(norm_factor) else 1.0
        elif mode == "linear":
            b = params.get("b", 0.0)
            t_mean = params.get("t_mean", 0.0)
            t_c = s - t_mean
            return np.exp(-b * t_c) if np.isfinite(b) else 1.0
        else:
            return 1.0
    
    all_ages_agg["slope"] = all_ages_agg.apply(get_slope_for_all_ages, axis=1)
    all_ages_agg["scale_factor"] = all_ages_agg.apply(get_scale_factor_for_all_ages, axis=1)
    
    # Now compute KCOR for all-ages aggregated data
    dose_pairs = get_dose_pairs(sheet_name)
    for num, den in dose_pairs:
        # Apply debug dose pair filter
        if DEBUG_DOSE_PAIR_ONLY and (num, den) != DEBUG_DOSE_PAIR_ONLY:
            continue
            
        gv_all = all_ages_agg[all_ages_agg["Dose"] == num].sort_values("DateDied")
        gu_all = all_ages_agg[all_ages_agg["Dose"] == den].sort_values("DateDied")
        
        if gv_all.empty or gu_all.empty:
            continue
        
        # Merge numerator and denominator
        merged_all = pd.merge(
            gv_all[["DateDied", "ISOweekDied", "MR", "MR_adj", "CH", "CH_actual", "cumD_adj", "cumD_unadj", 
                   "hazard_raw", "hazard_adj", "slope", "scale_factor", "MR_smooth", "t", "Alive", "Dead", "PT"]],
            gu_all[["DateDied", "ISOweekDied", "MR", "MR_adj", "CH", "CH_actual", "cumD_adj", "cumD_unadj",
                   "hazard_raw", "hazard_adj", "slope", "scale_factor", "MR_smooth", "t", "Alive", "Dead", "PT"]],
            on="DateDied", suffixes=("_num", "_den"), how="inner"
        ).sort_values("DateDied")
        
        if merged_all.empty:
            continue
        
        merged_all = merged_all.reset_index(drop=True).copy(deep=True)
        
        # Compute KCOR same way as per-age
        valid_denom = merged_all["CH_den"] > EPS
        merged_all["K_raw"] = np.where(valid_denom,
                                      merged_all["CH_num"] / merged_all["CH_den"], 
                                      np.nan)
        
        # Get baseline K_raw value at effective normalization week
        t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged_all) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
        baseline_k_raw = merged_all["K_raw"].iloc[t0_idx]
        if not (np.isfinite(baseline_k_raw) and baseline_k_raw > EPS):
            baseline_k_raw = 1.0
        
        # Compute final KCOR values normalized to baseline
        merged_all["KCOR"] = np.where(np.isfinite(merged_all["K_raw"]), 
                                      merged_all["K_raw"] / baseline_k_raw,
                                     np.nan)
        
        # KCOR 95% CI using post-anchor increments (Nelson–Aalen)
        t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged_all) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
        dCH_num = merged_all["CH_num"] - float(merged_all["CH_num"].iloc[t0_idx])
        dCH_den = merged_all["CH_den"] - float(merged_all["CH_den"].iloc[t0_idx])
        s_num = (merged_all.get("hazard_adj_num", np.nan) / (merged_all.get("hazard_raw_num", np.nan) + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        s_den = (merged_all.get("hazard_adj_den", np.nan) / (merged_all.get("hazard_raw_den", np.nan) + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        var_inc_num = (merged_all.get("Dead_num", 0.0).astype(float) / (merged_all.get("Alive_num", 0.0).astype(float) + EPS)**2) * (s_num.astype(float)**2)
        var_inc_den = (merged_all.get("Dead_den", 0.0).astype(float) / (merged_all.get("Alive_den", 0.0).astype(float) + EPS)**2) * (s_den.astype(float)**2)
        var_cum_num = var_inc_num.cumsum() - float(var_inc_num.iloc[t0_idx])
        var_cum_den = var_inc_den.cumsum() - float(var_inc_den.iloc[t0_idx])
        var_cum_num = np.clip(var_cum_num.replace([np.inf, -np.inf], np.nan).fillna(0.0), 0.0, np.inf)
        var_cum_den = np.clip(var_cum_den.replace([np.inf, -np.inf], np.nan).fillna(0.0), 0.0, np.inf)
        denom_num = np.maximum(np.abs(dCH_num.values), EPS)
        denom_den = np.maximum(np.abs(dCH_den.values), EPS)
        se_log_sq = (var_cum_num.values / (denom_num**2)) + (var_cum_den.values / (denom_den**2))
        merged_all["SE_logKCOR"] = np.sqrt(np.clip(se_log_sq, 0.0, np.inf))
        
        # Calculate 95% CI bounds on log scale, then exponentiate
        _ci_lower_raw = merged_all["KCOR"] * safe_exp(-1.96 * merged_all["SE_logKCOR"])
        _ci_upper_raw = merged_all["KCOR"] * safe_exp(1.96 * merged_all["SE_logKCOR"])
        merged_all["CI_lower"] = np.clip(_ci_lower_raw, 0, merged_all["KCOR"] * 10)
        merged_all["CI_upper"] = np.clip(_ci_upper_raw, merged_all["KCOR"] * 0.1, merged_all["KCOR"] * 10)
        merged_all.loc[merged_all.index <= t0_idx, ["CI_lower", "CI_upper"]] = np.nan
        
        # Check if either numerator or denominator cohort had abnormal fit (for all-ages)
        abnormal_fit_all_ages = False
        if slope6_params_map is not None:
            params_num_all = slope6_params_map.get((sheet_name, -2, int(num)), {})
            params_den_all = slope6_params_map.get((sheet_name, -2, int(den)), {})
            if isinstance(params_num_all, dict):
                abnormal_fit_all_ages = abnormal_fit_all_ages or params_num_all.get("abnormal_fit", False)
            if isinstance(params_den_all, dict):
                abnormal_fit_all_ages = abnormal_fit_all_ages or params_den_all.get("abnormal_fit", False)
        
        # Build output rows for all-ages
        for _, row in merged_all.iterrows():
            all_ages_rows.append({
                "EnrollmentDate": sheet_name,
                "ISOweekDied": row["ISOweekDied_num"],
                "Date": pd.to_datetime(row["DateDied"]).date(),
                "YearOfBirth": -2,  # -2 = all ages aggregated
                "Dose_num": num,
                "Dose_den": den,
                "KCOR": row["KCOR"],
                "CI_lower": row["CI_lower"],
                "CI_upper": row["CI_upper"],
                "CH_num": row["CH_num"],
                "hazard_adj_num": row["hazard_adj_num"],
                "hazard_num": row["hazard_raw_num"],
                "t_num": row["t_num"],
                "CH_den": row["CH_den"],
                "hazard_adj_den": row["hazard_adj_den"],
                "hazard_den": row["hazard_raw_den"],
                "t_den": row["t_den"],
                "abnormal_fit": abnormal_fit_all_ages
            })

    if out_rows or pooled_rows or all_ages_rows:
        return pd.concat(out_rows + [pd.DataFrame(pooled_rows)] + [pd.DataFrame(all_ages_rows)], ignore_index=True)
    return pd.DataFrame(columns=[
        "EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den",
        "KCOR","CI_lower","CI_upper","CH_num","hazard_adj_num","hazard_num","t_num",
        "CH_den","hazard_adj_den","hazard_den","t_den","abnormal_fit"
    ])

def build_kcor_o_rows(df, sheet_name):
    """Compute KCOR_o based on death-based slope adjustment and cumulative adjusted deaths.
    
    Steps:
      1) Compute death-based slopes via lookup windows.
      2) Adjust weekly deaths using multiplicative slope removal anchored at week KCOR_NORMALIZATION_WEEKS.
      3) Compute cumulative adjusted deaths per (YearOfBirth, Dose).
      4) For each dose pair and age, compute the ratio of cumulative deaths, then normalize by its
         value at week KCOR_NORMALIZATION_WEEKS to make it 1 at that anchor week.
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
            # Normalize at anchor week index (effective normalization week or first)
            t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
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

def build_kcor_ns_rows(df, sheet_name):
    """Compute KCOR_ns assuming zero slopes for all cohorts (no slope normalization).

    Method:
      - Use raw discrete hazard (hazard_raw) without slope adjustment.
      - Apply the same accumulation start rule (DYNAMIC_HVE_SKIP_WEEKS) as standard KCOR.
      - Compute cumulative raw hazard CH_ns per (YearOfBirth, Dose).
      - For each age and dose pair, compute K_raw_ns = CH_ns_num / CH_ns_den and
        normalize to 1 at week KCOR_NORMALIZATION_WEEKS -> KCOR_ns.
    """
    # Ensure needed columns and sorting; work on a copy to avoid side effects
    df_ns = df.sort_values(["YearOfBirth","Dose","DateDied"]).copy()
    # Accumulation start consistent with main KCOR path, but using hazard_raw
    df_ns["hazard_eff_ns"] = np.where(df_ns["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df_ns.get("hazard_raw", np.nan), 0.0)
    # Cumulative raw hazard per cohort/dose
    df_ns["CH_ns"] = df_ns.groupby(["YearOfBirth","Dose"]) ["hazard_eff_ns"].cumsum()

    out_rows = []
    by_age_dose = {(y,d): g.sort_values("DateDied") for (y,d), g in df_ns.groupby(["YearOfBirth","Dose"], sort=False)}
    dose_pairs = get_dose_pairs(sheet_name)
    for yob in df_ns["YearOfBirth"].unique():
        for num, den in dose_pairs:
            gv = by_age_dose.get((yob, num))
            gu = by_age_dose.get((yob, den))
            if gv is None or gu is None:
                continue
            gv_unique = gv[["DateDied","ISOweekDied","CH_ns"]].drop_duplicates(subset=["DateDied"], keep="first")
            gu_unique = gu[["DateDied","ISOweekDied","CH_ns"]].drop_duplicates(subset=["DateDied"], keep="first")
            merged = pd.merge(gv_unique, gu_unique, on="DateDied", suffixes=("_num","_den"), how="inner").sort_values("DateDied")
            if merged.empty:
                continue
            # Raw ratio of cumulative raw hazards
            valid = merged["CH_ns_den"] > EPS
            merged["K_raw_ns"] = np.where(valid, merged["CH_ns_num"] / merged["CH_ns_den"], np.nan)
            # Normalize at anchor index
            t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
            k0 = merged["K_raw_ns"].iloc[t0_idx]
            if not (np.isfinite(k0) and k0 > EPS):
                k0 = 1.0
            merged["KCOR_ns"] = np.where(np.isfinite(merged["K_raw_ns"]), merged["K_raw_ns"] / k0, np.nan)

            out = merged[["DateDied","ISOweekDied_num","KCOR_ns"]].copy()
            out["EnrollmentDate"] = sheet_name
            out["YearOfBirth"] = yob
            out["Dose_num"] = num
            out["Dose_den"] = den
            out.rename(columns={"ISOweekDied_num":"ISOweekDied","DateDied":"Date"}, inplace=True)
            out["Date"] = pd.to_datetime(out["Date"]).apply(lambda x: x.date())
            out_rows.append(out)

    # -------- All Ages rows (YearOfBirth = -2) for KCOR_ns --------
    # Aggregate all ages together as a single cohort (no age grouping) for non-slope-corrected KCOR
    df_ns_sorted = df_ns.sort_values("DateDied")
    
    # Aggregate across all YearOfBirth values for each Dose and DateDied
    # First aggregate basic counts
    all_ages_agg_ns = df_ns_sorted.groupby(["Dose", "DateDied"]).agg({
        "ISOweekDied": "first",
        "Alive": "sum",
        "Dead": "sum",
        "PT": "sum"
    }).reset_index()
    
    # Recompute MR from aggregated counts
    all_ages_agg_ns["MR"] = np.where(all_ages_agg_ns["PT"] > 0, all_ages_agg_ns["Dead"] / (all_ages_agg_ns["PT"] + EPS), np.nan)
    
    # Sort and compute time index
    all_ages_agg_ns = all_ages_agg_ns.sort_values(["Dose", "DateDied"])
    all_ages_agg_ns["t"] = all_ages_agg_ns.groupby("Dose").cumcount().astype(float)
    
    # Compute hazard_raw from aggregated MR (no slope adjustment for KCOR_ns)
    all_ages_agg_ns["hazard_raw"] = hazard_from_mr_improved(all_ages_agg_ns["MR"])
    
    # Apply accumulation start rule (DYNAMIC_HVE_SKIP_WEEKS)
    all_ages_agg_ns["hazard_eff_ns"] = np.where(all_ages_agg_ns["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), all_ages_agg_ns["hazard_raw"], 0.0)
    
    # Cumulative raw hazard per dose
    all_ages_agg_ns["CH_ns"] = all_ages_agg_ns.groupby("Dose")["hazard_eff_ns"].cumsum()
    
    # Compute KCOR_ns for all-ages aggregated data
    dose_pairs = get_dose_pairs(sheet_name)
    for num, den in dose_pairs:
        gv_all_ns = all_ages_agg_ns[all_ages_agg_ns["Dose"] == num].sort_values("DateDied")
        gu_all_ns = all_ages_agg_ns[all_ages_agg_ns["Dose"] == den].sort_values("DateDied")
        
        if gv_all_ns.empty or gu_all_ns.empty:
            continue
        
        # Merge numerator and denominator
        merged_all_ns = pd.merge(
            gv_all_ns[["DateDied", "ISOweekDied", "CH_ns"]],
            gu_all_ns[["DateDied", "ISOweekDied", "CH_ns"]],
            on="DateDied", suffixes=("_num", "_den"), how="inner"
        ).sort_values("DateDied")
        
        if merged_all_ns.empty:
            continue
        
        # Raw ratio of cumulative raw hazards
        valid = merged_all_ns["CH_ns_den"] > EPS
        merged_all_ns["K_raw_ns"] = np.where(valid, merged_all_ns["CH_ns_num"] / merged_all_ns["CH_ns_den"], np.nan)
        
        # Normalize at anchor index
        t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged_all_ns) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
        k0 = merged_all_ns["K_raw_ns"].iloc[t0_idx]
        if not (np.isfinite(k0) and k0 > EPS):
            k0 = 1.0
        merged_all_ns["KCOR_ns"] = np.where(np.isfinite(merged_all_ns["K_raw_ns"]), merged_all_ns["K_raw_ns"] / k0, np.nan)
        
        # Build output DataFrame for all-ages KCOR_ns (same format as regular age groups)
        out_all = merged_all_ns[["DateDied","ISOweekDied_num","KCOR_ns"]].copy()
        out_all["EnrollmentDate"] = sheet_name
        out_all["YearOfBirth"] = -2  # -2 = all ages aggregated
        out_all["Dose_num"] = num
        out_all["Dose_den"] = den
        out_all.rename(columns={"ISOweekDied_num":"ISOweekDied","DateDied":"Date"}, inplace=True)
        out_all["Date"] = pd.to_datetime(out_all["Date"]).apply(lambda x: x.date())
        out_rows.append(out_all)

    if out_rows:
        return pd.concat(out_rows, ignore_index=True)
    return pd.DataFrame(columns=["EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den","KCOR_ns"])

def build_kcor_o_deaths_details(df, sheet_name):
    """For each dose pair and age, output deaths/week, adjusted deaths/week, cumulative adjusted deaths,
    and normalized cumulative ratio (normalized to 1 at week KCOR_NORMALIZATION_WEEKS).

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
            t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
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
    global ENROLLMENT_DATES  # Declare global before any reference to ENROLLMENT_DATES
    global SLOPE_DEBUG_FILE
    output_dir = os.path.dirname(out_path)
    if not output_dir:
        output_dir = "."
    
    # Place slope debug CSV alongside the main KCOR output files
    SLOPE_DEBUG_FILE = os.path.join(output_dir, "KCOR_slope_debug.csv")
    # Clear the debug file at the start of each run
    if SLOPE_DEBUG_ENABLED:
        try:
            with open(SLOPE_DEBUG_FILE, "w", newline="", encoding="utf-8") as f:
                pass  # Create empty file or truncate existing file
        except Exception:
            pass  # Silently ignore errors in debug file setup
    dual_print, log_file_handle = setup_dual_output(output_dir, log_filename)
    
    # Print professional header
    # Note: datetime and timedelta are already imported at module level
    log_file_path = os.path.join(output_dir, log_filename)
    # Use forward slashes for display consistency
    log_file_display = log_file_path.replace('\\', '/')
    dual_print("="*80)
    dual_print(f"KCOR {VERSION} - Kirsch Cumulative Outcomes Ratio Analysis")
    dual_print("="*80)
    dual_print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mode_str = os.environ.get('KCOR_MODE', 'Primary Analysis')
    if MONTE_CARLO_MODE:
        dual_print(f"Mode: {mode_str} (Monte Carlo)")
    else:
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
    # Removed FINAL_KCOR parameters
    dual_print(f"  KCOR_NORMALIZATION_WEEKS = {KCOR_NORMALIZATION_WEEKS}")
    # Anchor-based slope parameters removed
    dual_print(f"  DYNAMIC_HVE_SKIP_WEEKS = {DYNAMIC_HVE_SKIP_WEEKS}")
    dual_print(f"  AGE_RANGE             = {AGE_RANGE}")
    # Legacy slope window removed
    # Moving-average parameters removed
    dual_print(f"  YEAR_RANGE            = {YEAR_RANGE}")
    dual_print(f"  ENROLLMENT_DATES      = {ENROLLMENT_DATES}")
    # MR is computed in annualized per 100k units; no additional display scaling applied
    dual_print(f"  DEBUG_VERBOSE         = {DEBUG_VERBOSE}")
    dual_print(f"  OVERRIDE_DOSE_PAIRS   = {OVERRIDE_DOSE_PAIRS}")
    dual_print(f"  OVERRIDE_YOBS         = {OVERRIDE_YOBS}")
    # Legacy quiet-anchor config removed
    dual_print(f"  KCOR_REPORTING_DATE   = {KCOR_REPORTING_DATE}")
    dual_print(f"  NEGATIVE_CONTROL_MODE = {NEGATIVE_CONTROL_MODE}")
    # Slope6 configuration
    dual_print(f"  SLOPE6_METHOD        = QuantReg (tau={SLOPE6_QUANTILE_TAU})  [Slope6: Time-centered linear quantile regression normalization for b >= 0]")
    dual_print(f"  SLOPE8_METHOD        = Quantile Regression (L-BFGS-B)  [Slope8: Depletion-mode normalization for all cohorts]")
    dual_print(f"  SLOPE6_QUANTILE_TAU  = {SLOPE6_QUANTILE_TAU}  [Quantile level for quantile regression (0.5 = median)]")
    dual_print(f"  SLOPE6_FIT_WINDOW    = 2022-01 to 2024-12  [Fixed window for regression fitting]")
    dual_print(f"  SLOPE6_APPLICATION_ENDPOINT = {SLOPE6_APPLICATION_END_ISO}  [Rightmost endpoint for determining centerpoint]")
    dual_print(f"  SLOPE6_BASELINE_WINDOW_LENGTH_MIN = {SLOPE6_BASELINE_WINDOW_LENGTH_MIN} weeks")
    dual_print(f"  SLOPE6_BASELINE_WINDOW_LENGTH_MAX = {SLOPE6_BASELINE_WINDOW_LENGTH_MAX} weeks")
    dual_print(f"  SLOPE6_BASELINE_START_YEAR = {SLOPE6_BASELINE_START_YEAR}")
    dual_print(f"  SLOPE6_MIN_DATA_POINTS = {SLOPE6_MIN_DATA_POINTS}")
    dual_print("="*80)
    dual_print("")
    
    xls = pd.ExcelFile(src_path)
    all_out = []
    pair_deaths_all = []
    by_dose_nc_all = []
    
    # Auto-derive enrollment dates from sheet names if not explicitly set
    if MONTE_CARLO_MODE:
        # In MC mode, all sheets are iterations (numbered "1", "2", "3", etc.)
        # Filter out any non-numeric sheets
        enrollment_sheets = [
            name for name in xls.sheet_names 
            if name.isdigit() or (name.replace('-', '').replace('_', '').isdigit())
        ]
        ENROLLMENT_DATES = sorted(enrollment_sheets, key=lambda x: int(x.replace('-', '').replace('_', '')))  # Sort numerically
        dual_print(f"[INFO] Monte Carlo mode: Processing {len(ENROLLMENT_DATES)} iterations")
    elif ENROLLMENT_DATES is None:
        # Filter out summary and MFG sheets (keep only main enrollment date sheets)
        enrollment_sheets = [
            name for name in xls.sheet_names 
            if not name.endswith('_summary') and '_MFG_' not in name
        ]
        ENROLLMENT_DATES = sorted(enrollment_sheets)  # Sort for consistent ordering
        dual_print(f"[INFO] Auto-derived enrollment dates from Excel file: {ENROLLMENT_DATES}")
    
    # Apply processing filters
    sheets_to_process = ENROLLMENT_DATES if ENROLLMENT_DATES else xls.sheet_names
    if YEAR_RANGE:
        start_year, end_year = YEAR_RANGE
        dual_print(f"[DEBUG] Limiting to age range: {start_year}-{end_year}")
    if ENROLLMENT_DATES:
        dual_print(f"[DEBUG] Limiting to enrollment dates: {ENROLLMENT_DATES}")
    
    # Initialize debug data collection (will be populated inside sheet loop)
    debug_data = []
    # Store slope6 normalization parameters per (EnrollmentDate, YoB, Dose) for later summary printing
    slope6_params_map = {}
    
    # For Monte Carlo mode: collect KCOR values at end of 2022 for summary
    mc_summary_data = [] if MONTE_CARLO_MODE else None
    
    # Slope6: Select fixed baseline window (fit window, no data collection needed since window is fixed)
    baseline_window = select_slope6_baseline_window([], dual_print)  # Empty list since we don't need data
    dual_print(f"[Slope6] Using fit window: {baseline_window[0]} to {baseline_window[1]}")
    dual_print(f"[Slope6] Application endpoint: {SLOPE6_APPLICATION_END_ISO} (rightmost point for determining centerpoint)")
    
    for sh in sheets_to_process:
        dual_print(f"[Info] Processing sheet: {sh}")
        df = pd.read_excel(src_path, sheet_name=sh)
        # prep
        df["DateDied"] = pd.to_datetime(df["DateDied"])
        # (Removed temporary diagnostics)
        
        # In MC mode, add YearOfBirth column if missing (should be -2 for all-ages)
        if MONTE_CARLO_MODE and "YearOfBirth" not in df.columns:
            df["YearOfBirth"] = -2
        
        # Filter out unreasonably large birth years (keep -1 for "not available", -2 for all ages in MC mode)
        if MONTE_CARLO_MODE:
            # In MC mode, YearOfBirth=-2 is valid (all ages aggregated)
            df = df[df["YearOfBirth"] <= 2020]
            # Filter to only YOB=-2 in MC mode (all ages aggregated)
            df = df[df["YearOfBirth"] == -2]
            dual_print(f"[DEBUG] Monte Carlo mode: Filtered to only YOB=-2 (all ages aggregated): {len(df)} rows")
        else:
            df = df[df["YearOfBirth"] <= 2020]
        
        # Filter to start from enrollment date
        if MONTE_CARLO_MODE:
            # In MC mode, all iterations use enrollment date 2022-06
            enrollment_date_str = "2022-06"
            enrollment_date = pd.to_datetime(enrollment_date_str + '-1', format='%G-%V-%u', errors='coerce')
            df = df[df["DateDied"] >= enrollment_date]
            dual_print(f"[DEBUG] Monte Carlo mode: Using enrollment date {enrollment_date.strftime('%m/%d/%Y')} (2022-06) for iteration {sh}")
            dual_print(f"[DEBUG] Filtered to start from enrollment date: {len(df)} rows")
        elif "_" in sh:
            # Normal mode: parse enrollment date from sheet name (format: YYYY_WW)
            year_str, week_str = sh.split("_")
            enrollment_year = int(year_str)
            enrollment_week = int(week_str)
            # Convert ISO week to date more accurately
            # Note: datetime and timedelta are already imported at module level
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
        
        # Apply debug age filter (skip in MC mode since YearOfBirth=-2 is always used)
        if YEAR_RANGE and not MONTE_CARLO_MODE:
            start_year, end_year = YEAR_RANGE
            df = df[(df["YearOfBirth"] >= start_year) & (df["YearOfBirth"] <= end_year)]
            dual_print(f"[DEBUG] Filtered to {len(df)} rows for ages {start_year}-{end_year}")
        
        # Apply sheet-specific dose filtering
        if MONTE_CARLO_MODE:
            # In MC mode, use 2022-06 dose pairs and limit to doses 0-3
            dose_pairs = get_dose_pairs("2022_06")
            max_dose = 3  # MC mode uses max_dose=3
            valid_doses = [0, 1, 2, 3]
            df = df[df["Dose"].isin(valid_doses)]
            dual_print(f"[DEBUG] Monte Carlo mode: Filtered to doses {valid_doses} (max dose {max_dose}): {len(df)} rows")
        else:
            dose_pairs = get_dose_pairs(sh)
            max_dose = max(max(pair) for pair in dose_pairs)
            valid_doses = list(range(max_dose + 1))  # Include all doses from 0 to max_dose
            df = df[df["Dose"].isin(valid_doses)]
            dual_print(f"[DEBUG] Filtered to doses {valid_doses} (max dose {max_dose}): {len(df)} rows")
        
        # Aggregate across sexes for each dose/date/age combination
        # Optionally bucket YearOfBirth into AGE_RANGE-year bins (e.g., 10 -> 1920, 1930, ...)
        # Skip AGE_RANGE bucketing in MC mode to preserve YOB=-2
        if not MONTE_CARLO_MODE:
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
        # Suppress debug message in MC mode (always aggregated, not informative)
        if not MONTE_CARLO_MODE:
            dual_print(f"[DEBUG] Aggregated across sexes: {len(df)} rows")
        
        df = df.sort_values(["YearOfBirth","Dose","DateDied"]).reset_index(drop=True)
        # person-time proxy and MR (internal probability per week)
        df["PT"]   = df["Alive"].astype(float).clip(lower=0.0)
        df["Dead"] = df["Dead"].astype(float).clip(lower=0.0)
        df["MR"]   = np.where(df["PT"] > 0, df["Dead"]/(df["PT"] + EPS), np.nan)
        df["t"]    = df.groupby(["YearOfBirth","Dose"]).cumcount().astype(float)

        # Apply centered moving average smoothing before quantile regression
        df = apply_moving_average(df)
        
        # Debug: Show effect of moving average
        # if DEBUG_VERBOSE:
        #     print(f"\n[DEBUG] Centered moving average smoothing (deprecated):")
        #     for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        #         if len(g) > 0:
        #             mr_orig = g["MR"].values
        #             mr_smooth = g["MR_smooth"].values
        #             print(f"  Age {yob}, Dose {dose}:")
        #             print(f"    Original MR range: {mr_orig.min():.6f} to {mr_orig.max():.6f}")
        #             print(f"    Smoothed MR range: {mr_smooth.min():.6f} to {mr_smooth.max():.6f}")
        
        # Monte Carlo mode: simplified processing path (skip SA mode and dynamic slopes)
        if MONTE_CARLO_MODE:
            # MC mode: skip SA mode and dynamic slope calculation, go directly to normalization
            # Remove legacy slope-adjusted columns and scale factors; keep raw MR only
            df["slope"] = 0.0
            df["scale_factor"] = 1.0
            df["MR_adj"] = df["MR"]
            
            # Apply discrete cumulative-hazard transform for mathematical exactness
            # Apply Slope6 normalization: Force linear mode for MC (all YOB=-2)
            effective_sheet_name_for_processing = "2022_06"
            df["sheet_name"] = effective_sheet_name_for_processing
            
            # Compute Slope6 normalization parameters for this sheet (force linear mode)
            slope6_params = compute_slope6_normalization(df, baseline_window, effective_sheet_name_for_processing, dual_print, force_linear_mode=True)
            
            # Persist normalization parameters
            for (yob_k, dose_k), params in slope6_params.items():
                try:
                    if isinstance(params, dict):
                        slope6_params_map[(effective_sheet_name_for_processing, int(yob_k), int(dose_k))] = params
                    else:
                        slope6_params_map[(effective_sheet_name_for_processing, int(yob_k), int(dose_k))] = {
                            "mode": "none",
                            "a": 0.0,
                            "b": 0.0,
                            "c": 0.0,
                            "t_mean": 0.0,
                            "tau": None
                        }
                except Exception:
                    if isinstance(params, dict):
                        slope6_params_map[(effective_sheet_name_for_processing, yob_k, dose_k)] = params
                    else:
                        slope6_params_map[(effective_sheet_name_for_processing, yob_k, dose_k)] = {
                            "mode": "none",
                            "a": 0.0,
                            "b": 0.0,
                            "c": 0.0,
                            "t_mean": 0.0,
                            "tau": None
                        }
            
            # Note: Do NOT modify raw MR. Normalization is applied later at the hazard level.
            mr_used = df["MR"]
            
            # Clip to avoid log(0) and ensure numerical stability
            df["hazard_raw"] = hazard_from_mr_improved(np.clip(mr_used, 0.0, 0.999))
            # Initialize adjusted hazard equal to raw
            df["hazard_adj"] = df["hazard_raw"]
            
            # Apply Slope6 normalization at hazard level using stored parameters
            for (yob, dose), g in df.groupby(["YearOfBirth", "Dose"], sort=False):
                params = slope6_params.get((yob, dose), {})
                if not isinstance(params, dict):
                    continue
                mode = params.get("mode", "none")
                if mode == "linear":
                    # Linear normalization: h_adj = h_raw * exp(-b * (t - t_mean))
                    a = params.get("a", 0.0)
                    b = params.get("b", 0.0)
                    t_mean = params.get("t_mean", 0.0)
                    if np.isfinite(b) and np.abs(b) > EPS:
                        g_sorted = g.sort_values("DateDied").copy()
                        t_vals = g_sorted["t"].values
                        # Apply linear normalization: exp(-b * (t - t_mean))
                        norm_factor = safe_exp(-b * (t_vals - t_mean))
                        df.loc[g_sorted.index, "hazard_adj"] = g_sorted["hazard_raw"].values * norm_factor
                elif mode == "slope8":
                    # Should not happen in MC mode, but handle gracefully
                    C = params.get("C", 0.0)
                    ka = params.get("ka", 0.0)
                    kb = params.get("kb", 0.0)
                    tau = params.get("tau", None)
                    if tau is not None and np.isfinite(tau) and tau > EPS:
                        g_sorted = g.sort_values("DateDied").copy()
                        s_vals = g_sorted["t"].values
                        # Apply slope8 normalization
                        norm_factor = safe_exp(-kb * s_vals - (ka - kb) * tau * (1.0 - safe_exp(-s_vals / tau)))
                        df.loc[g_sorted.index, "hazard_adj"] = g_sorted["hazard_raw"].values * norm_factor
            
            # Apply DYNAMIC_HVE_SKIP_WEEKS: start accumulation at this week index
            df["hazard_eff"] = np.where(df["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df["hazard_adj"], 0.0)
            df["CH"] = df.groupby(["YearOfBirth", "Dose"])["hazard_eff"].cumsum()
            df["CH_actual"] = df.groupby(["YearOfBirth", "Dose"])["hazard_eff"].cumsum()
            df["cumPT"] = df.groupby(["YearOfBirth", "Dose"])["PT"].cumsum()
            df["cumD_adj"] = df["CH"] * df["cumPT"]
            df["cumD_unadj"] = df.groupby(["YearOfBirth", "Dose"])["Dead"].cumsum()
            
            # Build KCOR rows using shared function
            out_sh = build_kcor_rows(df, effective_sheet_name_for_processing, dual_print, slope6_params_map)
            # Compute KCOR_ns and merge into output
            kcor_ns = build_kcor_ns_rows(df, effective_sheet_name_for_processing)
            if not kcor_ns.empty and not out_sh.empty:
                out_sh = pd.merge(
                    out_sh,
                    kcor_ns,
                    on=["EnrollmentDate","Date","YearOfBirth","Dose_num","Dose_den"],
                    how="left"
                )
            all_out.append(out_sh)
            
            # For Monte Carlo mode: collect KCOR values at end of 2022 for summary
            if MONTE_CARLO_MODE:
                out_sh_copy = out_sh.copy()
                out_sh_copy["Date"] = pd.to_datetime(out_sh_copy["Date"])
                target_date = pd.to_datetime("2022-12-31")
                if not out_sh_copy.empty:
                    out_sh_2022 = out_sh_copy[out_sh_copy["Date"].dt.year <= 2023]
                    if not out_sh_2022.empty:
                        diffs = (out_sh_2022["Date"] - target_date).abs()
                        idx_closest = diffs.idxmin()
                        closest_date = out_sh_2022.loc[idx_closest, "Date"]
                        closest_data = out_sh_2022[out_sh_2022["Date"] == closest_date]
                        for _, row in closest_data.iterrows():
                            mc_summary_data.append({
                                "Iteration": int(sh) if sh.isdigit() else sh,
                                "Dose_num": row["Dose_num"],
                                "Dose_den": row["Dose_den"],
                                "YearOfBirth": row["YearOfBirth"],
                                "Date": row["Date"],
                                "KCOR": row["KCOR"],
                                "CI_lower": row.get("CI_lower", np.nan),
                                "CI_upper": row.get("CI_upper", np.nan)
                            })
            
            # Collect per-pair deaths details for new output sheet
            pair_details = build_kcor_o_deaths_details(df, effective_sheet_name_for_processing)
            if not pair_details.empty:
                pair_deaths_all.append(pair_details)
            
            # Continue to next sheet in MC mode
            continue
        
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
                                # Use explicit windows defined by [start, end] inclusive around offsets
                                w1s, w1e = off1loc, off1loc
                                w2s, w2e = off2loc, off2loc
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
                    df2 = adjust_mr(df2, slopes, t0=KCOR_NORMALIZATION_WEEKS_EFFECTIVE)
                    # Add slope and scale_factor columns (required by downstream build_kcor_rows)
                    df2["slope"] = df2.apply(lambda row: slopes.get((row["YearOfBirth"], row["Dose"]), 0.0), axis=1)
                    df2["slope"] = np.clip(df2["slope"], -10.0, 10.0)
                    df2["scale_factor"] = df2.apply(lambda row: safe_exp(-df2["slope"].iloc[row.name] * (row["t"] - float(KCOR_NORMALIZATION_WEEKS_EFFECTIVE))), axis=1)
                    df2["hazard"] = hazard_from_mr_improved(df2["MR_adj"]) 
                    # Apply DYNAMIC_HVE_SKIP_WEEKS: start accumulation at this week index
                    df2["hazard_eff"] = np.where(df2["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df2["hazard"], 0.0)
                    df2["MR_eff"] = np.where(df2["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df2["MR"], 0.0)
                    df2["CH"] = df2.groupby(["YearOfBirth","Dose"]) ['hazard_eff'].cumsum()
                    df2["CH_actual"] = df2.groupby(["YearOfBirth","Dose"]) ['MR_eff'].cumsum()
                    df2["cumPT"] = df2.groupby(["YearOfBirth","Dose"]) ['PT'].cumsum()
                    df2["cumD_adj"] = df2["CH"] * df2["cumPT"]
                    df2["cumD_unadj"] = df2.groupby(["YearOfBirth","Dose"]) ['Dead'].cumsum()
                    out_sh_sa = build_kcor_rows(df2, sh, dual_print)
                    # Compute KCOR_ns and merge into SA output
                    kcor_ns_sa = build_kcor_ns_rows(df2, sh)
                    if not kcor_ns_sa.empty and not out_sh_sa.empty:
                        out_sh_sa = pd.merge(
                            out_sh_sa,
                            kcor_ns_sa,
                            on=["EnrollmentDate","Date","YearOfBirth","Dose_num","Dose_den"],
                            how="left"
                        )
                    out_sh_sa["param_slope_start"] = int(start_val)
                    out_sh_sa["param_slope_length"] = int(length_val)
                    produced_outputs.append(out_sh_sa)
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
        
        # Debug slope lines suppressed

        # Total slope by YoB (Alive-weighted at enrollment week t=0 across included doses)
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

        for yob in sorted(df["YearOfBirth"].unique()):
            numerator = 0.0
            denominator = 0.0
            alive_vax = 0.0
            alive_total = 0.0
            # Collect alive counts per dose present at t=0 for this YoB
            dose_alive = []  # list of (dose, alive_count)
            for dose in range(max_dose + 1):
                if (yob, dose) in slopes:
                    w = float(alive0_map.get((yob, dose), 0.0))
                    if np.isfinite(w) and w > 0.0:
                        numerator += w * float(slopes.get((yob, dose), 0.0))
                        denominator += w
                        alive_total += w
                        if dose >= 1:
                            alive_vax += w
                        dose_alive.append((dose, w))
            pct_vax = (alive_vax / alive_total * 100.0) if alive_total > EPS else np.nan
            if alive_total > EPS and dose_alive:
                percents = [f"{(w / alive_total * 100.0):.0f}%" for (dose, w) in dose_alive]
                alive_str = " ".join(percents)
            else:
                alive_str = "-"
            total_str = f" total={int(alive_total)}" if alive_total > EPS else ""
            if np.isfinite(pct_vax):
                dual_print(f"  YoB {yob}, % vaxxed={pct_vax:.0f}  alive_by_dose= {alive_str}{total_str}")
            else:
                dual_print(f"  YoB {yob}, % vaxxed=-  alive_by_dose= {alive_str}{total_str}")

        # done printing total slopes so we can print the note on how to interpret it
        # dual_print("\nNote: Note that computed mortality rate slopes should be positive since people don't get younger.")
        

        # Slope normalization: Slope6 applies only at hazard level.
        
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

        # Remove legacy slope-adjusted columns and scale factors; keep raw MR only
        df["slope"] = 0.0
        df["scale_factor"] = 1.0
        df["MR_adj"] = df["MR"]
        # Bug diagnostics for specific cohort/date (requested):
        # Log Alive, Dead, MR (raw) and MR_adj for YoB=1950, Dose=2 on 2022-01-10 and 2022-01-17 in 2021_13 sheet
        # removed temporary diagnostics
        
        # Apply discrete cumulative-hazard transform for mathematical exactness
        # Apply Slope8 normalization: Quantile regression depletion-mode normalization for all cohorts
        # Note: baseline_window (fit window) was selected globally before processing sheets
        
        # Add sheet_name to df for compute_slope6_normalization
        # In MC mode, use "2022_06" as the enrollment date, not the iteration number
        effective_sheet_name_for_processing = "2022_06" if MONTE_CARLO_MODE else sh
        df["sheet_name"] = effective_sheet_name_for_processing
        
        # Compute Slope6 normalization parameters for this sheet
        # In MC mode, force linear mode (skip slope8 attempts)
        slope6_params = compute_slope6_normalization(df, baseline_window, effective_sheet_name_for_processing, dual_print, force_linear_mode=MONTE_CARLO_MODE)
        
        # Persist normalization parameters (dict with mode, a, b, c, t_mean, tau) for this sheet for later summary printing
        # Use effective_sheet_name_for_processing (2022_06 in MC mode) for consistency
        for (yob_k, dose_k), params in slope6_params.items():
            try:
                # Store params dict
                if isinstance(params, dict):
                    slope6_params_map[(effective_sheet_name_for_processing, int(yob_k), int(dose_k))] = params
                else:
                    # Legacy format: shouldn't happen with new code, but be safe
                    slope6_params_map[(effective_sheet_name_for_processing, int(yob_k), int(dose_k))] = {
                        "mode": "none",
                        "a": 0.0,
                        "b": 0.0,
                        "c": 0.0,
                        "t_mean": 0.0,
                        "tau": None  # tau only valid for slope8 mode
                    }
            except Exception:
                if isinstance(params, dict):
                    slope6_params_map[(effective_sheet_name_for_processing, yob_k, dose_k)] = params
                else:
                    slope6_params_map[(effective_sheet_name_for_processing, yob_k, dose_k)] = {
                        "mode": "none",
                        "a": 0.0,
                        "b": 0.0,
                        "c": 0.0,
                        "t_mean": 0.0,
                        "tau": None  # tau only valid for slope8 mode
                    }

        # Note: Do NOT modify raw MR. Normalization is applied later at the hazard level.
        # Czech-specific MR correction removed; use raw MR moving forward
        mr_used = df["MR"]

        # Clip to avoid log(0) and ensure numerical stability
        df["hazard_raw"] = hazard_from_mr_improved(np.clip(mr_used, 0.0, 0.999))
        # Initialize adjusted hazard equal to raw
        df["hazard_adj"] = df["hazard_raw"]
        
        # Apply Slope6 normalization at hazard level using time-centered approach
        # Linear mode: h_norm = h * exp(-b_lin * t_c) where t_c = t - t_mean
        # Slope8 mode: h_norm = h * exp(-kb*s - (ka - kb)*tau*(1 - exp(-s/tau))) where s = t (no centering, C term excluded)
        # t_mean is computed from application window (enrollment_date to 2024-16) for linear mode
        if isinstance(slope6_params, dict) and len(slope6_params) > 0:
            try:
                # Track normalization application for debugging
                norm_debug_samples = []
                
                def apply_slope6_norm(row):
                    params = slope6_params.get((row["YearOfBirth"], row["Dose"]), {
                        "mode": "none",
                        "a": 0.0,
                        "b": 0.0,
                        "c": 0.0,
                        "t_mean": 0.0,
                        "tau": None  # tau only valid for slope8 mode
                    })
                    
                    if not isinstance(params, dict) or params.get("mode") == "none":
                        if len(norm_debug_samples) < 3:
                            norm_debug_samples.append({
                                "yob": row["YearOfBirth"], "dose": row["Dose"],
                                "reason": "no_params" if params.get("mode") == "none" else "not_dict",
                                "mode": params.get("mode", "unknown") if isinstance(params, dict) else type(params).__name__
                            })
                        return row["hazard_raw"]
                    
                    mode = params.get("mode", "linear")
                    
                    if mode == "linear":
                        b = params.get("b", 0.0)
                        t_mean = params.get("t_mean", 0.0)
                        # Compute centered time
                        t_c = row["t"] - t_mean
                        # Linear mode: h_norm = h * exp(-b * t_c)
                        result = row["hazard_raw"] * np.exp(-b * t_c)
                        if len(norm_debug_samples) < 3 and abs(b) > 1e-10:
                            norm_debug_samples.append({
                                "yob": row["YearOfBirth"], "dose": row["Dose"], "mode": mode,
                                "b": b, "t_c": t_c, "norm_factor": np.exp(-b * t_c),
                                "hazard_raw": row["hazard_raw"], "result": result
                            })
                        return result
                    elif mode == "slope8":
                        # C term excluded so adjustment is 1 at s=0
                        ka = params.get("ka", 0.0)
                        kb = params.get("kb", 0.0)
                        tau = params.get("tau", 1.0)
                        # Use s = t (time since enrollment, NOT centered)
                        s = row["t"]
                        # Ensure tau is positive and finite
                        if not np.isfinite(tau) or tau <= EPS:
                            tau = 1.0
                        # Slope8 mode: h_norm = h * exp(-kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))
                        norm_factor = np.exp(-kb * s - (ka - kb) * tau * (1.0 - np.exp(-s / (tau + EPS))))
                        # Ensure norm_factor is finite
                        if not np.isfinite(norm_factor):
                            norm_factor = 1.0
                        result = row["hazard_raw"] * norm_factor
                        # Collect debug samples for slope8
                        if len(norm_debug_samples) < 3:
                            norm_debug_samples.append({
                                "yob": row["YearOfBirth"], "dose": row["Dose"], "mode": mode,
                                "C": C, "ka": ka, "kb": kb, "tau": tau, "s": s,
                                "norm_factor": norm_factor, "hazard_raw": row["hazard_raw"], "result": result
                            })
                        return result
                    else:
                        # Unknown mode, no normalization
                        if len(norm_debug_samples) < 3:
                            norm_debug_samples.append({
                                "yob": row["YearOfBirth"], "dose": row["Dose"],
                                "reason": "unknown_mode", "mode": mode
                            })
                        return row["hazard_raw"]
                
                # Debug: Check parameters before applying normalization
                if DEBUG_VERBOSE:
                    dual_print(f"[DEBUG] Checking normalization parameters for {sh}:")
                    dual_print(f"  Total cohorts in slope6_params: {len(slope6_params)}")
                    sample_count = 0
                    for (yob, dose), params in list(slope6_params.items())[:5]:  # Show first 5
                        if isinstance(params, dict):
                            mode = params.get("mode", "none")
                            if mode == "slope8":
                                C = params.get("C", 0.0)
                                ka = params.get("ka", 0.0)
                                kb = params.get("kb", 0.0)
                                tau = params.get("tau", 1.0)
                                dual_print(f"  YoB={yob}, Dose={dose}: mode={mode}, C={C:.6e}, ka={ka:.6e}, kb={kb:.6e}, tau={tau:.6e}")
                            else:
                                dual_print(f"  YoB={yob}, Dose={dose}: mode={mode}")
                        sample_count += 1
                    if len(slope6_params) > 5:
                        dual_print(f"  ... and {len(slope6_params) - 5} more cohorts")
                
                df["hazard_adj"] = df.apply(apply_slope6_norm, axis=1)
                # Numerical safety: floor at tiny epsilon
                df["hazard_adj"] = np.clip(df["hazard_adj"], 0.0, None)
                
                # Debug: Report normalization samples
                if DEBUG_VERBOSE and norm_debug_samples:
                    dual_print(f"[DEBUG] Normalization samples for {sh}:")
                    for i, sample in enumerate(norm_debug_samples):
                        if "reason" in sample:
                            dual_print(f"  Sample {i+1}: YoB={sample['yob']}, Dose={sample['dose']}, reason={sample['reason']}, mode={sample.get('mode', 'N/A')}")
                        elif sample.get("mode") == "slope8":
                            dual_print(f"  Sample {i+1}: YoB={sample['yob']}, Dose={sample['dose']}, mode={sample['mode']}, "
                                     f"C={sample['C']:.6e}, ka={sample['ka']:.6e}, kb={sample['kb']:.6e}, tau={sample['tau']:.6e}, "
                                     f"norm_factor={sample['norm_factor']:.6e}, hazard_raw={sample['hazard_raw']:.6e}, result={sample['result']:.6e}")
                        else:
                            dual_print(f"  Sample {i+1}: YoB={sample['yob']}, Dose={sample['dose']}, mode={sample.get('mode', 'N/A')}, "
                                     f"norm_factor={sample.get('norm_factor', 1.0):.6e}")
                
                # Debug: Verify normalization was applied
                if DEBUG_VERBOSE:
                    # Check if hazard_adj differs from hazard_raw
                    diff_mask = np.abs(df["hazard_adj"] - df["hazard_raw"]) > 1e-10
                    n_different = diff_mask.sum()
                    n_total = len(df)
                    dual_print(f"[DEBUG] Normalization check for {sh}: {n_different}/{n_total} rows have different hazard_adj vs hazard_raw")
                    if n_different > 0:
                        sample_diff = df[diff_mask].head(3)
                        for idx, row in sample_diff.iterrows():
                            dual_print(f"  Sample: YoB={row['YearOfBirth']}, Dose={row['Dose']}, t={row['t']:.1f}, "
                                     f"hazard_raw={row['hazard_raw']:.6e}, hazard_adj={row['hazard_adj']:.6e}, "
                                     f"ratio={row['hazard_adj']/row['hazard_raw']:.6f}")
                    else:
                        dual_print(f"[DEBUG] WARNING: No normalization applied! All hazard_adj == hazard_raw")
                        # Check why - sample a few rows
                        for idx, row in df.head(5).iterrows():
                            params_check = slope6_params.get((row["YearOfBirth"], row["Dose"]), None)
                            if params_check is None:
                                dual_print(f"  Row {idx}: YoB={row['YearOfBirth']}, Dose={row['Dose']} - NO PARAMS FOUND")
                            elif isinstance(params_check, dict):
                                mode_check = params_check.get("mode", "none")
                                dual_print(f"  Row {idx}: YoB={row['YearOfBirth']}, Dose={row['Dose']} - mode={mode_check}")
                            else:
                                dual_print(f"  Row {idx}: YoB={row['YearOfBirth']}, Dose={row['Dose']} - params type={type(params_check)}")
                    
                    modes_used = {}
                    for (yob, dose), params in slope6_params.items():
                        if isinstance(params, dict):
                            mode = params.get("mode", "none")
                            modes_used[mode] = modes_used.get(mode, 0) + 1
                    if modes_used:
                        mode_str = ", ".join([f"{mode}: {count}" for mode, count in modes_used.items()])
                        dual_print(f"[DEBUG] Normalization modes applied for {sh}: {mode_str}")
                        # Check if any slope8 normalizations were applied
                        slope8_count = sum(1 for p in slope6_params.values() 
                                          if isinstance(p, dict) and p.get("mode") == "slope8")
                        if slope8_count > 0:
                            dual_print(f"[DEBUG] Slope8 normalization applied to {slope8_count} cohort(s)")
                        # Check for abnormal fits
                        abnormal_count = sum(1 for p in slope6_params.values() 
                                           if isinstance(p, dict) and p.get("abnormal_fit", False))
                        if abnormal_count > 0:
                            dual_print(f"[DEBUG] Found {abnormal_count} cohort(s) with abnormal fits")
                
                # Store slope and scale_factor for backward compatibility with output schema
                # slope represents b (the linear slope parameter)
                # scale_factor represents the normalization factor
                def get_slope(r):
                    params = slope6_params.get((r["YearOfBirth"], r["Dose"]), {
                        "mode": "none",
                        "b": 0.0
                    })
                    if isinstance(params, dict):
                        mode = params.get("mode", "none")
                        if mode == "slope8":
                            # For slope8, return b_original (the original b_lin from linear fit)
                            return float(params.get("b_original", 0.0))
                        else:
                            # For linear mode, return b
                            return float(params.get("b", 0.0))
                    return 0.0
                
                def get_scale_factor(r):
                    params = slope6_params.get((r["YearOfBirth"], r["Dose"]), {
                        "mode": "none",
                        "b": 0.0,
                        "c": 0.0,
                        "t_mean": 0.0
                    })
                    if not isinstance(params, dict):
                        return 1.0
                    
                    mode = params.get("mode", "none")
                    if mode == "none":
                        return 1.0
                    
                    if mode == "linear":
                        b = params.get("b", 0.0)
                        t_mean = params.get("t_mean", 0.0)
                        t_c = r["t"] - t_mean
                        return np.exp(-b * t_c)
                    elif mode == "slope8":
                        # C term excluded so adjustment is 1 at s=0
                        ka = params.get("ka", 0.0)
                        kb = params.get("kb", 0.0)
                        tau = params.get("tau", 1.0)
                        # Use s = t (time since enrollment, NOT centered)
                        s = r["t"]
                        # Slope8 scale factor: exp(-kb*s - (ka - kb)*tau*(1 - exp(-s/tau))) [C term excluded]
                        return np.exp(-kb * s - (ka - kb) * tau * (1.0 - np.exp(-s / (tau + EPS))))
                    else:
                        return 1.0
                
                df["slope"] = df.apply(get_slope, axis=1)
                df["scale_factor"] = df.apply(get_scale_factor, axis=1)
            except Exception as e:
                dual_print(f"SLOPE6_ERROR,EnrollmentDate={sh},error={str(e)}")
                import traceback
                if DEBUG_VERBOSE:
                    dual_print(f"[DEBUG] Full traceback for normalization error:")
                    dual_print(traceback.format_exc())
                # Fallback: no normalization
                df["hazard_adj"] = df["hazard_raw"]  # Ensure hazard_adj is set even on error
                df["slope"] = 0.0
                df["scale_factor"] = 1.0
        else:
            # No normalization parameters available
            df["slope"] = 0.0
            df["scale_factor"] = 1.0
        
        # Backward compatibility: keep 'hazard' as the adjusted hazard used in KCOR
        df["hazard"] = df["hazard_adj"]
        # Extra bug diagnostics for the 1950/Dose2 cohort on two dates — include hazard and CH
        # removed temporary diagnostics
        
        # Apply DYNAMIC_HVE_SKIP_WEEKS to accumulation start
        df["hazard_eff"] = np.where(df["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df["hazard"], 0.0)
        df["MR_eff"] = np.where(df["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), df["MR"], 0.0)
        
        # Calculate cumulative hazard (mathematically exact, not approximation)
        df["CH"] = df.groupby(["YearOfBirth","Dose"]) ["hazard_eff"].cumsum()

        # Debug: print CH_raw vs CH_adj for specific cohort on request
        try:
            if sh in ("2021_24", "2021-24"):
                dbg_mask = (df["YearOfBirth"] == 1940) & (df["Dose"] == 1)
                target = df[dbg_mask].sort_values("t")
                if not target.empty:
                    target_win = target.copy()
                    hraw_eff = target_win["hazard_raw"].to_numpy()
                    hadj_eff = target_win["hazard_adj"].to_numpy()
                    ch_raw_dbg = np.cumsum(hraw_eff)
                    ch_adj_dbg = np.cumsum(hadj_eff)
                    # Get Slope6 slope parameter for debugging
                    if isinstance(slope6_params, dict):
                        params_dbg = slope6_params.get((1940, 1), {"b": np.nan})
                        b_c_dbg = params_dbg.get("b", np.nan) if isinstance(params_dbg, dict) else np.nan
                    else:
                        b_c_dbg = np.nan
                    pass
                    # CSV header
                    # dual_print("SLOPE2_DEBUG,EnrollmentDate,YearOfBirth,Dose,Date,t,CH_raw,CH_adj,beta")
                    dates_iso = pd.to_datetime(target_win["DateDied"]).dt.date.astype(str).tolist()
                    for d_i, t_i, cr, ca in zip(dates_iso, target_win["t"].tolist(), ch_raw_dbg.tolist(), ch_adj_dbg.tolist()):
                        # dual_print(f"SLOPE2_DEBUG,{sh},1940,1,{d_i},{int(t_i)},{cr:.6f},{ca:.6f},{beta_dbg}")
                        pass
        except Exception:
            pass
        
        # Keep unadjusted data for comparison
        df["CH_actual"] = df.groupby(["YearOfBirth","Dose"]) ["MR_eff"].cumsum()
        
        # Keep cumD_adj for backward compatibility (now represents adjusted cumulative deaths)
        df["cumPT"] = df.groupby(["YearOfBirth","Dose"]) ["PT"].cumsum()
        df["cumD_adj"] = df["CH"] * df["cumPT"]
        
        # Keep unadjusted data for comparison
        df["cumD_unadj"] = df.groupby(["YearOfBirth","Dose"]) ["Dead"].cumsum()
        
        # Collect debug data for this sheet (after all columns are created)
        # Build per-dose per-age aggregates (needed for negative-control direct age comparisons)
        dose_pairs = get_dose_pairs(sh)
        max_dose = max(max(pair) for pair in dose_pairs)
        for dose in range(max_dose + 1):
            dose_df = df[df["Dose"] == dose]
            dose_data = (
                dose_df.groupby(["DateDied", "YearOfBirth"]).agg({
                    "ISOweekDied": "first",
                    "Dead": "sum",
                    "Alive": "sum",
                    "PT": "sum",
                    "MR": "mean",
                    "CH": "mean",
                    "CH_actual": "mean",
                    "hazard_raw": "mean",
                    "hazard_adj": "mean",
                    "cumD_unadj": "mean",
                    "t": "first",
                }).reset_index().sort_values(["DateDied", "YearOfBirth"]) 
            )
            # Collect minimal columns for NC
            if not dose_data.empty:
                tmp = dose_data[["DateDied","YearOfBirth","CH","t"]].copy()
                tmp["EnrollmentDate"] = sh
                tmp["Dose"] = dose
                by_dose_nc_all.append(tmp)
            # Populate debug_data (by_dose) for all doses
            for _, row in dose_data.iterrows():
                # Get slope from Slope6 parameters (stored in df["slope"] column)
                slope_val = row.get("slope", 0.0) if "slope" in row else 0.0
                debug_data.append({
                    "EnrollmentDate": sh,
                    "Date": row["DateDied"].date(),
                    "YearOfBirth": row["YearOfBirth"],
                    "Dose": dose,
                    "ISOweek": row["ISOweekDied"],
                    "Dead": row["Dead"],
                    "Alive": row["Alive"],
                    "MR": row["MR"],
                    # removed Cum_MR and Cum_MR_Actual per request
                    "Hazard": row["hazard_raw"],
                    "Hazard_adj": row["hazard_adj"],
                    "Slope": slope_val,
                    "Cum_deaths": row["cumD_unadj"],
                    "Cumu_Person_Time": row["PT"],
                    "Time_Index": row["t"]
                })
        
        # (Removed temporary diagnostics)
        
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

        # In MC mode, use "2022_06" as sheet_name for enrollment date processing
        # but keep iteration number for output sheet naming
        effective_sheet_name = "2022_06" if MONTE_CARLO_MODE else sh
        out_sh = build_kcor_rows(df, effective_sheet_name, dual_print, slope6_params_map)
        # Compute KCOR_ns and merge into output
        # In MC mode, use effective_sheet_name (2022_06) for consistency
        kcor_ns = build_kcor_ns_rows(df, effective_sheet_name)
        if not kcor_ns.empty and not out_sh.empty:
            out_sh = pd.merge(
                out_sh,
                kcor_ns,
                on=["EnrollmentDate","Date","YearOfBirth","Dose_num","Dose_den"],
                how="left"
            )
        all_out.append(out_sh)
        
        # For Monte Carlo mode: collect KCOR values at end of 2022 for summary
        if MONTE_CARLO_MODE:
            # Get KCOR values at end of 2022 (2022-12-31) for this iteration
            out_sh_copy = out_sh.copy()
            out_sh_copy["Date"] = pd.to_datetime(out_sh_copy["Date"])
            target_date = pd.to_datetime("2022-12-31")
            # Find closest date to target
            if not out_sh_copy.empty:
                # Filter to dates in 2022 or early 2023
                out_sh_2022 = out_sh_copy[out_sh_copy["Date"].dt.year <= 2023]
                if not out_sh_2022.empty:
                    diffs = (out_sh_2022["Date"] - target_date).abs()
                    idx_closest = diffs.idxmin()
                    closest_date = out_sh_2022.loc[idx_closest, "Date"]
                    # Collect data for all dose pairs at closest date
                    closest_data = out_sh_2022[out_sh_2022["Date"] == closest_date]
                    for _, row in closest_data.iterrows():
                        mc_summary_data.append({
                            "Iteration": int(sh) if sh.isdigit() else sh,
                            "Dose_num": row["Dose_num"],
                            "Dose_den": row["Dose_den"],
                            "YearOfBirth": row["YearOfBirth"],
                            "Date": row["Date"],
                            "KCOR": row["KCOR"],
                            "CI_lower": row.get("CI_lower", np.nan),
                            "CI_upper": row.get("CI_upper", np.nan)
                        })
        
        # Collect per-pair deaths details for new output sheet
        # In MC mode, use effective_sheet_name (2022_06) for consistency
        pair_details = build_kcor_o_deaths_details(df, effective_sheet_name)
        if not pair_details.empty:
            pair_deaths_all.append(pair_details)

    # Combine all results
    combined = pd.concat(all_out, ignore_index=True).sort_values(["EnrollmentDate","YearOfBirth","Dose_num","Dose_den","Date"])

    # Negative control mode: output summarized KCOR by (EnrollmentDate, YoB1, YoB2=YoB1+10, Dose)
    if int(NEGATIVE_CONTROL_MODE) == 1:
        try:
            neg_rows = []
            combined["Date"] = pd.to_datetime(combined["Date"])
            # Prepare per-dose, per-age cumulative hazard time series from earlier aggregation
            by_dose_nc = pd.concat(by_dose_nc_all, ignore_index=True) if by_dose_nc_all else pd.DataFrame()
            for sheet_name in sorted(combined["EnrollmentDate"].unique()):
                target_str = KCOR_REPORTING_DATE.get(sheet_name)
                target_dt = pd.to_datetime(target_str) if target_str else None
                data_sheet = combined[combined["EnrollmentDate"] == sheet_name]
                if data_sheet.empty:
                    continue
                # pick closest date to target; else latest
                if target_dt is not None:
                    diffs = (data_sheet["Date"] - target_dt)
                    # robust absolute difference for older pandas versions
                    idxmin = int(np.argmin(np.abs(diffs.to_numpy())))
                    report_date = data_sheet.iloc[idxmin]["Date"]
                else:
                    report_date = data_sheet["Date"].max()
                end_data = data_sheet[data_sheet["Date"] == report_date]
                # doses present in by_dose_nc for this sheet
                doses = sorted(by_dose_nc[by_dose_nc["EnrollmentDate"] == sheet_name]["Dose"].unique()) if not by_dose_nc.empty else []
                # age pairs YoB1 in 1920..1980, YoB2=YoB1+10
                yob_all = sorted([y for y in end_data["YearOfBirth"].unique() if y > 0])
                for yob1 in range(1920, 1990, 10):
                    yob2 = yob1 + 10
                    if yob1 not in yob_all or yob2 not in yob_all:
                        continue
                    for dose in doses:
                        # Direct KCOR: CH(dose,yob1)/CH(dose,yob2), baseline-normalized at effective normalization week
                        ts = by_dose_nc[(by_dose_nc["EnrollmentDate"] == sheet_name) & (by_dose_nc["Dose"] == dose) & (by_dose_nc["YearOfBirth"].isin([yob1, yob2]))]
                        if ts["YearOfBirth"].nunique() != 2 or ts.empty:
                            continue
                        # Align by date and compute ratio
                        pivot = ts.pivot_table(index="DateDied", columns="YearOfBirth", values="CH")
                        pivot = pivot.dropna()
                        if pivot.empty or yob1 not in pivot.columns or yob2 not in pivot.columns:
                            continue
                        pivot = pivot.sort_index()
                        # Choose t0 index (KCOR_NORMALIZATION_WEEKS_EFFECTIVE from start)
                        t0_idx = min(KCOR_NORMALIZATION_WEEKS_EFFECTIVE, len(pivot) - 1) if len(pivot) > 0 else 0
                        k_raw = pivot[yob1] / pivot[yob2]
                        baseline = k_raw.iloc[t0_idx] if len(k_raw) > t0_idx and np.isfinite(k_raw.iloc[t0_idx]) and k_raw.iloc[t0_idx] > EPS else 1.0
                        k_series = k_raw / baseline
                        # pick KCOR at reporting date (~closest available)
                        # closest row to report_date
                        td = (pivot.index - report_date)
                        idx_closest = int(np.argmin(np.abs(td.to_numpy())))
                        k_val = float(k_series.iloc[idx_closest]) if len(k_series) > idx_closest else np.nan
                        neg_rows.append({
                            "EnrollmentDate": sheet_name,
                            "YoB1": yob1,
                            "YoB2": yob2,
                            "Dose": int(dose),
                            "KCOR": k_val
                        })
            neg_df = pd.DataFrame(neg_rows)
            # Write next to cohort outputs: ../data/Czech/negative_control_test_summary.xlsx
            czech_dir = os.path.dirname(out_path) or "."
            os.makedirs(czech_dir, exist_ok=True)
            out_file = os.path.join(czech_dir, "negative_control_test_summary.xlsx")
            with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
                neg_df.to_excel(writer, index=False, sheet_name="negative_control")
            dual_print(f"[NegativeControl] Wrote {len(neg_df)} rows to {out_file}")
            dual_print("[NegativeControl] Skipping normal workbook/summary outputs.")
            return combined
        except Exception as e:
            dual_print(f"[NegativeControl] Error creating summary: {e}")
            # In NC mode, do not proceed with normal outputs even on error
            return combined
    
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
    # Skip console output in MC mode (MC summary handles that with per-iteration tables)
    if not MONTE_CARLO_MODE:
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
            
            # Display ASMR weights for this sheet if available
            global ASMR_WEIGHTS_BY_SHEET
            weights_display = ASMR_WEIGHTS_BY_SHEET.get(sheet_name, {})
            if weights_display:
                dual_print(f"\nASMR Expected-Deaths Weights (for KCOR and KCOR_ns pooling):")
                for yob in sorted(weights_display.keys()):
                    dual_print(f"  Age {yob}: weight = {weights_display[yob]:.6f}")
                dual_print(f"  Total weight: {sum(weights_display.values()):.6f}")
                dual_print("")
            
            end_data = sheet_data_all[sheet_data_all["Date"] == report_date]
            
            # Get dose pairs for this specific sheet
            dose_pairs = get_dose_pairs(sheet_name)
            
            for (dose_num, dose_den) in dose_pairs:
                dual_print(f"\nDose combination: {dose_num} vs {dose_den} [{sheet_name}]")
                dual_print("-" * 50)
                dual_print(f"{'YoB':>15} | KCOR [95% CI] | {'KCOR_ns':>19} | {'*':>1} | {'ka_num':>9} {'kb_num':>9} {'t_n':>3} {'ka_den':>9} {'kb_den':>9} {'t_d':>3}")
                dual_print("-" * 50)
                
                # Get data for this dose combination and sheet at reporting date
                dose_data = end_data[
                    (end_data["Dose_num"] == dose_num) & 
                    (end_data["Dose_den"] == dose_den)
                ]
                
                if dose_data.empty:
                    dual_print("  No data available for this dose combination")
                    continue
                
                # Show results by age (including ASMR = 0, all ages = -2)
            # Sort ages: negative ages first (0, -2, -1), then positive ages
            def age_sort_key(age):
                if age == 0:
                    return (0, 0)  # ASMR first
                elif age == -2:
                    return (0, 1)  # All ages second
                elif age == -1:
                    return (0, 2)  # Unknown third
                else:
                    return (1, age)  # Regular ages after
            
            ages_sorted = sorted(dose_data["YearOfBirth"].unique(), key=age_sort_key)
            for age in ages_sorted:
                age_data = dose_data[dose_data["YearOfBirth"] == age]
                if not age_data.empty:
                    kcor_val = age_data["KCOR"].iloc[0]
                    ci_lower = age_data["CI_lower"].iloc[0]
                    ci_upper = age_data["CI_upper"].iloc[0]
                    # Check if abnormal fit flag is present
                    if "abnormal_fit" in age_data.columns:
                        abnormal_fit_flag = bool(age_data["abnormal_fit"].iloc[0])
                    else:
                        abnormal_fit_flag = False
                    abnormal_marker = "*" if abnormal_fit_flag else ""
                    # KCOR_ns: For ASMR (age == 0), compute from age-group KCOR_ns values using same weights as KCOR
                    # For other ages, get from the data directly
                    if age == 0:
                        # Compute ASMR KCOR_ns by pooling age-group KCOR_ns values
                        # Use the same expected-deaths weights as used for KCOR pooling
                        try:
                            # Get stored weights for this sheet (already declared global at function level)
                            weights_ns = ASMR_WEIGHTS_BY_SHEET.get(sheet_name, {})
                            
                            # Get all age groups for this dose combination at reporting date
                            age_groups_data = end_data[
                                (end_data["Dose_num"] == dose_num) & 
                                (end_data["Dose_den"] == dose_den) &
                                (end_data["YearOfBirth"] > 0)  # Exclude ASMR and special ages
                            ]
                            if not age_groups_data.empty and "KCOR_ns" in age_groups_data.columns:
                                # If weights are available, use them; otherwise fall back to equal weights
                                if not weights_ns:
                                    # Fallback to equal weights if weights not available
                                    age_list = list(age_groups_data["YearOfBirth"].unique())
                                    weights_ns = {yob: 1.0 / len(age_list) for yob in age_list}
                                
                                # Pool KCOR_ns values using log-space weighted average (same as KCOR)
                                logs_ns = []
                                wts_ns = []
                                for yob_check, group_data in age_groups_data.groupby("YearOfBirth"):
                                    kcor_ns_age = group_data["KCOR_ns"].iloc[0] if "KCOR_ns" in group_data.columns else np.nan
                                    if np.isfinite(kcor_ns_age) and kcor_ns_age > EPS:
                                        logs_ns.append(safe_log(kcor_ns_age))
                                        wts_ns.append(weights_ns.get(yob_check, 0.0))
                                
                                if len(logs_ns) > 0 and sum(wts_ns) > 0:
                                    logs_arr_ns = np.array(logs_ns)
                                    wts_arr_ns = np.array(wts_ns)
                                    logK_ns = np.average(logs_arr_ns, weights=wts_arr_ns)
                                    kcor_ns_val = float(safe_exp(logK_ns))
                                else:
                                    kcor_ns_val = np.nan
                            else:
                                kcor_ns_val = np.nan
                        except Exception:
                            kcor_ns_val = np.nan
                        kcor_ns_str = "-" if not (isinstance(kcor_ns_val, (int, float)) and np.isfinite(kcor_ns_val)) else f"{kcor_ns_val:.4f}"
                    else:
                        # For non-ASMR ages, get KCOR_ns from data
                        try:
                            kcor_ns_val = age_data.get("KCOR_ns", pd.Series([np.nan])).iloc[0]
                        except Exception:
                            kcor_ns_val = np.nan
                        kcor_ns_str = "-" if not (isinstance(kcor_ns_val, (int, float)) and np.isfinite(kcor_ns_val)) else f"{kcor_ns_val:.4f}"
                    
                    # Fetch Slope6 normalization parameters (dict with mode, a, b, c, t_mean, tau) for numerator and denominator cohorts
                    key_age = int(age) if pd.notna(age) and not isinstance(age, (int, np.integer)) else age
                    # Try multiple key formats to handle different storage formats
                    lookup_key_num = (sheet_name, key_age, int(dose_num))
                    params_num = slope6_params_map.get(lookup_key_num, None)
                    if params_num is None:
                        # Try alternative key formats
                        params_num = slope6_params_map.get((sheet_name, key_age, dose_num), None)
                    if params_num is None:
                        key_age_int = int(key_age) if isinstance(key_age, (int, float, np.number)) else key_age
                        params_num = slope6_params_map.get((sheet_name, key_age_int, int(dose_num)), None)
                    if params_num is None:
                        # Parameters not found - this shouldn't happen if slope8 was computed
                        params_num = {"mode": "none", "b": np.nan}
                    
                    lookup_key_den = (sheet_name, key_age, int(dose_den))
                    params_den = slope6_params_map.get(lookup_key_den, None)
                    if params_den is None:
                        params_den = slope6_params_map.get((sheet_name, key_age, dose_den), None)
                    if params_den is None:
                        key_age_int = int(key_age) if isinstance(key_age, (int, float, np.number)) else key_age
                        params_den = slope6_params_map.get((sheet_name, key_age_int, int(dose_den)), None)
                    if params_den is None:
                        params_den = {"mode": "none", "b": np.nan}
                    
                    # Extract parameters from dict for logging
                    # For ASMR (age == 0), force ka/kb/tau to None since it's aggregated
                    if age == 0:
                        ka_num = None
                        kb_num = None
                        tau_num = None
                        ka_den = None
                        kb_den = None
                        tau_den = None
                        mode_num = "none"
                        mode_den = "none"
                    elif isinstance(params_num, dict):
                        mode_num = params_num.get("mode", "none")
                        if mode_num == "slope8":
                            beta_num = params_num.get("b_original", np.nan)
                            C_num = params_num.get("C", np.nan)
                            # Extract ka, kb, tau for slope8 mode; handle None and invalid values
                            ka_raw = params_num.get("ka", None)
                            ka_num = ka_raw if (ka_raw is not None and np.isfinite(ka_raw)) else None
                            kb_raw = params_num.get("kb", None)
                            kb_num = kb_raw if (kb_raw is not None and np.isfinite(kb_raw)) else None
                            tau_raw = params_num.get("tau", None)
                            # Debug: Check if tau_raw is suspiciously 0.5 (SLOPE6_QUANTILE_TAU)
                            if tau_raw is not None and abs(tau_raw - 0.5) < 1e-10:
                                # This is suspicious - tau should never be exactly 0.5 (that's the quantile level, not the depletion timescale)
                                # Use None to force display as "---" instead of showing incorrect 0.5
                                tau_num = None
                            else:
                                tau_num = tau_raw if (tau_raw is not None and np.isfinite(tau_raw) and tau_raw > EPS) else None
                            c_num = 0.0  # Not used in slope8
                        else:
                            # For non-slope8 modes, don't extract ka/kb/tau (they're not valid)
                            beta_num = params_num.get("b_original", params_num.get("b", np.nan))
                            c_num = params_num.get("c", 0.0)
                            C_num = np.nan
                            ka_num = None  # Not valid for non-slope8
                            kb_num = None  # Not valid for non-slope8
                            tau_num = None  # Not valid for non-slope8 (don't use SLOPE6_QUANTILE_TAU)
                    elif params_num is not None and not isinstance(params_num, dict):
                        # Legacy format: params_num is a scalar (shouldn't happen with slope8, but handle gracefully)
                        try:
                            beta_num = float(params_num)
                            c_num = 0.0
                            mode_num = "linear"
                            C_num = np.nan
                            ka_num = np.nan
                            kb_num = np.nan
                            tau_num = np.nan
                        except (ValueError, TypeError):
                            beta_num = np.nan
                            c_num = 0.0
                            mode_num = "none"
                            C_num = np.nan
                            ka_num = np.nan
                            kb_num = np.nan
                            tau_num = np.nan
                    else:
                        # params_num is None or invalid
                        beta_num = np.nan
                        c_num = 0.0
                        mode_num = "none"
                        C_num = np.nan
                        ka_num = np.nan
                        kb_num = np.nan
                        tau_num = np.nan
                    
                    if age != 0 and isinstance(params_den, dict):
                        mode_den = params_den.get("mode", "none")
                        if mode_den == "slope8":
                            beta_den = params_den.get("b_original", np.nan)
                            C_den = params_den.get("C", np.nan)
                            # Extract ka, kb, tau for slope8 mode; handle None and invalid values
                            ka_raw = params_den.get("ka", None)
                            ka_den = ka_raw if (ka_raw is not None and np.isfinite(ka_raw)) else None
                            kb_raw = params_den.get("kb", None)
                            kb_den = kb_raw if (kb_raw is not None and np.isfinite(kb_raw)) else None
                            tau_raw = params_den.get("tau", None)
                            # Debug: Check if tau_raw is suspiciously 0.5 (SLOPE6_QUANTILE_TAU)
                            if tau_raw is not None and abs(tau_raw - 0.5) < 1e-10:
                                # This is suspicious - tau should never be exactly 0.5 (that's the quantile level, not the depletion timescale)
                                # Use None to force display as "---" instead of showing incorrect 0.5
                                tau_den = None
                            else:
                                tau_den = tau_raw if (tau_raw is not None and np.isfinite(tau_raw) and tau_raw > EPS) else None
                            c_den = 0.0  # Not used in slope8
                        else:
                            # For non-slope8 modes, don't extract ka/kb/tau (they're not valid)
                            beta_den = params_den.get("b_original", params_den.get("b", np.nan))
                            c_den = params_den.get("c", 0.0)
                            C_den = np.nan
                            ka_den = None  # Not valid for non-slope8
                            kb_den = None  # Not valid for non-slope8
                            tau_den = None  # Not valid for non-slope8 (don't use SLOPE6_QUANTILE_TAU)
                    elif params_den is not None and not isinstance(params_den, dict):
                        # Legacy format: params_den is a scalar (shouldn't happen with slope8, but handle gracefully)
                        try:
                            beta_den = float(params_den)
                            c_den = 0.0
                            mode_den = "linear"
                            C_den = np.nan
                            ka_den = np.nan
                            kb_den = np.nan
                            tau_den = np.nan
                        except (ValueError, TypeError):
                            beta_den = np.nan
                            c_den = 0.0
                            mode_den = "none"
                            C_den = np.nan
                            ka_den = np.nan
                            kb_den = np.nan
                            tau_den = np.nan
                    else:
                        # params_den is None or invalid
                        beta_den = np.nan
                        c_den = 0.0
                        mode_den = "none"
                        C_den = np.nan
                        ka_den = np.nan
                        kb_den = np.nan
                        tau_den = np.nan
                    
                    if age == 0:
                        age_label = "ASMR (direct)"
                    elif age == -2:
                        age_label = "All Ages"
                    elif age == -1:
                        age_label = "(unknown)"
                    else:
                        age_label = f"{age}"
                    
                    # Format ka, kb values for numerator and denominator (3 significant digits)
                    def format_param(val):
                        """Format parameter value with 3 significant digits, right-aligned in 9 characters."""
                        if val is None or not np.isfinite(val):
                            return "      ---"
                        # Use 3 significant digits with appropriate format
                        abs_val = abs(val)
                        if abs_val == 0.0:
                            return "      0.0"
                        elif abs_val < 0.001:
                            # Scientific notation for very small numbers
                            return f"{val:.2e}".rjust(9)
                        elif abs_val < 1.0:
                            # 3 decimal places for numbers < 1
                            return f"{val:.3f}".rjust(9)
                        elif abs_val < 10.0:
                            # 2 decimal places for numbers 1-10
                            return f"{val:.2f}".rjust(9)
                        elif abs_val < 100.0:
                            # 1 decimal place for numbers 10-100
                            return f"{val:.1f}".rjust(9)
                        else:
                            # Integer for numbers >= 100
                            return f"{val:.0f}".rjust(9)
                    
                    # Format tau values as integers (right-aligned in 3 characters)
                    def format_tau(val):
                        """Format tau value as integer, right-aligned in 3 characters."""
                        if val is None or not np.isfinite(val):
                            return "---"
                        # Format as integer
                        return f"{int(round(val)):>3}"
                    
                    # Show ka/kb/tau if they exist and are finite, only for slope8 mode
                    # For non-slope8 modes, ka/kb/tau should be None
                    ka_num_str = format_param(ka_num if (ka_num is not None and np.isfinite(ka_num)) else None)
                    kb_num_str = format_param(kb_num if (kb_num is not None and np.isfinite(kb_num)) else None)
                    tau_num_str = format_tau(tau_num if (tau_num is not None and np.isfinite(tau_num)) else None)
                    ka_den_str = format_param(ka_den if (ka_den is not None and np.isfinite(ka_den)) else None)
                    kb_den_str = format_param(kb_den if (kb_den is not None and np.isfinite(kb_den)) else None)
                    tau_den_str = format_tau(tau_den if (tau_den is not None and np.isfinite(tau_den)) else None)
                    
                    if age == 0:
                        dual_print(f"  {age_label:15} | {kcor_val:8.4f} [{ci_lower:.3f}, {ci_upper:.3f}] | {kcor_ns_str:>7} | {abnormal_marker:>1} | {ka_num_str} {kb_num_str} {tau_num_str} {ka_den_str} {kb_den_str} {tau_den_str}")
                    else:
                        # Just show the ka/kb/tau values, no parameter details in parentheses
                        dual_print(f"  {age_label:15} | {kcor_val:8.4f} [{ci_lower:.3f}, {ci_upper:.3f}] | {kcor_ns_str:>7} | {abnormal_marker:>1} | {ka_num_str} {kb_num_str} {tau_num_str} {ka_den_str} {kb_den_str} {tau_den_str}")

                # --- Print M/P KCOR summaries by decades when available ---
                try:
                    # Choose appropriate MFG sheet per cohort
                    mfg_sheet_map = {
                        "2021_24": (2, "MFG_MP_2021_24_D2_decades"),
                        "2022_06": (3, "MFG_MP_2022_06_D3_decades"),
                        "2022_47": (4, "MFG_MP_2022_47_D4_decades"),
                    }
                    if sheet_name in mfg_sheet_map:
                        dose_num, mfg_sheet_name = mfg_sheet_map[sheet_name]
                        # Read back the just-written sheet from tmp_path if possible; else from out_path after move
                        # Since we're in reporting before write, recompute quickly from combined data would be heavy.
                        # Instead, try to compute on the fly from CMR MFG sheet.
                        # We fallback to reading from input CMR workbook directly.
                        # Use the helper defined earlier to read raw CMR MFG sheet
                        def _read_cmr_mfg(base_sheet: str, dnum: int):
                            try:
                                src = pd.ExcelFile(src_path)
                                return pd.read_excel(src, sheet_name=f"{base_sheet}_MFG_D{dnum}")
                            except Exception:
                                return pd.DataFrame()
                        dfm = _read_cmr_mfg(sheet_name, dose_num)
                        if not dfm.empty:
                            # Bucket by decades and print last-available KCOR per decade
                            dfm["DateDied"] = pd.to_datetime(dfm["DateDied"], errors='coerce')
                            dfm = dfm[dfm["DateDied"].notna()]
                            dfm["YearOfBirth"] = pd.to_numeric(dfm["YearOfBirth"], errors='coerce').fillna(-1).astype(int)
                            dfm["Decade"] = (dfm["YearOfBirth"] // 10) * 10
                            # Aggregate
                            agg = dfm.groupby(["Decade","MFG","DateDied"]).agg({"Alive":"sum","Dead":"sum"}).reset_index()
                            agg = agg.sort_values(["Decade","MFG","DateDied"])\
                                     .reset_index(drop=True)
                            agg["PT"] = agg["Alive"].astype(float).clip(lower=0.0)
                            agg["MR"] = np.where(agg["PT"] > 0, agg["Dead"]/(agg["PT"] + EPS), np.nan)
                            agg["t"] = agg.groupby(["Decade","MFG"]).cumcount().astype(float)
                            agg["hazard_raw"] = hazard_from_mr_improved(agg["MR"].to_numpy())
                            agg["hazard_eff"] = np.where(agg["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), agg["hazard_raw"], 0.0)
                            agg["CH"] = agg.groupby(["Decade","MFG"]) ['hazard_eff'].cumsum()
                            dual_print(f"\nM/P by decades (Dose {dose_num})")
                            dual_print("-" * 50)
                            for dec, gdec in agg.groupby("Decade", sort=True):
                                gm = gdec[gdec["MFG"]=='M'][["DateDied","CH"]]
                                gp = gdec[gdec["MFG"]=='P'][["DateDied","CH"]]
                                merged = pd.merge(gm, gp, on="DateDied", suffixes=("_M","_P"), how="inner").sort_values("DateDied")
                                if merged.empty:
                                    continue
                                t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
                                valid = merged["CH_P"] > EPS
                                merged["K_raw_MP"] = np.where(valid, merged["CH_M"] / merged["CH_P"], np.nan)
                                k0 = merged["K_raw_MP"].iloc[t0_idx]
                                if not (np.isfinite(k0) and k0 > EPS):
                                    k0 = 1.0
                                merged["KCOR_MP"] = np.where(np.isfinite(merged["K_raw_MP"]), merged["K_raw_MP"] / k0, np.nan)
                                # pick closest to report_date
                                td = (merged["DateDied"] - report_date)
                                idx = int(np.nanargmin(np.abs(td.values.astype('timedelta64[D]').astype(float)))) if len(merged)>0 else -1
                                if idx >= 0:
                                    val = float(merged["KCOR_MP"].iloc[idx])
                                    dual_print(f"  {int(dec)}: M/P KCOR = {val:.4f}")
                except Exception as _e_print_mfg:
                    dual_print(f"[WARN] Failed to print M/P summaries: {_e_print_mfg}")
        
        dual_print("="*80)
    # End of console output section (suppressed in MC mode)

    # write (standard mode handled later with retry block)

    # Write main analysis file with retry logic
    max_retries = 3
    retry_count = 0
    
    if not _is_sa_mode():
        if MONTE_CARLO_MODE:
            # MC mode: write one sheet per iteration
            while retry_count < max_retries:
                try:
                    tmp_dir = "/tmp" if os.path.isdir("/tmp") else os.path.dirname(out_path) or "."
                    base_no_ext, _ = os.path.splitext(os.path.basename(out_path))
                    tmp_base = base_no_ext + ".tmp.xlsx"
                    tmp_path = os.path.join(tmp_dir, tmp_base)
                    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                        # Write each iteration as a separate sheet
                        for i, iteration_sheet in enumerate(sheets_to_process):
                            iteration_data = all_out[i] if i < len(all_out) else pd.DataFrame()
                            if not iteration_data.empty:
                                # Drop deprecated columns
                                drop_cols = [
                                    "MR_adj_num","MR_adj_den",
                                    "CH_actual_num","CH_actual_den",
                                    "slope_num","slope_den",
                                    "scale_factor_num","scale_factor_den",
                                    "MR_smooth_num","MR_smooth_den"
                                ]
                                iteration_data_clean = iteration_data.copy()
                                for c in drop_cols:
                                    if c in iteration_data_clean.columns:
                                        iteration_data_clean.drop(columns=[c], inplace=True)
                                iteration_data_clean["Date"] = iteration_data_clean["Date"].apply(lambda x: x.date() if hasattr(x, 'date') else x)
                                # Use iteration number as sheet name
                                sheet_name = str(iteration_sheet)
                                iteration_data_clean.to_excel(writer, index=False, sheet_name=sheet_name)
                                dual_print(f"[MC] Wrote iteration {sheet_name} to sheet '{sheet_name}' ({len(iteration_data_clean)} rows)")
                    # Move temp file to final location
                    shutil.move(tmp_path, out_path)
                    dual_print(f"[MC] Wrote Monte Carlo output to {out_path}")
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        dual_print(f"❌ Failed to write MC output after {max_retries} retries: {e}")
                        raise
                    import time
                    time.sleep(0.5)
            
            # Create MC summary
            if mc_summary_data:
                create_mc_summary(mc_summary_data, dual_print)
        else:
            # Normal mode: write combined output
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
                        # FINAL_KCOR fields removed
                        "  KCOR_NORMALIZATION_WEEKS",
                        "  SLOPE_ANCHOR_T",
                        # "  SLOPE_WINDOW_SIZE",  # obsolete
                        # removed MA params
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
                        f"   - t0 = baseline week (effective normalization week = {KCOR_NORMALIZATION_WEEKS + DYNAMIC_HVE_SKIP_WEEKS})",
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
                        "    - MR_num: Weekly mortality rate for numerator (annualized per 100,000)",
                        "    - CH_num: Cumulative hazard for numerator (unitless)",
                        "    - CH_actual_num: Cumulative sum of unadjusted weekly MR for numerator (unitless)",
                        "    - hazard_num: Discrete hazard for numerator (unitless)",
                        "    - slope_num: Estimated slope used for numerator",
                        "    - scale_factor_num: exp(-slope*(t - t0)) used for numerator",
                        "    - MR_smooth_num: Smoothed MR for numerator (annualized per 100,000)",
                        "    - t_num: Week index from enrollment for numerator",
                        "    - MR_den: Weekly mortality rate for denominator (annualized per 100,000)",
                        "    - MR_den: Weekly mortality rate for denominator (annualized per 100,000)",
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
                        "",
                        "",
                        f"{KCOR_NORMALIZATION_WEEKS}",
                        f"{SLOPE_ANCHOR_T}",
                        f"",
                        # removed MA params
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
                    # Drop deprecated/unused columns from dose_pairs before writing
                    drop_cols = [
                        "MR_adj_num","MR_adj_den",
                        "CH_actual_num","CH_actual_den",
                        "slope_num","slope_den",
                        "scale_factor_num","scale_factor_den",
                        "MR_smooth_num","MR_smooth_den"
                    ]
                    all_data = combined.copy()
                    for c in drop_cols:
                        if c in all_data.columns:
                            all_data.drop(columns=[c], inplace=True)
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
                        "",
                        "",
                        f"{KCOR_NORMALIZATION_WEEKS}",
                        f"{SLOPE_ANCHOR_T}",
                        f"",
                        # removed MA params
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
                    # Always write by_dose sheet (trim to used columns)
                    by_dose_cols = [
                        "EnrollmentDate","Date","YearOfBirth","Dose","ISOweek",
                        "Dead","Alive","MR","Hazard","Hazard_adj","Cum_deaths","Cumu_Person_Time","Time_Index"
                    ]
                    debug_trim = debug_df[[c for c in by_dose_cols if c in debug_df.columns]].copy()
                    debug_trim.to_excel(writer, index=False, sheet_name="by_dose")

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

                    # --- Manufacturer KCOR (Moderna vs Pfizer) comparisons ---
                    # Read auxiliary MFG sheets from input workbook and compute M/P KCOR by cohort
                    try:
                        def _read_mfg_sheet(base_sheet_name: str, dose_num: int) -> pd.DataFrame:
                            sheet = f"{base_sheet_name}_MFG_D{dose_num}"
                            try:
                                return pd.read_excel(src_path, sheet_name=sheet)
                            except Exception:
                                return pd.DataFrame()

                        def _compute_kcor_mp(df_base: pd.DataFrame, enrollment: str, dose_num: int, by_decade: bool) -> pd.DataFrame:
                            if df_base.empty:
                                return pd.DataFrame()
                            dfb = df_base.copy()
                            # Normalize types
                            dfb["DateDied"] = pd.to_datetime(dfb["DateDied"], errors='coerce')
                            dfb = dfb[dfb["DateDied"].notna()]
                            dfb["YearOfBirth"] = pd.to_numeric(dfb["YearOfBirth"], errors='coerce')
                            dfb = dfb[dfb["YearOfBirth"].notna()]
                            dfb["YearOfBirth"] = dfb["YearOfBirth"].astype(int)
                            # Grouping key for ages
                            if by_decade:
                                dfb["YearOfBirth"] = (dfb["YearOfBirth"] // 10) * 10
                            else:
                                dfb["YearOfBirth"] = 0
                            # Aggregate across Sex and DCCI within age bucket
                            grouped = (
                                dfb.groupby(["YearOfBirth","MFG","DateDied"], sort=False)
                                   .agg({"ISOweekDied":"first","Alive":"sum","Dead":"sum"})
                                   .reset_index()
                            )
                            grouped = grouped.sort_values(["YearOfBirth","MFG","DateDied"])\
                                             .reset_index(drop=True)
                            # Compute MR and hazards
                            grouped["PT"] = grouped["Alive"].astype(float).clip(lower=0.0)
                            grouped["Dead"] = grouped["Dead"].astype(float).clip(lower=0.0)
                            grouped["MR"] = np.where(grouped["PT"] > 0, grouped["Dead"]/(grouped["PT"] + EPS), np.nan)
                            grouped["t"] = grouped.groupby(["YearOfBirth","MFG"]).cumcount().astype(float)
                            grouped["hazard_raw"] = hazard_from_mr_improved(grouped["MR"].to_numpy())
                            grouped["hazard_eff"] = np.where(grouped["t"] >= float(DYNAMIC_HVE_SKIP_WEEKS), grouped["hazard_raw"], 0.0)
                            grouped["CH"] = grouped.groupby(["YearOfBirth","MFG"]) ['hazard_eff'].cumsum()
                            # Build M/P KCOR per age bucket
                            out_rows = []
                            for yob, g_age in grouped.groupby("YearOfBirth", sort=False):
                                gm = g_age[g_age["MFG"] == 'M'][["DateDied","ISOweekDied","CH"]].drop_duplicates("DateDied")
                                gp = g_age[g_age["MFG"] == 'P'][["DateDied","ISOweekDied","CH"]].drop_duplicates("DateDied")
                                if gm.empty or gp.empty:
                                    continue
                                merged = pd.merge(gm, gp, on="DateDied", suffixes=("_M","_P"), how="inner").sort_values("DateDied")
                                if merged.empty:
                                    continue
                                valid = merged["CH_P"] > EPS
                                merged["K_raw_MP"] = np.where(valid, merged["CH_M"] / merged["CH_P"], np.nan)
                                t0_idx = KCOR_NORMALIZATION_WEEKS_EFFECTIVE if len(merged) > KCOR_NORMALIZATION_WEEKS_EFFECTIVE else 0
                                k0 = merged["K_raw_MP"].iloc[t0_idx]
                                if not (np.isfinite(k0) and k0 > EPS):
                                    k0 = 1.0
                                merged["KCOR_MP"] = np.where(np.isfinite(merged["K_raw_MP"]), merged["K_raw_MP"] / k0, np.nan)
                                out = merged[["DateDied","ISOweekDied_M","KCOR_MP"]].copy()
                                out.rename(columns={"ISOweekDied_M":"ISOweekDied","DateDied":"Date"}, inplace=True)
                                out["EnrollmentDate"] = enrollment
                                out["YearOfBirth"] = int(yob)
                                out["Dose_num"] = dose_num
                                out["Dose_den"] = dose_num
                                out["MFG_num"] = 'M'
                                out["MFG_den"] = 'P'
                                out["Date"] = pd.to_datetime(out["Date"]).apply(lambda x: x.date())
                                out_rows.append(out)
                            if out_rows:
                                cols = ["EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den","MFG_num","MFG_den","KCOR_MP"]
                                return pd.concat(out_rows, ignore_index=True)[cols]
                            return pd.DataFrame(columns=["EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den","MFG_num","MFG_den","KCOR_MP"])

                        wrote_any_mfg = False
                        # 2021_24: Dose 2, decades
                        df_mfg_2124 = _read_mfg_sheet("2021_24", 2)
                        mp_2124 = _compute_kcor_mp(df_mfg_2124, "2021_24", 2, by_decade=True)
                        if not mp_2124.empty:
                            mp_2124.to_excel(writer, index=False, sheet_name="MFG_MP_2021_24_D2_decades")
                            dual_print("[MFG] Wrote M/P KCOR: 2021_24 Dose 2 by decades (", len(mp_2124), "rows)")
                            wrote_any_mfg = True
                        # 2022_06: Dose 3, decades
                        df_mfg_2206 = _read_mfg_sheet("2022_06", 3)
                        mp_2206 = _compute_kcor_mp(df_mfg_2206, "2022_06", 3, by_decade=True)
                        if not mp_2206.empty:
                            mp_2206.to_excel(writer, index=False, sheet_name="MFG_MP_2022_06_D3_decades")
                            dual_print("[MFG] Wrote M/P KCOR: 2022_06 Dose 3 by decades (", len(mp_2206), "rows)")
                            wrote_any_mfg = True
                        # 2022_47: Dose 4, decades
                        df_mfg_2247 = _read_mfg_sheet("2022_47", 4)
                        mp_2247 = _compute_kcor_mp(df_mfg_2247, "2022_47", 4, by_decade=True)
                        if not mp_2247.empty:
                            mp_2247.to_excel(writer, index=False, sheet_name="MFG_MP_2022_47_D4_decades")
                            dual_print("[MFG] Wrote M/P KCOR: 2022_47 Dose 4 by decades (", len(mp_2247), "rows)")
                            wrote_any_mfg = True
                        if not wrote_any_mfg:
                            dual_print("[MFG] No M/P manufacturer comparisons generated (missing or empty MFG sheets)")
                    except Exception as _e_mfg:
                        dual_print(f"[WARN] Failed to add MFG M/P comparisons: {_e_mfg}")
                    
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
                    dual_print(f"[Done] Wrote {len(combined)} rows to {out_path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
            # removed param_final_kcor_min grouping
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
                # Offsets optional; dynamic anchors removed
                if pd.isna(off1) or pd.isna(slope_len):
                            off1, slope_len = (None, None)
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
                    "param_normalization_week": KCOR_NORMALIZATION_WEEKS,
                    # removed MA params
                    # "param_slope_window_size" obsolete with slope2; keep empty for backward-compat headers
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
        # Standard summary (skip console output in MC mode, MC summary handles that)
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
                        ]
                        
                        # Sort ages: negative ages first (0, -2, -1), then positive ages
                        def age_sort_key(age):
                            if age == 0:
                                return (0, 0)  # ASMR first
                            elif age == -2:
                                return (0, 1)  # All ages second
                            elif age == -1:
                                return (0, 2)  # Unknown third
                            else:
                                return (1, age)  # Regular ages after
                        
                        dose_data = dose_data.copy()
                        dose_data["_sort_key"] = dose_data["YearOfBirth"].apply(age_sort_key)
                        dose_data = dose_data.sort_values("_sort_key").drop(columns=["_sort_key"])
                        
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
                                age_label = "ASMR (direct)"
                            elif age == -2:
                                age_label = "All Ages"
                            elif age == -1:
                                age_label = "(unknown)"
                            else:
                                age_label = f"{age}"
                            
                            # Check if abnormal fit flag is present
                            abnormal_fit_flag = row.get("abnormal_fit", False) if "abnormal_fit" in row.index else False
                            abnormal_marker = "*" if abnormal_fit_flag else ""
                            
                            summary_rows.append({
                                "Dose_Combination": "",
                                "YearOfBirth": age_label,
                                "KCOR": f"{row['KCOR']:.4f}{abnormal_marker}",
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

def create_mc_summary(mc_summary_data, dual_print):
    """Create Monte Carlo summary statistics at end of 2022.
    
    Args:
        mc_summary_data: List of dicts with keys: Iteration, Dose_num, Dose_den, YearOfBirth, Date, KCOR, CI_lower, CI_upper
        dual_print: Function to print to both console and log file
    """
    if not mc_summary_data:
        dual_print("[MC Summary] No Monte Carlo data available for summary")
        return
    
    dual_print("\n" + "="*80)
    dual_print("MONTE CARLO SUMMARY - KCOR VALUES AT END OF 2022")
    dual_print("="*80)
    
    df_mc = pd.DataFrame(mc_summary_data)
    
    # Filter to end of 2022 (2022-12-31 or closest date)
    df_mc["Date"] = pd.to_datetime(df_mc["Date"])
    target_date = pd.to_datetime("2022-12-31")
    
    # Get dose pairs
    dose_pairs = get_dose_pairs("2022_06")
    
    # Store processed data for each dose combination (for statistics stage)
    dose_combo_stats = {}
    
    # STAGE 1: Print per-iteration tables for each dose combination
    for dose_num, dose_den in dose_pairs:
        dual_print(f"\nDose combination: {dose_num} vs {dose_den}")
        dual_print("-" * 60)
        
        # Filter to this dose pair
        pair_data = df_mc[
            (df_mc["Dose_num"] == dose_num) & 
            (df_mc["Dose_den"] == dose_den)
        ]
        
        if pair_data.empty:
            dual_print("  No data available for this dose combination")
            continue
        
        # Filter to YearOfBirth=-2 (all ages) if available, otherwise use all ages
        all_ages_data = pair_data[pair_data["YearOfBirth"] == -2]
        if all_ages_data.empty:
            # Fallback to any available YearOfBirth
            all_ages_data = pair_data
        
        if all_ages_data.empty:
            dual_print("  No data available for this dose combination")
            continue
        
        # Extract KCOR values (remove NaN) and sort by iteration
        all_ages_data_clean = all_ages_data[all_ages_data["KCOR"].notna()].copy()
        
        if len(all_ages_data_clean) == 0:
            dual_print("  No valid KCOR values available")
            continue
        
        # Group by iteration and take one KCOR value per iteration (deduplicate)
        # Since all rows for the same iteration should have the same KCOR value,
        # we can just take the first one per iteration
        all_ages_data_unique = all_ages_data_clean.groupby("Iteration").first().reset_index()
        all_ages_data_unique = all_ages_data_unique.sort_values("Iteration")
        
        # Store KCOR values for statistics stage
        kcor_values = all_ages_data_unique["KCOR"]
        dose_combo_stats[(dose_num, dose_den)] = {
            "kcor_values": kcor_values,
            "num_iterations": len(all_ages_data_unique)
        }
        
        # Print per-iteration table (show KCOR values for each iteration)
        dual_print(f"{'Iteration':>12} {'KCOR':>12}")
        dual_print("-" * 25)
        for _, row in all_ages_data_unique.iterrows():
            iteration = row["Iteration"]
            kcor_val = row["KCOR"]
            dual_print(f"{iteration:>12} {kcor_val:>12.4f}")
    
    # STAGE 2: Print summary statistics for all dose combinations
    dual_print("\n" + "="*80)
    dual_print("SUMMARY STATISTICS")
    dual_print("="*80)
    
    for dose_num, dose_den in dose_pairs:
        if (dose_num, dose_den) not in dose_combo_stats:
            continue
        
        stats = dose_combo_stats[(dose_num, dose_den)]
        kcor_values = stats["kcor_values"]
        num_iterations = stats["num_iterations"]
        
        dual_print(f"\nDose combination: {dose_num} vs {dose_den}")
        dual_print("-" * 60)
        
        # Compute statistics
        mean_kcor = kcor_values.mean()
        median_kcor = kcor_values.median()
        std_kcor = kcor_values.std()
        min_kcor = kcor_values.min()
        max_kcor = kcor_values.max()
        
        # Percentiles for 95% CI equivalent
        p2_5 = kcor_values.quantile(0.025)
        p97_5 = kcor_values.quantile(0.975)
        p25 = kcor_values.quantile(0.25)
        p75 = kcor_values.quantile(0.75)
        
        dual_print(f"  Iterations: {num_iterations}")
        dual_print(f"  Mean KCOR:   {mean_kcor:.4f}")
        dual_print(f"  Median KCOR: {median_kcor:.4f}")
        dual_print(f"  Std Dev:     {std_kcor:.4f}")
        dual_print(f"  Min:         {min_kcor:.4f}")
        dual_print(f"  Max:         {max_kcor:.4f}")
        dual_print(f"  2.5th %ile:  {p2_5:.4f}")
        dual_print(f"  25th %ile:   {p25:.4f}")
        dual_print(f"  75th %ile:   {p75:.4f}")
        dual_print(f"  97.5th %ile: {p97_5:.4f}")
        dual_print(f"  95% Range:   [{p2_5:.4f}, {p97_5:.4f}]")
    
    dual_print("\n" + "="*80)

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

#!/usr/bin/env python3
"""
Slope Normalization Test Script

This script applies three slope normalization methods from KCOR.py to booster_d0_slope.csv:
1. Linear median quantile regression (fit_linear_median)
2. TRF bounded depletion-mode normalization (fit_slope7_depletion)
3. LM unbounded depletion-mode normalization (fit_slope7_depletion_lm)

Outputs an Excel file with adjusted hazard values and fitted parameters.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import least_squares
import warnings
from pathlib import Path
from datetime import datetime

# Constants from KCOR.py
EPS = 1e-12
SLOPE6_MIN_DATA_POINTS = 5
SLOPE6_QUANTILE_TAU = 0.5

# MIN_DELTA_K and MIN_TAU for TRF method
MIN_DELTA_K = 0.0
MIN_TAU = 1e-3

# Fit window from KCOR.py (baseline_window)
FIT_WINDOW_START = "2022-01"
FIT_WINDOW_END = "2024-12"


def _iso_to_date_slope6(isostr: str):
    """Convert ISO week string (YYYY-WW) to datetime (Monday of that week)."""
    y, w = isostr.split("-")
    return datetime.fromisocalendar(int(y), int(w), 1)


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
    where k_∞ = k_0 + Δk, ensuring k_∞ ≥ k_0 (monotonicity constraint).
    
    Parameters
    ----------
    s : array-like
        Time values in weeks since enrollment (NOT centered, s=0 at enrollment).
    logh : array-like
        log(hazard) values aligned with s.
    
    Returns
    -------
    C, k_inf, k_0, tau : floats
        Fitted parameters:
        - C: intercept
        - k_inf (kb): long-run background slope = k_0 + Δk (may be negative, but k_inf ≥ k_0)
        - k_0 (ka): slope at enrollment (may be negative)
        - tau: depletion timescale in weeks (must be > 0)
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
    
    # Δk: difference between long-run and initial slope (must be ≥ 0)
    # Estimate from later points vs initial slope
    if len(logh_valid) >= 5:
        later_slope = (logh_valid[-1] - logh_valid[-3]) / (s_valid[-1] - s_valid[-3] + EPS)
        delta_k_init = max(later_slope - k_0_init, 0.001)  # Ensure Δk ≥ 0 and well above bound
    else:
        # If we can't estimate later slope, assume small positive Δk
        delta_k_init = 0.001
    
    # tau: depletion timescale (weeks), initial guess based on data span
    tau_init = max((s_valid.max() - s_valid.min()) / 3.0, max(MIN_TAU * 10, 1.0))  # Ensure well above bound
    
    # Store initial estimates for return
    initial_params = (float(C_init), float(k_0_init), float(delta_k_init), float(tau_init))
    
    # Parameter vector: [C, k_0, Δk, tau]
    p0 = np.array([C_init, k_0_init, delta_k_init, tau_init], dtype=float)
    
    # Bounds: Δk ≥ 0, tau > MIN_TAU, C and k_0 unbounded (use ±inf for "no bound")
    lower_bounds = np.array([-np.inf, -np.inf, MIN_DELTA_K, MIN_TAU], dtype=float)
    upper_bounds = np.array([ np.inf,  np.inf,        np.inf,    np.inf], dtype=float)
    
    # Ensure initial guess is within bounds before calling least_squares
    p0 = np.clip(p0, lower_bounds, upper_bounds)
    bounds = (lower_bounds, upper_bounds)
    
    def residual_func(p):
        """Residual function for least squares."""
        C, k_0, delta_k, tau = p
        # Model: log h(s) = C + (k_0 + Δk)*s - Δk*τ*(1 - exp(-s/τ))
        # This is equivalent to: C + k_∞*s + (k_0 - k_∞)*τ*(1 - exp(-s/τ)) where k_∞ = k_0 + Δk
        k_inf = k_0 + delta_k
        predicted = C + k_inf * s_valid - delta_k * tau * (1.0 - np.exp(-s_valid / (tau + EPS)))
        return logh_valid - predicted
    
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
            # Ensure constraints are satisfied
            delta_k = max(delta_k, MIN_DELTA_K)  # Ensure Δk ≥ 0
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
            warnings.warn(f"TRF fit did not converge: {result.message}", RuntimeWarning)
            return (np.nan, np.nan, np.nan, np.nan), initial_params
    except Exception as e:
        # Log the exception for debugging
        warnings.warn(f"fit_slope7_depletion failed: {str(e)}", RuntimeWarning)
        return (np.nan, np.nan, np.nan, np.nan), initial_params


def fit_slope7_depletion_lm(s, logh):
    """
    Comparator fit: depletion-mode normalization using unbounded Levenberg–Marquardt (method='lm').
    
    Uses the same functional form as fit_slope7_depletion, but WITHOUT bounds and with method='lm'
    for nonlinear least squares. Intended for diagnostics/comparison only.
    
    Returns
    -------
    C, k_inf, k_0, tau : floats
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
        delta_k_init = max(later_slope - k_0_init, 0.001)
    else:
        delta_k_init = 0.001
    
    tau_init = max((s_valid.max() - s_valid.min()) / 3.0, 1.0)
    
    # Store initial estimates for return
    initial_params = (float(C_init), float(k_0_init), float(delta_k_init), float(tau_init))
    
    p0 = np.array([C_init, k_0_init, delta_k_init, tau_init], dtype=float)
    
    def residual_func(p):
        C, k_0, delta_k, tau = p
        k_inf = k_0 + delta_k
        predicted = C + k_inf * s_valid - delta_k * tau * (1.0 - np.exp(-s_valid / (tau + EPS)))
        return logh_valid - predicted
    
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
            k_inf = k_0 + delta_k
            if np.isfinite(C) and np.isfinite(k_inf) and np.isfinite(k_0) and np.isfinite(tau):
                return (float(C), float(k_inf), float(k_0), float(tau)), initial_params
            else:
                return (np.nan, np.nan, np.nan, np.nan), initial_params
        else:
            warnings.warn(f"LM fit did not converge: {result.message}", RuntimeWarning)
            return (np.nan, np.nan, np.nan, np.nan), initial_params
    except Exception as e:
        warnings.warn(f"fit_slope7_depletion_lm failed: {str(e)}", RuntimeWarning)
        return (np.nan, np.nan, np.nan, np.nan), initial_params


def main():
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Read input CSV
    input_file = script_dir / "booster_d0_slope.csv"
    print(f"Reading input file: {input_file}")
    # Try different encodings in case file is not UTF-8
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Successfully read CSV with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise RuntimeError(f"Could not read CSV file with any of the attempted encodings: {encodings}")
    
    # Clean up the dataframe - remove rows with empty dates or parameter notes
    df = df[df['date'].notna()]
    df = df[df['date'].str.strip() != '']
    # Remove rows that look like parameter notes (contain "=" or "parameters")
    df = df[~df['date'].str.contains('=', na=False)]
    df = df[~df['date'].str.contains('parameters', case=False, na=False)]
    df = df[~df['date'].str.contains('Here are', case=False, na=False)]
    
    # NOTE: KCOR.py iterates through fit_weeks (2022-01 to 2024-12) but reports 115 points
    # which matches the CSV total. This suggests KCOR might be using all available data
    # or the CSV was created from data already filtered to the fit window.
    # For now, we'll use all data points to match KCOR's reported count of 115.
    # If KCOR is actually filtering, we'd expect 111 points (fit window: 2022-01 to 2024-12,
    # but CSV starts at 2022-06, so 111 weeks from 2022-06 to 2024-12).
    
    print(f"Using all data points from CSV (matching KCOR's reported count of 115)")
    
    # Extract columns
    dates = df['date'].values
    hazard_raw = pd.to_numeric(df['h(t) raw'], errors='coerce').values
    hazard_adj_existing = pd.to_numeric(df['h(t) adj'], errors='coerce').values
    
    # Create sequential time index starting from 0 (matching KCOR's t values which are cumcount)
    t = np.arange(len(hazard_raw), dtype=float)
    
    # Filter out invalid values
    valid_mask = np.isfinite(hazard_raw) & (hazard_raw > 0)
    t_valid = t[valid_mask]
    hazard_raw_valid = hazard_raw[valid_mask]
    logh_valid = np.log(hazard_raw_valid)
    
    print(f"Processing {len(t_valid)} data points")
    print(f"Time range: {t_valid.min()} to {t_valid.max()}")
    print(f"Hazard range: {hazard_raw_valid.min():.6e} to {hazard_raw_valid.max():.6e}")
    
    # Method 1: Linear median quantile regression
    print("\n=== Method 1: Linear Median Quantile Regression ===")
    try:
        a_lin, b_lin, t_mean = fit_linear_median(t_valid, logh_valid, tau=SLOPE6_QUANTILE_TAU)
        print(f"Fitted parameters: a_lin={a_lin:.6e}, b_lin={b_lin:.6e}, t_mean={t_mean:.6f}")
        
        # Apply normalization: h_adj = h_raw * exp(-b_lin * (t - t_mean))
        t_c = t - t_mean
        h_adj_linear = hazard_raw * np.exp(-b_lin * t_c)
    except Exception as e:
        print(f"ERROR: Linear fit failed: {e}")
        a_lin, b_lin, t_mean = np.nan, np.nan, np.nan
        h_adj_linear = np.full_like(hazard_raw, np.nan)
    
    # Method 2: TRF Depletion-mode normalization
    print("\n=== Method 2: TRF Depletion-mode (Bounded) ===")
    try:
        (C_trf, kb_trf, ka_trf, tau_trf), (C_init_trf, ka_init_trf, delta_k_init_trf, tau_init_trf) = fit_slope7_depletion(t_valid, logh_valid)
        print(f"Initial estimates: C={C_init_trf:.6e}, k_0={ka_init_trf:.6e}, Δk={delta_k_init_trf:.6e}, tau={tau_init_trf:.6f}")
        if np.isfinite(C_trf):
            print(f"Fitted parameters: C={C_trf:.6e}, k_inf={kb_trf:.6e}, k_0={ka_trf:.6e}, tau={tau_trf:.6f}")
            # Apply normalization: h_adj = h_raw * exp(-C - kb*t - (ka - kb)*tau*(1 - exp(-t/tau)))
            h_adj_trf = hazard_raw * np.exp(-C_trf - kb_trf * t - (ka_trf - kb_trf) * tau_trf * (1.0 - np.exp(-t / (tau_trf + EPS))))
        else:
            print("ERROR: TRF fit failed (returned NaN)")
            h_adj_trf = np.full_like(hazard_raw, np.nan)
    except Exception as e:
        print(f"ERROR: TRF fit failed: {e}")
        C_trf, kb_trf, ka_trf, tau_trf = np.nan, np.nan, np.nan, np.nan
        C_init_trf, ka_init_trf, delta_k_init_trf, tau_init_trf = np.nan, np.nan, np.nan, np.nan
        h_adj_trf = np.full_like(hazard_raw, np.nan)
    
    # Method 3: LM Depletion-mode normalization
    print("\n=== Method 3: LM Depletion-mode (Unbounded) ===")
    try:
        (C_lm, kb_lm, ka_lm, tau_lm), (C_init_lm, ka_init_lm, delta_k_init_lm, tau_init_lm) = fit_slope7_depletion_lm(t_valid, logh_valid)
        print(f"Initial estimates: C={C_init_lm:.6e}, k_0={ka_init_lm:.6e}, Δk={delta_k_init_lm:.6e}, tau={tau_init_lm:.6f}")
        if np.isfinite(C_lm):
            print(f"Fitted parameters: C={C_lm:.6e}, k_inf={kb_lm:.6e}, k_0={ka_lm:.6e}, tau={tau_lm:.6f}")
            # Apply normalization: h_adj = h_raw * exp(-C - kb*t - (ka - kb)*tau*(1 - exp(-t/tau)))
            h_adj_lm = hazard_raw * np.exp(-C_lm - kb_lm * t - (ka_lm - kb_lm) * tau_lm * (1.0 - np.exp(-t / (tau_lm + EPS))))
        else:
            print("ERROR: LM fit failed (returned NaN)")
            h_adj_lm = np.full_like(hazard_raw, np.nan)
    except Exception as e:
        print(f"ERROR: LM fit failed: {e}")
        C_lm, kb_lm, ka_lm, tau_lm = np.nan, np.nan, np.nan, np.nan
        C_init_lm, ka_init_lm, delta_k_init_lm, tau_init_lm = np.nan, np.nan, np.nan, np.nan
        h_adj_lm = np.full_like(hazard_raw, np.nan)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'date': dates,
        'h(t) raw': hazard_raw,
        'h(t) adj': hazard_adj_existing,
        'h_adj_linear': h_adj_linear,
        'h_adj_trf': h_adj_trf,
        'h_adj_lm': h_adj_lm
    })
    
    # Create parameters DataFrame
    param_rows = []
    
    # Linear method parameters
    param_rows.append({'method': 'Linear', 'parameter': 'a_lin', 'value': a_lin, 'description': 'Intercept from quantile regression'})
    param_rows.append({'method': 'Linear', 'parameter': 'b_lin', 'value': b_lin, 'description': 'Slope from quantile regression'})
    param_rows.append({'method': 'Linear', 'parameter': 't_mean', 'value': t_mean, 'description': 'Time centering constant (mean of t)'})
    
    # TRF method parameters
    param_rows.append({'method': 'TRF', 'parameter': 'C', 'value': C_trf, 'description': 'Intercept (fitted)'})
    param_rows.append({'method': 'TRF', 'parameter': 'k_inf (kb)', 'value': kb_trf, 'description': 'Long-run background slope (fitted)'})
    param_rows.append({'method': 'TRF', 'parameter': 'k_0 (ka)', 'value': ka_trf, 'description': 'Slope at enrollment (fitted)'})
    param_rows.append({'method': 'TRF', 'parameter': 'tau', 'value': tau_trf, 'description': 'Depletion timescale in weeks (fitted)'})
    param_rows.append({'method': 'TRF', 'parameter': 'C_init', 'value': C_init_trf, 'description': 'Initial estimate: mean of first 3 log(hazard) values'})
    param_rows.append({'method': 'TRF', 'parameter': 'k_0_init', 'value': ka_init_trf, 'description': 'Initial estimate: slope from first two points'})
    param_rows.append({'method': 'TRF', 'parameter': 'delta_k_init', 'value': delta_k_init_trf, 'description': 'Initial estimate: max(later_slope - k_0_init, 0.001)'})
    param_rows.append({'method': 'TRF', 'parameter': 'tau_init', 'value': tau_init_trf, 'description': 'Initial estimate: (max(t) - min(t)) / 3.0'})
    param_rows.append({'method': 'TRF', 'parameter': 'bounds', 'value': 'Δk ≥ 0, tau > 1e-3', 'description': 'Bounds enforced (C and k_0 unbounded)'})
    param_rows.append({'method': 'TRF', 'parameter': 'max_nfev', 'value': 1000, 'description': 'Maximum function evaluations'})
    param_rows.append({'method': 'TRF', 'parameter': 'ftol', 'value': 1e-8, 'description': 'Function tolerance'})
    param_rows.append({'method': 'TRF', 'parameter': 'xtol', 'value': 1e-8, 'description': 'Parameter tolerance'})
    
    # LM method parameters
    param_rows.append({'method': 'LM', 'parameter': 'C', 'value': C_lm, 'description': 'Intercept (fitted)'})
    param_rows.append({'method': 'LM', 'parameter': 'k_inf (kb)', 'value': kb_lm, 'description': 'Long-run background slope (fitted)'})
    param_rows.append({'method': 'LM', 'parameter': 'k_0 (ka)', 'value': ka_lm, 'description': 'Slope at enrollment (fitted)'})
    param_rows.append({'method': 'LM', 'parameter': 'tau', 'value': tau_lm, 'description': 'Depletion timescale in weeks (fitted)'})
    param_rows.append({'method': 'LM', 'parameter': 'C_init', 'value': C_init_lm, 'description': 'Initial estimate: mean of first 3 log(hazard) values'})
    param_rows.append({'method': 'LM', 'parameter': 'k_0_init', 'value': ka_init_lm, 'description': 'Initial estimate: slope from first two points'})
    param_rows.append({'method': 'LM', 'parameter': 'delta_k_init', 'value': delta_k_init_lm, 'description': 'Initial estimate: max(later_slope - k_0_init, 0.001)'})
    param_rows.append({'method': 'LM', 'parameter': 'tau_init', 'value': tau_init_lm, 'description': 'Initial estimate: (max(t) - min(t)) / 3.0'})
    param_rows.append({'method': 'LM', 'parameter': 'bounds', 'value': 'None', 'description': 'No bounds (unbounded optimization)'})
    param_rows.append({'method': 'LM', 'parameter': 'max_nfev', 'value': 1000, 'description': 'Maximum function evaluations'})
    param_rows.append({'method': 'LM', 'parameter': 'ftol', 'value': 1e-8, 'description': 'Function tolerance'})
    param_rows.append({'method': 'LM', 'parameter': 'xtol', 'value': 1e-8, 'description': 'Parameter tolerance'})
    
    param_df = pd.DataFrame(param_rows)
    
    # Write to Excel
    output_file = script_dir / "slope_normalization_test_output.xlsx"
    print(f"\nWriting output to: {output_file}")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        output_df.to_excel(writer, sheet_name='data', index=False)
        param_df.to_excel(writer, sheet_name='parameters', index=False)
    
    print(f"Successfully wrote {len(output_df)} rows to 'data' sheet")
    print(f"Successfully wrote {len(param_df)} parameter rows to 'parameters' sheet")
    print(f"\nOutput file: {output_file}")


if __name__ == "__main__":
    main()


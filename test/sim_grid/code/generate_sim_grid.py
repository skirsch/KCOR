#!/usr/bin/env python3
"""
KCOR Simulation Grid Generator

Generates a prespecified simulation grid demonstrating KCOR's operating
characteristics and failure-mode diagnostics for the Statistics in Medicine
methods paper.

Six scenarios are implemented:
1. Gamma-frailty null (strong selection) - different θ values, no effect
2. Injected hazard increase - r=1.2 during effect window
3. Injected hazard decrease - r=0.8 during effect window
4. Non-gamma frailty null - lognormal frailty, no effect
5. Quiet-window contamination - external shock during quiet window
6. Sparse-events regime - small cohorts, low baseline hazard

All times are in event-time units (weeks since cohort entry).

Usage:
    python generate_sim_grid.py --output-data DATA.xlsx --output-results RESULTS.xlsx --output-diagnostics DIAG.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import quad

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SimConfig:
    """Configuration for simulation grid."""
    # Time parameters (event-time units: weeks since cohort entry)
    n_weeks: int = 120
    quiet_window_start: int = 20
    quiet_window_end: int = 80
    effect_window_start: int = 20
    effect_window_end: int = 80
    skip_weeks: int = 2  # Early weeks to skip for KCOR accumulation
    
    # Default cohort parameters
    n_initial: int = 100_000
    baseline_hazard: float = 0.002  # Weekly baseline hazard rate k
    
    # Frailty parameters for gamma-frailty scenarios
    theta_A: float = 1.0  # High frailty variance (strong depletion)
    theta_B: float = 0.3  # Lower frailty variance (weaker depletion)
    
    # Effect multipliers
    hazard_increase_r: float = 1.2
    hazard_decrease_r: float = 0.70  # Stronger decrease to ensure clear detection
    
    # Sparse regime parameters
    sparse_n_initial: int = 1_000
    sparse_baseline_hazard: float = 0.0005
    
    # Non-gamma frailty (lognormal) parameters
    lognormal_sigma: float = 0.8  # σ for lognormal (mean=1)
    
    # Contamination shock parameters
    shock_start: int = 30
    shock_end: int = 50
    shock_multiplier: float = 2.0
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Enrollment date for KCOR_CMR format
    enrollment_date: str = "2021-06-14"


# ============================================================================
# Gamma-Frailty Model Functions
# ============================================================================

def gamma_frailty_hazard(t: np.ndarray, k: float, theta: float) -> np.ndarray:
    """
    Compute cohort-level hazard under gamma-frailty model.
    
    h(t) = k / (1 + theta * k * t)
    
    Args:
        t: Time since cohort entry (weeks)
        k: Baseline hazard rate
        theta: Frailty variance (selection strength)
    
    Returns:
        Cohort-level hazard at each time point
    """
    t_arr = np.asarray(t, dtype=float)
    if theta <= 1e-10:
        return k * np.ones_like(t_arr)
    return k / (1 + theta * k * t_arr)


def gamma_frailty_cumhaz(t: np.ndarray, k: float, theta: float) -> np.ndarray:
    """
    Compute observed cumulative hazard under gamma-frailty model.
    
    H_obs(t) = (1/theta) * log(1 + theta * k * t)
    
    With theta -> 0 limit: H(t) = k * t
    """
    t_arr = np.asarray(t, dtype=float)
    if theta <= 1e-10:
        return k * t_arr
    return (1.0 / theta) * np.log1p(theta * k * t_arr)


def invert_gamma_frailty(H_obs: np.ndarray, theta: float) -> np.ndarray:
    """
    Invert gamma-frailty cumulative hazard to baseline cumulative hazard.
    
    H0(t) = (exp(theta * H_obs(t)) - 1) / theta
    
    With theta -> 0 limit: H0(t) = H_obs(t)
    """
    H_arr = np.asarray(H_obs, dtype=float)
    if theta <= 1e-10:
        return H_arr.copy()
    return np.expm1(theta * H_arr) / theta


def H_model(t: np.ndarray, k: float, theta: float) -> np.ndarray:
    """Model-predicted cumulative hazard (same as gamma_frailty_cumhaz)."""
    return gamma_frailty_cumhaz(t, k, theta)


def fit_k_theta_cumhaz(
    t: np.ndarray, 
    H_obs: np.ndarray, 
    k0: Optional[float] = None, 
    theta0: float = 0.1
) -> Tuple[Tuple[float, float], Dict]:
    """
    Fit gamma-frailty parameters (k, theta) from observed cumulative hazard.
    
    Uses nonlinear least squares in cumulative-hazard space.
    
    Args:
        t: Time points (weeks since enrollment)
        H_obs: Observed cumulative hazard values
        k0: Initial guess for k (if None, estimated from data)
        theta0: Initial guess for theta
    
    Returns:
        ((k_hat, theta_hat), diagnostics_dict)
    """
    EPS = 1e-12
    
    t_arr = np.asarray(t, dtype=float)
    H_arr = np.asarray(H_obs, dtype=float)
    
    # Filter to valid points
    mask = np.isfinite(t_arr) & np.isfinite(H_arr)
    t_arr = t_arr[mask]
    H_arr = H_arr[mask]
    n_obs = len(t_arr)
    
    if n_obs < 2:
        return (np.nan, np.nan), {
            "success": False,
            "n_obs": n_obs,
            "rmse_Hobs": np.nan,
            "status": None,
            "message": "insufficient_points",
        }
    
    # Initial guess for k from slope
    if k0 is None:
        try:
            k0_est = float(np.polyfit(t_arr, H_arr, 1)[0])
        except Exception:
            dt = np.diff(t_arr)
            dH = np.diff(H_arr)
            k0_est = float(np.nanmean(dH / np.where(np.abs(dt) > EPS, dt, np.nan)))
        if not np.isfinite(k0_est):
            k0_est = 1e-8
        k0 = max(k0_est, 1e-12)
    else:
        k0 = max(float(k0), 1e-12)
    
    theta0 = max(float(theta0), 0.0)
    
    def _residuals(params):
        k, theta = params
        if (k <= 0.0) or (theta < 0.0) or (not np.isfinite(k)) or (not np.isfinite(theta)):
            return np.ones_like(H_arr) * 1e6
        return H_model(t_arr, k, theta) - H_arr
    
    try:
        res = least_squares(
            _residuals,
            x0=[k0, theta0],
            bounds=([1e-15, 0.0], [1.0, 50.0]),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            max_nfev=2000,
        )
        k_hat, theta_hat = res.x
        
        # Compute RMSE
        H_pred = H_model(t_arr, k_hat, theta_hat)
        rmse = np.sqrt(np.mean((H_arr - H_pred) ** 2))
        
        return (k_hat, theta_hat), {
            "success": res.success,
            "n_obs": n_obs,
            "rmse_Hobs": rmse,
            "status": res.status,
            "message": res.message,
        }
    except Exception as e:
        return (np.nan, np.nan), {
            "success": False,
            "n_obs": n_obs,
            "rmse_Hobs": np.nan,
            "status": None,
            "message": str(e),
        }


# ============================================================================
# Non-Gamma Frailty (Lognormal)
# ============================================================================

def lognormal_cohort_survival(t: float, k: float, sigma: float) -> float:
    """
    Compute cohort survival under lognormal frailty.
    
    Frailty z ~ LogNormal(mu, sigma^2) with E[z] = 1.
    This requires mu = -sigma^2/2 for mean=1.
    
    S(t) = E[exp(-z * k * t)] = integral of exp(-z*k*t) * f_Z(z) dz
    
    Uses numerical integration.
    """
    if t <= 0:
        return 1.0
    
    # For mean=1: mu = -sigma^2 / 2
    mu = -sigma**2 / 2
    
    def integrand(z):
        if z <= 0:
            return 0.0
        # Lognormal PDF
        pdf = np.exp(-0.5 * ((np.log(z) - mu) / sigma)**2) / (z * sigma * np.sqrt(2 * np.pi))
        # Survival contribution
        return np.exp(-z * k * t) * pdf
    
    result, _ = quad(integrand, 0, 50, limit=100)
    return max(result, 1e-15)


def simulate_lognormal_frailty_cohort(
    config: SimConfig,
    n_initial: int,
    k: float,
    sigma: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate cohort mortality under lognormal frailty using Monte Carlo.
    
    Returns:
        weeks, alive, dead arrays
    """
    np.random.seed(seed)
    
    weeks = np.arange(config.n_weeks)
    alive = np.zeros(config.n_weeks, dtype=float)
    dead = np.zeros(config.n_weeks, dtype=float)
    
    # For mean=1: mu = -sigma^2 / 2
    mu = -sigma**2 / 2
    
    # Draw frailty for each individual
    frailties = np.random.lognormal(mean=mu, sigma=sigma, size=n_initial)
    
    # Track which individuals are alive
    is_alive = np.ones(n_initial, dtype=bool)
    
    for t in range(config.n_weeks):
        n_alive = np.sum(is_alive)
        alive[t] = n_alive
        
        if n_alive == 0:
            dead[t] = 0
            continue
        
        # Individual hazards: h_i(t) = z_i * k
        # Probability of death in this interval
        individual_hazards = frailties[is_alive] * k
        p_death = 1 - np.exp(-individual_hazards)
        
        # Determine deaths
        deaths_mask = np.random.random(n_alive) < p_death
        n_deaths = np.sum(deaths_mask)
        
        dead[t] = n_deaths
        
        # Update alive status
        alive_indices = np.where(is_alive)[0]
        is_alive[alive_indices[deaths_mask]] = False
    
    return weeks, alive, dead


# ============================================================================
# Simulation Scenarios
# ============================================================================

def simulate_gamma_frailty_cohort(
    config: SimConfig,
    n_initial: int,
    k: float,
    theta: float,
    hazard_multiplier: float = 1.0,
    effect_start: Optional[int] = None,
    effect_end: Optional[int] = None,
    external_shock: bool = False,
    shock_start: Optional[int] = None,
    shock_end: Optional[int] = None,
    shock_mult: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate cohort mortality under gamma-frailty model.
    
    Args:
        config: Simulation configuration
        n_initial: Initial cohort size
        k: Baseline hazard rate
        theta: Frailty variance
        hazard_multiplier: Multiplier for injected effect (1.0 = no effect)
        effect_start: Start week of effect window (None = no effect)
        effect_end: End week of effect window
        external_shock: Whether to apply external shock
        shock_start: Start week of shock
        shock_end: End week of shock
        shock_mult: Shock hazard multiplier
    
    Returns:
        weeks, alive, dead arrays
    """
    weeks = np.arange(config.n_weeks)
    alive = np.zeros(config.n_weeks, dtype=float)
    dead = np.zeros(config.n_weeks, dtype=float)
    
    current_alive = float(n_initial)
    
    for t in range(config.n_weeks):
        alive[t] = current_alive
        
        if current_alive <= 0:
            dead[t] = 0
            continue
        
        # Base hazard from gamma-frailty model
        h = gamma_frailty_hazard(np.array([t]), k, theta)[0]
        
        # Apply injected effect if in effect window
        if effect_start is not None and effect_end is not None:
            if effect_start <= t < effect_end:
                h *= hazard_multiplier
        
        # Apply external shock if applicable
        if external_shock and shock_start is not None and shock_end is not None:
            if shock_start <= t < shock_end:
                h *= shock_mult
        
        # Convert hazard to probability
        p_death = 1 - np.exp(-h)
        
        # Compute deaths (deterministic for reproducibility)
        deaths = current_alive * p_death
        
        dead[t] = deaths
        current_alive -= deaths
    
    return weeks, alive, dead


def run_scenario_1_gamma_null(config: SimConfig) -> Dict:
    """
    Scenario 1: Gamma-frailty null (strong selection).
    
    Two cohorts with different theta values, no treatment effect.
    Expected: KCOR ≈ 1.0
    """
    # Cohort A: high frailty variance (Dose 0)
    weeks_A, alive_A, dead_A = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, config.theta_A
    )
    
    # Cohort B: lower frailty variance (Dose 1)
    weeks_B, alive_B, dead_B = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, config.theta_B
    )
    
    return {
        "scenario": "gamma_null",
        "label": "Gamma-Frailty Null",
        "cohorts": [
            {"dose": 0, "weeks": weeks_A, "alive": alive_A, "dead": dead_A, "theta_true": config.theta_A},
            {"dose": 1, "weeks": weeks_B, "alive": alive_B, "dead": dead_B, "theta_true": config.theta_B},
        ],
        "expected_kcor": "≈ 1.0",
    }


def run_scenario_2_hazard_increase(config: SimConfig) -> Dict:
    """
    Scenario 2: Injected hazard increase (r=1.2).
    
    Both cohorts have same theta (no confounding from differential frailty).
    Dose 1 has hazard multiplied by 1.2 during the effect window.
    Expected: KCOR > 1.05
    """
    # Use same theta for both cohorts to isolate the effect
    theta_shared = config.theta_B  # Lower frailty for cleaner detection
    
    # Cohort A: control (no effect)
    weeks_A, alive_A, dead_A = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, theta_shared
    )
    
    # Cohort B: treatment (hazard increase)
    weeks_B, alive_B, dead_B = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, theta_shared,
        hazard_multiplier=config.hazard_increase_r,
        effect_start=config.effect_window_start,
        effect_end=config.effect_window_end,
    )
    
    return {
        "scenario": "hazard_increase",
        "label": "Injected Hazard Increase",
        "cohorts": [
            {"dose": 0, "weeks": weeks_A, "alive": alive_A, "dead": dead_A, "theta_true": theta_shared},
            {"dose": 1, "weeks": weeks_B, "alive": alive_B, "dead": dead_B, "theta_true": theta_shared},
        ],
        "expected_kcor": "> 1.05",
        "effect_window": (config.effect_window_start, config.effect_window_end),
        "hazard_multiplier": config.hazard_increase_r,
    }


def run_scenario_3_hazard_decrease(config: SimConfig) -> Dict:
    """
    Scenario 3: Injected hazard decrease (r=0.70).
    
    Both cohorts have same theta (no confounding from differential frailty).
    Expected: KCOR < 0.95
    """
    # Use same theta for both cohorts to isolate the effect
    theta_shared = config.theta_B  # Lower frailty for cleaner detection
    
    # Cohort A: control (no effect)
    weeks_A, alive_A, dead_A = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, theta_shared
    )
    
    # Cohort B: treatment (hazard decrease)
    weeks_B, alive_B, dead_B = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, theta_shared,
        hazard_multiplier=config.hazard_decrease_r,
        effect_start=config.effect_window_start,
        effect_end=config.effect_window_end,
    )
    
    return {
        "scenario": "hazard_decrease",
        "label": "Injected Hazard Decrease",
        "cohorts": [
            {"dose": 0, "weeks": weeks_A, "alive": alive_A, "dead": dead_A, "theta_true": theta_shared},
            {"dose": 1, "weeks": weeks_B, "alive": alive_B, "dead": dead_B, "theta_true": theta_shared},
        ],
        "expected_kcor": "< 0.95",
        "effect_window": (config.effect_window_start, config.effect_window_end),
        "hazard_multiplier": config.hazard_decrease_r,
    }


def run_scenario_4_nongamma_null(config: SimConfig) -> Dict:
    """
    Scenario 4: Non-gamma frailty null (lognormal).
    
    Frailty from lognormal distribution, no treatment effect.
    Expected: Degraded fit diagnostics, KCOR may deviate slightly.
    """
    # Cohort A: lognormal frailty (higher variance)
    weeks_A, alive_A, dead_A = simulate_lognormal_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, 
        config.lognormal_sigma, seed=config.seed
    )
    
    # Cohort B: lognormal frailty (lower variance)
    weeks_B, alive_B, dead_B = simulate_lognormal_frailty_cohort(
        config, config.n_initial, config.baseline_hazard,
        config.lognormal_sigma * 0.5, seed=config.seed + 100
    )
    
    return {
        "scenario": "nongamma_null",
        "label": "Non-Gamma Frailty (Lognormal)",
        "cohorts": [
            {"dose": 0, "weeks": weeks_A, "alive": alive_A, "dead": dead_A, "theta_true": np.nan},
            {"dose": 1, "weeks": weeks_B, "alive": alive_B, "dead": dead_B, "theta_true": np.nan},
        ],
        "expected_kcor": "Degraded fit",
    }


def run_scenario_5_contamination(config: SimConfig) -> Dict:
    """
    Scenario 5: Quiet-window contamination.
    
    External shock affects BOTH cohorts during weeks 30-50 (overlapping quiet window).
    Expected: Poor fit diagnostics due to non-frailty curvature.
    """
    # Cohort A: with external shock
    weeks_A, alive_A, dead_A = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, config.theta_A,
        external_shock=True,
        shock_start=config.shock_start,
        shock_end=config.shock_end,
        shock_mult=config.shock_multiplier,
    )
    
    # Cohort B: with same external shock
    weeks_B, alive_B, dead_B = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, config.theta_B,
        external_shock=True,
        shock_start=config.shock_start,
        shock_end=config.shock_end,
        shock_mult=config.shock_multiplier,
    )
    
    return {
        "scenario": "contamination",
        "label": "Quiet-Window Contamination",
        "cohorts": [
            {"dose": 0, "weeks": weeks_A, "alive": alive_A, "dead": dead_A, "theta_true": config.theta_A},
            {"dose": 1, "weeks": weeks_B, "alive": alive_B, "dead": dead_B, "theta_true": config.theta_B},
        ],
        "expected_kcor": "Poor diagnostics",
        "shock_window": (config.shock_start, config.shock_end),
    }


def run_scenario_6_sparse(config: SimConfig) -> Dict:
    """
    Scenario 6: Sparse-events regime.
    
    Small cohorts with low baseline hazard, no treatment effect.
    Expected: Weak identifiability, noisy estimates.
    """
    # Cohort A: small cohort, low hazard
    weeks_A, alive_A, dead_A = simulate_gamma_frailty_cohort(
        config, config.sparse_n_initial, config.sparse_baseline_hazard, config.theta_A
    )
    
    # Cohort B: small cohort, low hazard
    weeks_B, alive_B, dead_B = simulate_gamma_frailty_cohort(
        config, config.sparse_n_initial, config.sparse_baseline_hazard, config.theta_B
    )
    
    return {
        "scenario": "sparse",
        "label": "Sparse Events",
        "cohorts": [
            {"dose": 0, "weeks": weeks_A, "alive": alive_A, "dead": dead_A, "theta_true": config.theta_A},
            {"dose": 1, "weeks": weeks_B, "alive": alive_B, "dead": dead_B, "theta_true": config.theta_B},
        ],
        "expected_kcor": "Weak identifiability",
        "n_initial": config.sparse_n_initial,
    }


# ============================================================================
# KCOR Processing and Diagnostics
# ============================================================================

def compute_kcor_for_scenario(
    scenario_data: Dict,
    config: SimConfig,
) -> Dict:
    """
    Compute KCOR and diagnostics for a scenario.
    
    Returns:
        Dictionary with KCOR trajectory and diagnostic metrics
    """
    results = {
        "scenario": scenario_data["scenario"],
        "label": scenario_data["label"],
        "cohort_results": [],
        "kcor_trajectory": None,
    }
    
    # Process each cohort
    H0_trajectories = {}
    for cohort in scenario_data["cohorts"]:
        dose = cohort["dose"]
        weeks = cohort["weeks"]
        alive = cohort["alive"]
        dead = cohort["dead"]
        
        # Compute hazard from mortality rate
        MR = np.where(alive > 0, dead / alive, 0)
        MR = np.clip(MR, 0, 0.99)  # Safety clip
        hazard = -np.log(1 - MR)
        hazard = np.nan_to_num(hazard, nan=0, posinf=0, neginf=0)
        
        # Apply skip-week rule and compute cumulative hazard
        h_eff = np.where(weeks >= config.skip_weeks, hazard, 0)
        H_obs = np.cumsum(h_eff)
        
        # Extract quiet-window data for fitting
        quiet_mask = (weeks >= config.quiet_window_start) & (weeks <= config.quiet_window_end)
        t_quiet = weeks[quiet_mask].astype(float)
        H_quiet = H_obs[quiet_mask]
        
        # Fit gamma-frailty model
        (k_hat, theta_hat), fit_diag = fit_k_theta_cumhaz(t_quiet, H_quiet)
        
        # Invert to get depletion-neutralized cumulative hazard
        if np.isfinite(theta_hat) and theta_hat >= 0:
            H0 = invert_gamma_frailty(H_obs, theta_hat)
        else:
            H0 = H_obs.copy()
            theta_hat = 0.0
        
        # Compute post-normalization linearity (R² from linear fit)
        valid_mask = (weeks >= config.skip_weeks) & (weeks <= config.quiet_window_end)
        t_valid = weeks[valid_mask].astype(float)
        H0_valid = H0[valid_mask]
        
        if len(t_valid) > 2:
            # Linear regression
            coeffs = np.polyfit(t_valid, H0_valid, 1)
            H0_pred = np.polyval(coeffs, t_valid)
            ss_res = np.sum((H0_valid - H0_pred) ** 2)
            ss_tot = np.sum((H0_valid - np.mean(H0_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r_squared = np.nan
        
        cohort_result = {
            "dose": dose,
            "k_hat": k_hat,
            "theta_hat": theta_hat,
            "theta_true": cohort.get("theta_true", np.nan),
            "rmse": fit_diag.get("rmse_Hobs", np.nan),
            "r_squared": r_squared,
            "n_obs": fit_diag.get("n_obs", 0),
            "fit_success": fit_diag.get("success", False),
            "H_obs": H_obs,
            "H0": H0,
        }
        results["cohort_results"].append(cohort_result)
        H0_trajectories[dose] = H0
    
    # Compute KCOR trajectory (Dose 1 / Dose 0)
    if 0 in H0_trajectories and 1 in H0_trajectories:
        H0_num = H0_trajectories[1]  # Treatment
        H0_den = H0_trajectories[0]  # Control
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            kcor_raw = np.where(H0_den > 1e-10, H0_num / H0_den, np.nan)
        
        # Normalize to 1 at baseline (skip_weeks + normalization buffer)
        norm_week = config.skip_weeks + 4
        if norm_week < len(kcor_raw) and np.isfinite(kcor_raw[norm_week]) and kcor_raw[norm_week] > 0:
            kcor = kcor_raw / kcor_raw[norm_week]
        else:
            kcor = kcor_raw
        
        results["kcor_trajectory"] = kcor
        
        # Compute summary statistics
        diagnostic_window = (weeks >= 20) & (weeks <= 100)
        kcor_diagnostic = kcor[diagnostic_window]
        kcor_diagnostic = kcor_diagnostic[np.isfinite(kcor_diagnostic)]
        
        if len(kcor_diagnostic) > 0:
            results["kcor_median"] = float(np.median(kcor_diagnostic))
            results["kcor_mean"] = float(np.mean(kcor_diagnostic))
            results["kcor_min"] = float(np.min(kcor_diagnostic))
            results["kcor_max"] = float(np.max(kcor_diagnostic))
        else:
            results["kcor_median"] = np.nan
            results["kcor_mean"] = np.nan
            results["kcor_min"] = np.nan
            results["kcor_max"] = np.nan
    
    return results


# ============================================================================
# Output Generation
# ============================================================================

def generate_kcor_cmr_format(
    scenario_data: Dict,
    config: SimConfig,
) -> pd.DataFrame:
    """
    Convert scenario data to KCOR_CMR format.
    
    Columns: ISOweekDied, DateDied, YearOfBirth, Sex, Dose, Alive, Dead
    """
    rows = []
    
    # Generate ISO weeks and dates from enrollment
    start_date = datetime.strptime(config.enrollment_date, "%Y-%m-%d")
    
    for cohort in scenario_data["cohorts"]:
        dose = cohort["dose"]
        weeks = cohort["weeks"]
        alive = cohort["alive"]
        dead = cohort["dead"]
        
        for t in range(len(weeks)):
            date = start_date + timedelta(weeks=int(t))
            iso_year, iso_week, _ = date.isocalendar()
            
            rows.append({
                "ISOweekDied": f"{iso_year}-{iso_week:02d}",
                "DateDied": date.strftime("%Y-%m-%d"),
                "YearOfBirth": 1950,  # Fixed for simulation
                "Sex": "F",
                "Dose": dose,
                "Alive": int(alive[t]),
                "Dead": int(dead[t]),
            })
    
    return pd.DataFrame(rows)


def save_results(
    all_results: List[Dict],
    config: SimConfig,
    output_data: str,
    output_results: str,
    output_diagnostics: str,
):
    """Save all results to files."""
    
    # Create output directories
    for path in [output_data, output_results, output_diagnostics]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save KCOR_CMR format data
    with pd.ExcelWriter(output_data, engine='xlsxwriter') as writer:
        for result in all_results:
            scenario = result["scenario"]
            # Find original scenario data
            df = result.get("cmr_data")
            if df is not None:
                df.to_excel(writer, sheet_name=scenario[:31], index=False)
    
    # Save KCOR results
    with pd.ExcelWriter(output_results, engine='xlsxwriter') as writer:
        for result in all_results:
            scenario = result["scenario"]
            
            # KCOR trajectory
            weeks = np.arange(config.n_weeks)
            kcor = result.get("kcor_trajectory", np.full(config.n_weeks, np.nan))
            
            df_kcor = pd.DataFrame({
                "Week": weeks,
                "KCOR": kcor,
            })
            df_kcor.to_excel(writer, sheet_name=f"{scenario[:25]}_kcor", index=False)
            
            # Cohort summaries
            cohort_rows = []
            for cr in result.get("cohort_results", []):
                cohort_rows.append({
                    "Dose": cr["dose"],
                    "k_hat": cr["k_hat"],
                    "theta_hat": cr["theta_hat"],
                    "theta_true": cr["theta_true"],
                    "RMSE": cr["rmse"],
                    "R_squared": cr["r_squared"],
                    "n_obs": cr["n_obs"],
                    "fit_success": cr["fit_success"],
                })
            df_cohort = pd.DataFrame(cohort_rows)
            df_cohort.to_excel(writer, sheet_name=f"{scenario[:25]}_fit", index=False)
    
    # Save diagnostics CSV
    diag_rows = []
    for result in all_results:
        scenario = result["scenario"]
        label = result.get("label", scenario)
        
        for cr in result.get("cohort_results", []):
            diag_rows.append({
                "scenario": scenario,
                "label": label,
                "dose": cr["dose"],
                "k_hat": cr["k_hat"],
                "theta_hat": cr["theta_hat"],
                "theta_true": cr["theta_true"],
                "rmse": cr["rmse"],
                "r_squared": cr["r_squared"],
                "n_obs": cr["n_obs"],
                "fit_success": cr["fit_success"],
                "kcor_median": result.get("kcor_median", np.nan),
                "kcor_mean": result.get("kcor_mean", np.nan),
            })
    
    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(output_diagnostics, index=False)
    
    print(f"Saved data to: {output_data}")
    print(f"Saved results to: {output_results}")
    print(f"Saved diagnostics to: {output_diagnostics}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate KCOR simulation grid for methods paper"
    )
    parser.add_argument(
        "--output-data", "-d",
        default="test/sim_grid/data/sim_grid_data.xlsx",
        help="Output path for KCOR_CMR format data"
    )
    parser.add_argument(
        "--output-results", "-r",
        default="test/sim_grid/out/sim_grid_results.xlsx",
        help="Output path for KCOR results"
    )
    parser.add_argument(
        "--output-diagnostics", "-g",
        default="test/sim_grid/out/sim_grid_diagnostics.csv",
        help="Output path for diagnostics CSV"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = SimConfig(seed=args.seed)
    
    print("=" * 60)
    print("KCOR Simulation Grid Generator")
    print("=" * 60)
    print(f"Time horizon: {config.n_weeks} weeks")
    print(f"Quiet window: weeks {config.quiet_window_start}-{config.quiet_window_end}")
    print(f"Effect window: weeks {config.effect_window_start}-{config.effect_window_end}")
    print(f"Random seed: {config.seed}")
    print()
    
    # Run all scenarios
    scenarios = [
        ("Scenario 1: Gamma-Frailty Null", run_scenario_1_gamma_null),
        ("Scenario 2: Injected Hazard Increase", run_scenario_2_hazard_increase),
        ("Scenario 3: Injected Hazard Decrease", run_scenario_3_hazard_decrease),
        ("Scenario 4: Non-Gamma Frailty", run_scenario_4_nongamma_null),
        ("Scenario 5: Quiet-Window Contamination", run_scenario_5_contamination),
        ("Scenario 6: Sparse Events", run_scenario_6_sparse),
    ]
    
    all_results = []
    
    for name, scenario_fn in scenarios:
        print(f"Running {name}...")
        scenario_data = scenario_fn(config)
        
        # Generate KCOR_CMR format
        cmr_df = generate_kcor_cmr_format(scenario_data, config)
        
        # Compute KCOR and diagnostics
        result = compute_kcor_for_scenario(scenario_data, config)
        result["cmr_data"] = cmr_df
        
        # Print summary
        print(f"  KCOR median (weeks 20-100): {result.get('kcor_median', np.nan):.4f}")
        for cr in result.get("cohort_results", []):
            print(f"  Dose {cr['dose']}: θ̂={cr['theta_hat']:.4f}, RMSE={cr['rmse']:.6f}, R²={cr['r_squared']:.4f}")
        print()
        
        all_results.append(result)
    
    # Save all results
    save_results(
        all_results,
        config,
        args.output_data,
        args.output_results,
        args.output_diagnostics,
    )
    
    # Print acceptance criteria check
    print("=" * 60)
    print("Acceptance Criteria Check")
    print("=" * 60)
    
    # Check null scenarios (1, 4, 5, 6)
    null_scenarios = ["gamma_null", "nongamma_null", "contamination", "sparse"]
    for result in all_results:
        if result["scenario"] in null_scenarios:
            median = result.get("kcor_median", np.nan)
            if np.isfinite(median):
                in_range = 0.95 <= median <= 1.05
                status = "✓ PASS" if in_range else "✗ FAIL"
                print(f"{result['label']}: median KCOR = {median:.4f} {status}")
            else:
                print(f"{result['label']}: median KCOR = N/A")
    
    # Check effect scenarios (2, 3)
    for result in all_results:
        if result["scenario"] == "hazard_increase":
            median = result.get("kcor_median", np.nan)
            if np.isfinite(median):
                passes = median > 1.05
                status = "✓ PASS" if passes else "✗ FAIL"
                print(f"{result['label']}: median KCOR = {median:.4f} (expected > 1.05) {status}")
        elif result["scenario"] == "hazard_decrease":
            median = result.get("kcor_median", np.nan)
            if np.isfinite(median):
                passes = median < 0.95
                status = "✓ PASS" if passes else "✗ FAIL"
                print(f"{result['label']}: median KCOR = {median:.4f} (expected < 0.95) {status}")
    
    print()
    print("Simulation grid generation complete!")


if __name__ == "__main__":
    main()


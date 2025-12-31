#!/usr/bin/env python3
"""
Cox Bias Demonstration Simulation

This script demonstrates that Cox proportional hazards regression produces spurious
hazard ratios under frailty heterogeneity even when the true treatment effect is zero.

Two cohorts are simulated with identical baseline hazards and no treatment effect;
cohorts differ only in gamma frailty variance (θ). Despite the true hazard ratio
being 1 by construction, Cox regression produces increasingly non-null hazard ratios
as θ increases, while KCOR remains centered near unity.

Usage:
    python generate_cox_bias_sim.py --output-results RESULTS.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CoxBiasConfig:
    """Configuration for Cox bias demonstration simulation."""
    n_per_cohort: int = 200_000  # Reduce to 50_000 for faster runs
    baseline_rate: float = 0.002  # per week
    censor_horizon: int = 52  # weeks
    seed: int = 123
    post_start_week: int = 8  # Week after which to compute KCOR slope


# ============================================================================
# Utilities: Nelson-Aalen H(t)
# ============================================================================

def nelson_aalen(time: np.ndarray, event: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Nelson-Aalen cumulative hazard estimator on a specified time grid.
    
    Args:
        time: observed time (event or censor)
        event: 1 if event, 0 if censored
        t_grid: increasing time points to evaluate H(t)
    
    Returns:
        H_hat at each t_grid
    """
    # Sort by time
    order = np.argsort(time)
    time_s = time[order]
    event_s = event[order].astype(int)

    # Unique event times
    uniq_times = np.unique(time_s[event_s == 1])
    if uniq_times.size == 0:
        return np.zeros_like(t_grid, dtype=float)

    # Precompute risk set sizes at each unique event time
    H_by_time = {}
    n = len(time_s)

    for t in uniq_times:
        d = np.sum((time_s == t) & (event_s == 1))
        Y = np.sum(time_s >= t)
        if Y > 0:
            H_by_time[t] = d / Y

    # Build cumulative hazard step function
    H = np.zeros_like(t_grid, dtype=float)
    cum = 0.0
    j = 0
    uniq_times_sorted = np.sort(uniq_times)
    for i, tg in enumerate(t_grid):
        while j < len(uniq_times_sorted) and uniq_times_sorted[j] <= tg:
            cum += H_by_time[uniq_times_sorted[j]]
            j += 1
        H[i] = cum
    return H


# ============================================================================
# Gamma frailty inversion in H-space
# ============================================================================

def invert_gamma_frailty(H_obs: np.ndarray, theta: float) -> np.ndarray:
    """
    Given observed cumulative hazard under gamma frailty mixture:
      H_obs = (1/theta) * ln(1 + theta * H0)
    invert to get baseline (depletion-neutralized) cumulative hazard:
      H0 = (exp(theta * H_obs) - 1) / theta
    
    For theta -> 0, H0 = H_obs
    
    Args:
        H_obs: Observed cumulative hazard
        theta: Frailty variance parameter
    
    Returns:
        Depletion-neutralized baseline cumulative hazard
    """
    if theta is None or theta == 0.0 or np.abs(theta) < 1e-12:
        return H_obs.copy()
    # Numerical stability for small theta
    x = theta * H_obs
    return np.expm1(x) / theta


# ============================================================================
# Simulation
# ============================================================================

def simulate_two_cohorts(theta_B: float, cfg: CoxBiasConfig) -> pd.DataFrame:
    """
    Simulate two cohorts with identical baseline hazards and no treatment effect.
    
    Cohort A: theta=0 (z=1, no frailty heterogeneity)
    Cohort B: theta=theta_B (z ~ Gamma(1/theta, 1/theta) mean=1 var=theta)
    
    Baseline hazard: constant baseline_rate, no treatment effect.
    Right censor at censor_horizon.
    
    Args:
        theta_B: Frailty variance for cohort B
        cfg: Simulation configuration
    
    Returns:
        DataFrame with columns: time, event, cohort (0=A, 1=B)
    """
    rng = np.random.default_rng(cfg.seed + int(theta_B * 1000))

    nA = cfg.n_per_cohort
    nB = cfg.n_per_cohort

    # Frailty
    zA = np.ones(nA)
    if theta_B == 0:
        zB = np.ones(nB)
    else:
        shape = 1.0 / theta_B
        scale = theta_B
        zB = rng.gamma(shape=shape, scale=scale, size=nB)

    # Event times: exponential with rate = z * baseline_rate
    # T = -ln(U) / (z * lambda)
    U_A = rng.uniform(size=nA)
    U_B = rng.uniform(size=nB)
    T_A = -np.log(U_A) / (zA * cfg.baseline_rate)
    T_B = -np.log(U_B) / (zB * cfg.baseline_rate)

    # Apply administrative censoring
    C = cfg.censor_horizon
    time_A = np.minimum(T_A, C)
    time_B = np.minimum(T_B, C)
    event_A = (T_A <= C).astype(int)
    event_B = (T_B <= C).astype(int)

    dfA = pd.DataFrame({"time": time_A, "event": event_A, "cohort": 0})
    dfB = pd.DataFrame({"time": time_B, "event": event_B, "cohort": 1})
    return pd.concat([dfA, dfB], ignore_index=True)


# ============================================================================
# Cox Regression (PHReg)
# ============================================================================

def fit_cox_phreg(df: pd.DataFrame) -> Tuple[float, Tuple[float, float], float]:
    """
    Fit Cox model with cohort indicator only.
    
    Args:
        df: DataFrame with columns time, event, cohort
    
    Returns:
        HR, (CI_low, CI_high), p_value
    """
    # PHReg expects endog=time, exog array, status=event
    exog = df[["cohort"]].astype(float).to_numpy()
    model = PHReg(df["time"].to_numpy(), exog, status=df["event"].to_numpy())
    res = model.fit(disp=False)

    beta = res.params[0]
    se = res.bse[0]
    hr = float(np.exp(beta))
    ci_low = float(np.exp(beta - 1.96 * se))
    ci_high = float(np.exp(beta + 1.96 * se))
    p = float(res.pvalues[0])
    return hr, (ci_low, ci_high), p


# ============================================================================
# KCOR Computation
# ============================================================================

def compute_kcor_from_data(
    df: pd.DataFrame,
    theta_B: float,
    t_grid: np.ndarray,
    post_start_week: int = 8
) -> Tuple[float, float]:
    """
    Estimate cohort-specific observed cumulative hazards via Nelson-Aalen,
    invert gamma frailty for cohort B (theta_B), identity for cohort A (theta=0),
    then compute KCOR(t) = H0_B(t) / H0_A(t).

    Args:
        df: DataFrame with columns time, event, cohort
        theta_B: Known frailty variance for cohort B
        t_grid: Time grid for evaluation
        post_start_week: Week after which to compute slope
    
    Returns:
        kcor_asymptote: KCOR at final time
        kcor_slope: Slope of KCOR(t) on t_grid for t >= post_start_week (simple OLS)
    """
    dfA = df[df["cohort"] == 0]
    dfB = df[df["cohort"] == 1]

    Hobs_A = nelson_aalen(dfA["time"].to_numpy(), dfA["event"].to_numpy(), t_grid)
    Hobs_B = nelson_aalen(dfB["time"].to_numpy(), dfB["event"].to_numpy(), t_grid)

    H0_A = invert_gamma_frailty(Hobs_A, theta=0.0)
    H0_B = invert_gamma_frailty(Hobs_B, theta=theta_B)

    # Avoid division issues early when hazards are ~0
    eps = 1e-12
    kcor_t = (H0_B + eps) / (H0_A + eps)

    kcor_asymptote = float(kcor_t[-1])

    mask = t_grid >= post_start_week
    x = t_grid[mask].astype(float)
    y = kcor_t[mask].astype(float)

    if len(x) < 2:
        return kcor_asymptote, 0.0

    # Simple slope via least squares: y = a + b x
    x_centered = x - x.mean()
    if np.sum(x_centered ** 2) < 1e-12:
        b = 0.0
    else:
        b = float((x_centered @ (y - y.mean())) / (x_centered @ x_centered))
    
    return kcor_asymptote, b


# ============================================================================
# Main: Run grid and save results
# ============================================================================

def run_theta_grid(theta_list: List[float], cfg: CoxBiasConfig) -> pd.DataFrame:
    """
    Run simulation grid over theta values.
    
    Args:
        theta_list: List of theta values to simulate
        cfg: Simulation configuration
    
    Returns:
        DataFrame with results
    """
    rows: List[Dict] = []
    t_grid = np.arange(0, cfg.censor_horizon + 1, dtype=float)

    for theta in theta_list:
        print(f"Running simulation for theta_B = {theta}...", file=sys.stderr)
        df = simulate_two_cohorts(theta_B=theta, cfg=cfg)

        hr, (lo, hi), p = fit_cox_phreg(df)
        kcor_asym, kcor_slope = compute_kcor_from_data(
            df, theta_B=theta, t_grid=t_grid, post_start_week=cfg.post_start_week
        )

        rows.append({
            "theta_B": theta,
            "cox_HR": hr,
            "cox_CI_low": lo,
            "cox_CI_high": hi,
            "cox_p": p,
            "kcor_asymptote": kcor_asym,
            "kcor_post_slope": kcor_slope,
        })

        print(f"theta={theta:>5}: Cox HR={hr:.4f} [{lo:.4f},{hi:.4f}] p={p:.3g} | "
              f"KCOR_end={kcor_asym:.4f} slope={kcor_slope:.3e}", file=sys.stderr)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Cox bias demonstration simulation results"
    )
    parser.add_argument(
        "--output-results",
        type=str,
        default="cox_bias_results.csv",
        help="Output CSV file path for results"
    )
    parser.add_argument(
        "--n-per-cohort",
        type=int,
        default=200_000,
        help="Number of individuals per cohort (default: 200000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed (default: 123)"
    )
    
    args = parser.parse_args()

    cfg = CoxBiasConfig(
        n_per_cohort=args.n_per_cohort,
        baseline_rate=0.002,
        censor_horizon=52,
        seed=args.seed,
        post_start_week=8
    )

    theta_list = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    df_out = run_theta_grid(theta_list, cfg)

    # Save results
    output_dir = os.path.dirname(args.output_results)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_out.to_csv(args.output_results, index=False)
    print(f"Wrote results to: {args.output_results}", file=sys.stderr)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
S7 Simulation: Joint Frailty and Treatment Effect

This script implements the S7 simulation from Appendix B.6, which evaluates KCOR
under conditions where both selection-induced depletion (frailty heterogeneity)
and a true treatment effect (harm or benefit) are present simultaneously.

The simulation tests temporal separability: frailty parameters are estimated
during a quiet window where treatment effects are negligible, then KCOR should
remain flat during the quiet window and detect effects only during the treatment window.

Usage:
    python generate_s7_sim.py --output-data DATA.xlsx --output-results RESULTS.xlsx --output-diagnostics DIAG.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import gamma

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
# Import from same directory
from generate_sim_grid import (
    SimConfig,
    gamma_frailty_cumhaz,
    invert_gamma_frailty,
    fit_k_theta_cumhaz,
    H_model,
)


@dataclass
class S7Config:
    """Configuration for S7 simulation."""
    # Time parameters (event-time units: weeks since cohort entry)
    n_weeks: int = 260  # 5 years
    quiet_window_start: int = 80
    quiet_window_end: int = 140
    effect_window_start: int = 10  # Early effect (or 150 for late)
    effect_window_end: int = 25
    skip_weeks: int = 2
    
    # Cohort parameters
    n_initial: int = 2_000_000  # Large cohorts for stability
    
    # Frailty parameters (differing between cohorts)
    theta0: float = 0.3  # Cohort 0 frailty variance
    theta1: float = 0.8  # Cohort 1 frailty variance
    
    # Baseline hazard
    baseline_hazard: float = 0.002  # Weekly baseline hazard rate k
    
    # Treatment effect parameters
    effect_shape: str = "step"  # "step", "ramp", or "bump"
    effect_multiplier: float = 1.2  # r > 1 for harm, r < 1 for benefit
    
    # Random seed
    seed: int = 42
    
    # Enrollment date for KCOR_CMR format
    enrollment_date: str = "2021-06-14"


def make_frailty_mixture(theta: float, m: int = 60, q_lo: float = 0.001, q_hi: float = 0.999) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate Gamma(mean=1, var=theta) by m quantile bins.
    
    Returns (z_vals, w_vals) where sum w = 1.
    """
    # Gamma params: shape=1/theta, scale=theta => mean=1, var=theta
    k = 1.0 / theta
    scale = theta
    
    # Quantile grid
    qs = np.linspace(q_lo, q_hi, m + 1)
    # Use mid-quantile as representative point of each bin
    q_mid = 0.5 * (qs[:-1] + qs[1:])
    
    # Gamma PPF
    z_vals = gamma.ppf(q_mid, a=k, scale=scale)
    w_vals = np.diff(qs)  # Equal probability mass per bin
    
    return z_vals, w_vals


def baseline_hazard(t: np.ndarray, a: float = -6.2, b: float = 0.0) -> np.ndarray:
    """Log-linear baseline per week."""
    return np.exp(a + b * np.asarray(t))


def effect_multiplier(t: np.ndarray, t_on: int, t_off: int, kind: str = "step", r: float = 1.2) -> np.ndarray:
    """Compute time-localized treatment effect multiplier."""
    t_arr = np.asarray(t, dtype=float)
    result = np.ones_like(t_arr)
    
    if kind == "step":
        mask = (t_arr >= t_on) & (t_arr <= t_off)
        result[mask] = r
    elif kind == "ramp":
        mask = (t_arr >= t_on) & (t_arr <= t_off)
        if np.any(mask):
            u = (t_arr[mask] - t_on) / max(1, (t_off - t_on))
            result[mask] = 1.0 + u * (r - 1.0)
    elif kind == "bump":
        # Smooth pulse centered in window
        mu = 0.5 * (t_on + t_off)
        sigma = max(1.0, 0.25 * (t_off - t_on))
        result = 1.0 + (r - 1.0) * np.exp(-0.5 * ((t_arr - mu) / sigma) ** 2)
    else:
        raise ValueError(f"Unknown effect kind: {kind}")
    
    return result


def simulate_cohort(
    N: int,
    z_vals: np.ndarray,
    w_vals: np.ndarray,
    T: int,
    t_on: int,
    t_off: int,
    kind: str,
    r: float,
    baseline_k: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate cohort using discrete frailty-mixture approach.
    
    Args:
        N: Initial cohort size
        z_vals: Frailty values for each bin
        w_vals: Weights (probabilities) for each bin
        T: Time horizon (weeks)
        t_on: Effect window start
        t_off: Effect window end
        kind: Effect shape ("step", "ramp", "bump")
        r: Effect multiplier
        baseline_k: Baseline hazard level
    
    Returns:
        (weeks, alive, dead) arrays
    """
    survivors = (N * w_vals).copy()
    deaths = np.zeros(T, dtype=float)
    alive = np.zeros(T + 1, dtype=float)
    alive[0] = N
    
    for t in range(T):
        h0 = baseline_k  # Constant baseline hazard
        mult = effect_multiplier(np.array([t]), t_on, t_off, kind=kind, r=r)[0]
        
        # Hazard per bin this week
        h_bin = z_vals * h0 * mult
        
        # Convert hazard to death probability for the week (discrete-time approx)
        p_bin = 1.0 - np.exp(-h_bin)
        
        d_bin = survivors * p_bin
        deaths[t] = d_bin.sum()
        
        survivors = survivors - d_bin
        alive[t + 1] = survivors.sum()
    
    weeks = np.arange(T + 1)
    return weeks, alive, deaths


def compute_cumulative_hazard(weeks: np.ndarray, alive: np.ndarray, dead: np.ndarray, skip_weeks: int) -> np.ndarray:
    """Compute observed cumulative hazard from alive/dead arrays."""
    T = len(weeks) - 1
    H_obs = np.zeros(T + 1, dtype=float)
    
    for t in range(T):
        if t < skip_weeks:
            continue
        if alive[t] > 0:
            h_t = -np.log(1.0 - dead[t] / alive[t]) if dead[t] > 0 else 0.0
            H_obs[t + 1] = H_obs[t] + h_t
        else:
            H_obs[t + 1] = H_obs[t]
    
    return H_obs


def run_s7_scenario(
    config: S7Config,
    effect_shape: str = "step",
    effect_multiplier: float = 1.2,
    effect_window: Tuple[int, int] = (10, 25),
    overlap_variant: bool = False,
) -> Dict:
    """
    Run a single S7 scenario.
    
    Args:
        config: S7 configuration
        effect_shape: "step", "ramp", or "bump"
        effect_multiplier: Treatment effect multiplier (r)
        effect_window: (t_on, t_off) effect window
        overlap_variant: If True, intentionally overlap quiet and effect windows
    
    Returns:
        Dictionary with simulation results
    """
    np.random.seed(config.seed)
    
    # Adjust windows for overlap variant
    if overlap_variant:
        # Overlap quiet window with effect window
        quiet_start = config.quiet_window_start
        quiet_end = config.quiet_window_end
        effect_start = quiet_start - 10
        effect_end = quiet_start + 15
    else:
        quiet_start = config.quiet_window_start
        quiet_end = config.quiet_window_end
        effect_start, effect_end = effect_window
    
    # Create frailty mixtures
    z0, w0 = make_frailty_mixture(config.theta0, m=60)
    z1, w1 = make_frailty_mixture(config.theta1, m=60)
    
    # Simulate cohorts
    # Cohort 0: no effect
    weeks0, alive0, dead0 = simulate_cohort(
        config.n_initial, z0, w0, config.n_weeks,
        t_on=9999, t_off=9999, kind="step", r=1.0,
        baseline_k=config.baseline_hazard
    )
    
    # Cohort 1: with effect
    weeks1, alive1, dead1 = simulate_cohort(
        config.n_initial, z1, w1, config.n_weeks,
        t_on=effect_start, t_off=effect_end, kind=effect_shape, r=effect_multiplier,
        baseline_k=config.baseline_hazard
    )
    
    # Compute cumulative hazards
    H_obs0 = compute_cumulative_hazard(weeks0, alive0, dead0, config.skip_weeks)
    H_obs1 = compute_cumulative_hazard(weeks1, alive1, dead1, config.skip_weeks)
    
    # Fit frailty parameters during quiet window
    t_quiet = np.arange(quiet_start, quiet_end + 1)
    mask0 = (t_quiet >= config.skip_weeks) & (t_quiet < len(H_obs0))
    mask1 = (t_quiet >= config.skip_weeks) & (t_quiet < len(H_obs1))
    
    t_fit0 = t_quiet[mask0]
    H_fit0 = H_obs0[t_fit0]
    t_fit1 = t_quiet[mask1]
    H_fit1 = H_obs1[t_fit1]
    
    # Fit parameters
    (k0_hat, theta0_hat), diag0 = fit_k_theta_cumhaz(t_fit0, H_fit0)
    (k1_hat, theta1_hat), diag1 = fit_k_theta_cumhaz(t_fit1, H_fit1)
    
    # Normalize cumulative hazards
    H_tilde0 = invert_gamma_frailty(H_obs0, theta0_hat)
    H_tilde1 = invert_gamma_frailty(H_obs1, theta1_hat)
    
    # Compute KCOR
    kcor = np.divide(H_tilde1, H_tilde0, out=np.ones_like(H_tilde1), where=H_tilde0 != 0)
    
    return {
        "weeks": weeks0,
        "H_obs0": H_obs0,
        "H_obs1": H_obs1,
        "H_tilde0": H_tilde0,
        "H_tilde1": H_tilde1,
        "KCOR": kcor,
        "k0_hat": k0_hat,
        "theta0_hat": theta0_hat,
        "k1_hat": k1_hat,
        "theta1_hat": theta1_hat,
        "diag0": diag0,
        "diag1": diag1,
        "quiet_window": (quiet_start, quiet_end),
        "effect_window": (effect_start, effect_end),
        "effect_shape": effect_shape,
        "effect_multiplier": effect_multiplier,
        "overlap_variant": overlap_variant,
        "alive0": alive0,
        "dead0": dead0,
        "alive1": alive1,
        "dead1": dead1,
    }


def generate_kcor_cmr_format(results: Dict, config: S7Config, scenario_label: str) -> pd.DataFrame:
    """Convert simulation results to KCOR_CMR format."""
    weeks = results["weeks"]
    alive0 = results["alive0"]
    dead0 = results["dead0"]
    alive1 = results["alive1"]
    dead1 = results["dead1"]
    
    # Create DataFrame in KCOR_CMR format
    rows = []
    enrollment_date = datetime.strptime(config.enrollment_date, "%Y-%m-%d")
    
    for t in range(len(weeks) - 1):
        week_date = enrollment_date + timedelta(weeks=t)
        iso_week = week_date.isocalendar()
        iso_week_str = f"{iso_week[0]}_{iso_week[1]:02d}"
        
        # Cohort 0 (Dose 0)
        rows.append({
            "EnrollmentDate": config.enrollment_date,
            "ISOweekDied": iso_week_str,
            "DateDied": week_date.strftime("%Y-%m-%d"),
            "YearOfBirth": 1950,  # Fixed for simplicity
            "Sex": "M",  # Aggregated
            "Dose": 0,
            "Alive": int(alive0[t]),
            "Dead": int(dead0[t]),
        })
        
        # Cohort 1 (Dose 1)
        rows.append({
            "EnrollmentDate": config.enrollment_date,
            "ISOweekDied": iso_week_str,
            "DateDied": week_date.strftime("%Y-%m-%d"),
            "YearOfBirth": 1950,
            "Sex": "M",
            "Dose": 1,
            "Alive": int(alive1[t]),
            "Dead": int(dead1[t]),
        })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate S7 simulation data and results")
    parser.add_argument("--output-data", required=True, help="Output Excel file for simulation data")
    parser.add_argument("--output-results", required=True, help="Output Excel file for KCOR results")
    parser.add_argument("--output-diagnostics", required=True, help="Output CSV file for diagnostics")
    parser.add_argument("--config-seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = S7Config(seed=args.config_seed)
    
    # Run S7 scenarios according to punchlist8 spec
    scenarios = []
    
    # Minimal grid: 3 shapes × 2 magnitudes × 2 timings + 1 overlap = 13 runs
    effect_shapes = ["step", "ramp", "bump"]
    effect_magnitudes = [1.2, 0.8]  # harm, benefit
    effect_timings = [(10, 25), (150, 190)]  # early, late
    
    for shape in effect_shapes:
        for r in effect_magnitudes:
            for timing in effect_timings:
                label = f"S7_{shape}_r{r:.1f}_{'early' if timing[0] < 50 else 'late'}"
                result = run_s7_scenario(
                    config,
                    effect_shape=shape,
                    effect_multiplier=r,
                    effect_window=timing,
                    overlap_variant=False,
                )
                scenarios.append((label, result))
    
    # Overlap variant
    label = "S7_overlap_step_r1.2"
    result = run_s7_scenario(
        config,
        effect_shape="step",
        effect_multiplier=1.2,
        effect_window=(config.quiet_window_start - 10, config.quiet_window_start + 15),
        overlap_variant=True,
    )
    scenarios.append((label, result))
    
    # Generate output files
    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_results), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_diagnostics), exist_ok=True)
    
    # Write data (use first scenario as example)
    if scenarios:
        df_data = generate_kcor_cmr_format(scenarios[0][1], config, scenarios[0][0])
        with pd.ExcelWriter(args.output_data, engine='openpyxl') as writer:
            df_data.to_excel(writer, sheet_name='S7_data', index=False)
    
    # Write results and diagnostics
    results_rows = []
    diag_rows = []
    
    for label, result in scenarios:
        weeks = result["weeks"]
        kcor = result["KCOR"]
        
        # Results: KCOR trajectory
        for t in range(len(weeks)):
            results_rows.append({
                "scenario": label,
                "week": int(weeks[t]),
                "KCOR": float(kcor[t]),
                "H_obs0": float(result["H_obs0"][t]),
                "H_obs1": float(result["H_obs1"][t]),
                "H_tilde0": float(result["H_tilde0"][t]),
                "H_tilde1": float(result["H_tilde1"][t]),
            })
        
        # Diagnostics
        diag_rows.append({
            "scenario": label,
            "k0_hat": float(result["k0_hat"]),
            "theta0_hat": float(result["theta0_hat"]),
            "k1_hat": float(result["k1_hat"]),
            "theta1_hat": float(result["theta1_hat"]),
            "rmse0": float(result["diag0"].get("rmse_Hobs", np.nan)),
            "rmse1": float(result["diag1"].get("rmse_Hobs", np.nan)),
            "n_obs0": int(result["diag0"].get("n_obs", 0)),
            "n_obs1": int(result["diag1"].get("n_obs", 0)),
            "success0": bool(result["diag0"].get("success", False)),
            "success1": bool(result["diag1"].get("success", False)),
            "quiet_window": f"{result['quiet_window'][0]}-{result['quiet_window'][1]}",
            "effect_window": f"{result['effect_window'][0]}-{result['effect_window'][1]}",
            "effect_shape": result["effect_shape"],
            "effect_multiplier": result["effect_multiplier"],
            "overlap_variant": result["overlap_variant"],
        })
    
    df_results = pd.DataFrame(results_rows)
    df_diag = pd.DataFrame(diag_rows)
    
    with pd.ExcelWriter(args.output_results, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='S7_results', index=False)
    
    df_diag.to_csv(args.output_diagnostics, index=False)
    
    print(f"Generated S7 simulation data: {len(scenarios)} scenarios", file=sys.stderr)
    print(f"  Data: {args.output_data}", file=sys.stderr)
    print(f"  Results: {args.output_results}", file=sys.stderr)
    print(f"  Diagnostics: {args.output_diagnostics}", file=sys.stderr)


if __name__ == "__main__":
    main()


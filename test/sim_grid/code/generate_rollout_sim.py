#!/usr/bin/env python3
"""
Obel/Chemaitelly design-mimic simulation.

This script simulates rollout-style cohorts with:
- Calendar-time wave-shaped hazard (epidemic waves)
- Uptake correlated with latent frailty
- Censoring/entry windows similar to "enrollment during rollout"

Demonstrates that Cox regression can return strongly non-null HRs
under a true null treatment effect, whereas KCOR remains near null.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from lifelines import CoxPHFitter

# Add parent directory to path to import shared functions
sys.path.insert(0, os.path.dirname(__file__))
from generate_sim_grid import (
    SimConfig,
    fit_k_theta_cumhaz,
    invert_gamma_frailty,
)

def simulate_rollout_cohort(
    N: int,
    baseline_k: float,
    theta: float,
    wave_amplitude: float,
    wave_period: float,
    wave_phase: float,
    uptake_correlation: float,
    enrollment_start: int,
    enrollment_end: int,
    n_weeks: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a rollout-style cohort with wave-shaped hazard and frailty-correlated uptake.
    
    Args:
        N: Initial cohort size
        baseline_k: Baseline hazard rate
        theta: Frailty variance
        wave_amplitude: Amplitude of epidemic wave (multiplicative)
        wave_period: Period of wave in weeks
        wave_phase: Phase offset for wave
        uptake_correlation: Correlation between frailty and uptake timing (0-1)
        enrollment_start: Start week of enrollment window
        enrollment_end: End week of enrollment window
        n_weeks: Total follow-up weeks
        rng: Random number generator
    
    Returns:
        Tuple of (weeks, alive, dead, frailty_values)
    """
    # Generate frailty values (gamma distribution)
    alpha = 1.0 / theta
    beta = 1.0 / theta
    frailty = rng.gamma(alpha, 1/beta, N)
    
    # Generate enrollment times correlated with frailty
    # Lower frailty (healthier) enrolls earlier
    frailty_normalized = (frailty - np.mean(frailty)) / np.std(frailty)
    enrollment_times = (
        enrollment_start + 
        (enrollment_end - enrollment_start) * 
        (0.5 - uptake_correlation * frailty_normalized / 2)
    )
    enrollment_times = np.clip(enrollment_times, enrollment_start, enrollment_end)
    
    # Initialize arrays
    weeks = np.arange(n_weeks + 1)
    alive = np.zeros(n_weeks + 1, dtype=int)
    dead = np.zeros(n_weeks + 1, dtype=int)
    
    # Track individual survival
    individual_alive = np.ones(N, dtype=bool)
    individual_enrolled = np.zeros(N, dtype=bool)
    
    # Simulate week by week
    for t in range(n_weeks):
        # Determine who is enrolled by week t
        enrolled_mask = (enrollment_times <= t) & individual_alive
        n_enrolled = np.sum(enrolled_mask)
        
        if n_enrolled == 0:
            alive[t] = 0
            dead[t] = 0
            continue
        
        # Compute wave-shaped baseline hazard
        wave_factor = 1.0 + wave_amplitude * np.sin(2 * np.pi * t / wave_period + wave_phase)
        h_base = baseline_k * wave_factor
        
        # Compute individual hazards (frailty × baseline)
        individual_hazards = frailty[enrolled_mask] * h_base
        
        # Weekly survival probability for each individual
        weekly_survival = np.exp(-individual_hazards)
        
        # Sample deaths
        deaths_this_week = rng.binomial(1, 1 - weekly_survival, n_enrolled)
        n_dead = np.sum(deaths_this_week)
        
        # Update individual status
        dead_indices = np.where(enrolled_mask)[0][deaths_this_week == 1]
        individual_alive[dead_indices] = False
        
        alive[t] = n_enrolled - n_dead
        dead[t] = n_dead
    
    # Final week
    alive[n_weeks] = np.sum((enrollment_times <= n_weeks) & individual_alive)
    
    return weeks, alive, dead, frailty


def run_rollout_scenario(
    config: SimConfig,
    wave_amplitude: float = 0.5,
    wave_period: float = 26.0,  # ~6 months
    uptake_correlation: float = 0.6,
    enrollment_start: int = 0,
    enrollment_end: int = 20,
) -> Dict:
    """
    Run rollout-style scenario simulation.
    
    Returns results comparing Cox HR and KCOR under null treatment effect.
    """
    rng = np.random.default_rng(config.seed)
    
    # Simulate control cohort (Dose 0)
    weeks_0, alive_0, dead_0, frailty_0 = simulate_rollout_cohort(
        N=config.n_initial,
        baseline_k=config.baseline_hazard,
        theta=config.theta_A,
        wave_amplitude=wave_amplitude,
        wave_period=wave_period,
        wave_phase=0.0,
        uptake_correlation=uptake_correlation,
        enrollment_start=enrollment_start,
        enrollment_end=enrollment_end,
        n_weeks=config.n_weeks,
        rng=rng,
    )
    
    # Simulate treatment cohort (Dose 1) - SAME parameters (null by construction)
    weeks_1, alive_1, dead_1, frailty_1 = simulate_rollout_cohort(
        N=config.n_initial,
        baseline_k=config.baseline_hazard,
        theta=config.theta_B,
        wave_amplitude=wave_amplitude,
        wave_period=wave_period,
        wave_phase=0.0,  # Same phase
        uptake_correlation=uptake_correlation,
        enrollment_start=enrollment_start,
        enrollment_end=enrollment_end,
        n_weeks=config.n_weeks,
        rng=np.random.default_rng(config.seed + 1000),
    )
    
    # Prepare data for Cox regression
    # Create individual-level data
    cox_data = []
    
    for dose in [0, 1]:
        weeks = weeks_0 if dose == 0 else weeks_1
        alive = alive_0 if dose == 0 else alive_1
        dead = dead_0 if dose == 0 else dead_1
        
        for t in range(len(weeks) - 1):
            if alive[t] > 0 and dead[t] > 0:
                # Create one row per death
                for _ in range(dead[t]):
                    cox_data.append({
                        'dose': dose,
                        'time': t + 1,
                        'event': 1,
                    })
    
    if len(cox_data) == 0:
        cox_hr = np.nan
        cox_p = np.nan
    else:
        df_cox = pd.DataFrame(cox_data)
        
        # Fit Cox model
        try:
            cph = CoxPHFitter()
            cph.fit(df_cox, duration_col='time', event_col='event')
            cox_hr = float(np.exp(cph.hazard_ratios_['dose']))
            cox_p = float(cph.summary['p'].iloc[0])
        except Exception as e:
            print(f"  Cox regression failed: {e}")
            cox_hr = np.nan
            cox_p = np.nan
    
    # Compute KCOR
    # Compute cumulative hazards
    MR_0 = np.where(alive_0 > 0, dead_0 / alive_0, 0)
    MR_0 = np.clip(MR_0, 0, 0.99)
    hazard_0 = -np.log(1 - MR_0)
    hazard_0 = np.nan_to_num(hazard_0, nan=0, posinf=0, neginf=0)
    
    MR_1 = np.where(alive_1 > 0, dead_1 / alive_1, 0)
    MR_1 = np.clip(MR_1, 0, 0.99)
    hazard_1 = -np.log(1 - MR_1)
    hazard_1 = np.nan_to_num(hazard_1, nan=0, posinf=0, neginf=0)
    
    # Apply skip-weeks
    h_eff_0 = np.where(weeks_0 >= config.skip_weeks, hazard_0, 0)
    h_eff_1 = np.where(weeks_1 >= config.skip_weeks, hazard_1, 0)
    H_obs_0 = np.cumsum(h_eff_0)
    H_obs_1 = np.cumsum(h_eff_1)
    
    # Fit frailty parameters
    quiet_mask_0 = (weeks_0 >= config.quiet_window_start) & (weeks_0 <= config.quiet_window_end)
    t_quiet_0 = weeks_0[quiet_mask_0].astype(float)
    H_quiet_0 = H_obs_0[quiet_mask_0]
    
    quiet_mask_1 = (weeks_1 >= config.quiet_window_start) & (weeks_1 <= config.quiet_window_end)
    t_quiet_1 = weeks_1[quiet_mask_1].astype(float)
    H_quiet_1 = H_obs_1[quiet_mask_1]
    
    (k0_hat, theta0_hat), _ = fit_k_theta_cumhaz(t_quiet_0, H_quiet_0)
    (k1_hat, theta1_hat), _ = fit_k_theta_cumhaz(t_quiet_1, H_quiet_1)
    
    # Normalize
    H0_0 = invert_gamma_frailty(H_obs_0, theta0_hat)
    H0_1 = invert_gamma_frailty(H_obs_1, theta1_hat)
    
    # Compute KCOR
    with np.errstate(divide='ignore', invalid='ignore'):
        kcor_raw = np.where(H0_0 > 1e-10, H0_1 / H0_0, np.nan)
    
    norm_week_idx = config.skip_weeks + 4
    if norm_week_idx < len(kcor_raw) and np.isfinite(kcor_raw[norm_week_idx]) and kcor_raw[norm_week_idx] > 0:
        kcor = kcor_raw / kcor_raw[norm_week_idx]
    else:
        kcor = kcor_raw
    
    # Summary over diagnostic window
    diagnostic_mask = (weeks_0 >= 20) & (weeks_0 <= 100)
    kcor_diagnostic = kcor[diagnostic_mask]
    kcor_diagnostic = kcor_diagnostic[np.isfinite(kcor_diagnostic)]
    
    kcor_median = float(np.median(kcor_diagnostic)) if len(kcor_diagnostic) > 0 else np.nan
    
    return {
        "cox_hr": cox_hr,
        "cox_p": cox_p,
        "kcor_median": kcor_median,
        "theta0_hat": float(theta0_hat),
        "theta1_hat": float(theta1_hat),
        "wave_amplitude": wave_amplitude,
        "uptake_correlation": uptake_correlation,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate rollout-style design-mimic simulation"
    )
    parser.add_argument(
        "--output-results", "-o",
        default="test/sim_grid/out/rollout_sim_results.csv",
        help="Output path for results CSV"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    config = SimConfig(seed=args.seed)
    
    print("=" * 60)
    print("Rollout-Style Design-Mimic Simulation")
    print("=" * 60)
    print("Simulating rollout cohorts with wave-shaped hazards and")
    print("frailty-correlated uptake (null treatment effect)...")
    print()
    
    result = run_rollout_scenario(config)
    
    print(f"Cox HR: {result['cox_hr']:.4f} (p = {result['cox_p']:.2e})")
    print(f"KCOR median: {result['kcor_median']:.4f}")
    print()
    
    if result['cox_p'] < 0.05:
        print("✓ Cox regression returns significant non-null HR under true null")
    if abs(result['kcor_median'] - 1.0) < 0.05:
        print("✓ KCOR remains centered near null")
    
    # Save results
    df = pd.DataFrame([result])
    os.makedirs(os.path.dirname(args.output_results), exist_ok=True)
    df.to_csv(args.output_results, index=False)
    print(f"\nSaved results to: {args.output_results}")


if __name__ == "__main__":
    main()


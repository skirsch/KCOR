#!/usr/bin/env python3
"""
Skip-weeks sensitivity analysis for KCOR.

This script generates a figure showing KCOR(t) computed with different
skip_weeks values (0, 4, 8) on the same fixed-cohort comparison to
illustrate dynamic selection effects in early follow-up.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Add parent directory to path to import shared functions
sys.path.insert(0, os.path.dirname(__file__))
from generate_sim_grid import (
    SimConfig,
    simulate_gamma_frailty_cohort,
    fit_k_theta_cumhaz,
    invert_gamma_frailty,
)

def compute_kcor_with_skip_weeks(
    weeks: np.ndarray,
    alive: np.ndarray,
    dead: np.ndarray,
    skip_weeks: int,
    quiet_window_start: int,
    quiet_window_end: int,
    normalization_weeks: int = 4,
) -> np.ndarray:
    """
    Compute KCOR trajectory with specified skip_weeks.
    
    Args:
        weeks: Time points
        alive: Number alive at start of each week
        dead: Number dead during each week
        skip_weeks: Number of weeks to skip before accumulating hazard
        quiet_window_start: Start of quiet window for frailty fitting
        quiet_window_end: End of quiet window for frailty fitting
        normalization_weeks: Weeks after skip_weeks to use for normalization baseline
    
    Returns:
        KCOR(t) trajectory
    """
    # Compute hazard from mortality rate
    MR = np.where(alive > 0, dead / alive, 0)
    MR = np.clip(MR, 0, 0.99)
    hazard = -np.log(1 - MR)
    hazard = np.nan_to_num(hazard, nan=0, posinf=0, neginf=0)
    
    # Apply skip-week rule and compute cumulative hazard
    h_eff = np.where(weeks >= skip_weeks, hazard, 0)
    H_obs = np.cumsum(h_eff)
    
    # Extract quiet-window data for fitting
    quiet_mask = (weeks >= quiet_window_start) & (weeks <= quiet_window_end)
    t_quiet = weeks[quiet_mask].astype(float)
    H_quiet = H_obs[quiet_mask]
    
    if len(t_quiet) < 3:
        return np.full_like(weeks, np.nan)
    
    # Fit gamma-frailty model
    (k_hat, theta_hat), fit_diag = fit_k_theta_cumhaz(t_quiet, H_quiet)
    
    # Invert to get depletion-neutralized cumulative hazard
    if np.isfinite(theta_hat) and theta_hat >= 0:
        H0 = invert_gamma_frailty(H_obs, theta_hat)
    else:
        H0 = H_obs.copy()
    
    return H0


def generate_skip_weeks_sensitivity_figure(
    output_path: str,
    config: SimConfig,
) -> None:
    """
    Generate skip-weeks sensitivity figure.
    
    Uses a representative gamma-frailty null scenario and computes
    KCOR(t) with skip_weeks = 0, 4, 8.
    """
    print("Generating skip-weeks sensitivity analysis...")
    
    # Generate representative scenario (gamma-frailty null)
    weeks_A, alive_A, dead_A = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, config.theta_A
    )
    
    weeks_B, alive_B, dead_B = simulate_gamma_frailty_cohort(
        config, config.n_initial, config.baseline_hazard, config.theta_B
    )
    
    # Compute KCOR for different skip_weeks values
    skip_values = [0, 4, 8]
    kcor_trajectories = {}
    
    for skip_weeks in skip_values:
        print(f"  Computing KCOR with skip_weeks={skip_weeks}...")
        
        # Temporarily modify config skip_weeks
        original_skip = config.skip_weeks
        config.skip_weeks = skip_weeks
        
        # Compute normalized cumulative hazards for both cohorts
        H0_A = compute_kcor_with_skip_weeks(
            weeks_A, alive_A, dead_A,
            skip_weeks, config.quiet_window_start, config.quiet_window_end
        )
        
        H0_B = compute_kcor_with_skip_weeks(
            weeks_B, alive_B, dead_B,
            skip_weeks, config.quiet_window_start, config.quiet_window_end
        )
        
        # Compute KCOR ratio (Dose 1 / Dose 0, where A is dose 1, B is dose 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            kcor_raw = np.where(H0_B > 1e-10, H0_A / H0_B, np.nan)
        
        # Normalize at effective baseline week
        norm_week_idx = skip_weeks + 4  # 4 weeks after skip_weeks
        if norm_week_idx < len(kcor_raw) and np.isfinite(kcor_raw[norm_week_idx]) and kcor_raw[norm_week_idx] > 0:
            kcor = kcor_raw / kcor_raw[norm_week_idx]
        else:
            kcor = kcor_raw
        
        kcor_trajectories[skip_weeks] = kcor
        
        # Restore original skip_weeks
        config.skip_weeks = original_skip
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KCOR trajectories for early follow-up (first 30 weeks)
    early_weeks = weeks_A[weeks_A <= 30]
    early_idx = len(early_weeks)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    for i, skip_weeks in enumerate(skip_values):
        kcor = kcor_trajectories[skip_weeks][:early_idx]
        ax.plot(early_weeks, kcor, 
                label=f'skip_weeks = {skip_weeks}',
                color=colors[i], linewidth=2, alpha=0.8)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Null (KCOR=1)')
    ax.set_xlabel('Weeks since enrollment', fontsize=12)
    ax.set_ylabel('KCOR(t)', fontsize=12)
    ax.set_title('Skip-window sensitivity illustrates dynamic selection effects in early follow-up', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.85, 1.15)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate skip-weeks sensitivity figure"
    )
    parser.add_argument(
        "--output-figure", "-o",
        default="test/sim_grid/out/fig_skip_weeks_sensitivity.png",
        help="Output path for figure"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    config = SimConfig(seed=args.seed)
    
    generate_skip_weeks_sensitivity_figure(args.output_figure, config)
    print("Skip-weeks sensitivity analysis complete!")


if __name__ == "__main__":
    main()


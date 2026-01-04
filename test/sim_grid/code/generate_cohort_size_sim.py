#!/usr/bin/env python3
"""
Minimum cohort size simulation for KCOR.

This script simulates KCOR behavior across different cohort sizes to
determine the minimum cohort size where KCOR works reliably.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

# Add parent directory to path to import shared functions
sys.path.insert(0, os.path.dirname(__file__))
from generate_sim_grid import (
    SimConfig,
    simulate_gamma_frailty_cohort,
    fit_k_theta_cumhaz,
    invert_gamma_frailty,
    compute_kcor_for_scenario,
)

def simulate_cohort_size_scenario(
    n_per_arm: int,
    config: SimConfig,
    n_replicates: int = 500,
) -> Dict:
    """
    Simulate KCOR for a given cohort size with multiple replicates.
    
    Args:
        n_per_arm: Number of individuals per arm
        config: Simulation configuration
        n_replicates: Number of Monte Carlo replicates
    
    Returns:
        Dictionary with summary statistics
    """
    kcor_values = []
    false_signal_count = 0
    
    for rep in range(n_replicates):
        # Generate gamma-frailty null scenario
        # Use different seeds for each replicate
        original_seed = config.seed
        config.seed = config.seed + rep
        
        weeks_A, alive_A, dead_A = simulate_gamma_frailty_cohort(
            config, n_per_arm, config.baseline_hazard, config.theta_A
        )
        
        config.seed = config.seed + 1000
        weeks_B, alive_B, dead_B = simulate_gamma_frailty_cohort(
            config, n_per_arm, config.baseline_hazard, config.theta_B
        )
        
        # Restore original seed
        config.seed = original_seed
        
        scenario_data = {
            "scenario": "gamma_null",
            "label": f"Gamma-Frailty Null (N={n_per_arm})",
            "cohorts": [
                {"dose": 0, "weeks": weeks_A, "alive": alive_A, "dead": dead_A, "theta_true": config.theta_A},
                {"dose": 1, "weeks": weeks_B, "alive": alive_B, "dead": dead_B, "theta_true": config.theta_B},
            ],
        }
        
        # Compute KCOR
        result = compute_kcor_for_scenario(scenario_data, config)
        
        # Extract median KCOR over diagnostic window (weeks 20-100)
        kcor_median = result.get("kcor_median", np.nan)
        
        if np.isfinite(kcor_median):
            kcor_values.append(kcor_median)
            
            # Check for false signal: |log(KCOR)| > 0.05
            if abs(np.log(kcor_median)) > 0.05:
                false_signal_count += 1
    
    kcor_array = np.array(kcor_values)
    
    if len(kcor_array) == 0:
        return {
            "n_per_arm": n_per_arm,
            "n_replicates": n_replicates,
            "n_successful": 0,
            "mean_kcor": np.nan,
            "sd_kcor": np.nan,
            "median_kcor": np.nan,
            "ci_width_95": np.nan,
            "false_signal_rate": np.nan,
        }
    
    # Compute statistics
    mean_kcor = float(np.mean(kcor_array))
    sd_kcor = float(np.std(kcor_array))
    median_kcor = float(np.median(kcor_array))
    
    # 95% CI width
    ci_low, ci_high = np.percentile(kcor_array, [2.5, 97.5])
    ci_width_95 = float(ci_high - ci_low)
    
    # False signal rate
    false_signal_rate = false_signal_count / len(kcor_array)
    
    return {
        "n_per_arm": n_per_arm,
        "n_replicates": n_replicates,
        "n_successful": len(kcor_array),
        "mean_kcor": mean_kcor,
        "sd_kcor": sd_kcor,
        "median_kcor": median_kcor,
        "ci_width_95": ci_width_95,
        "false_signal_rate": false_signal_rate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def generate_cohort_size_analysis(
    output_path: str,
    config: SimConfig,
    cohort_sizes: List[int] = None,
    n_replicates: int = 500,
) -> pd.DataFrame:
    """
    Generate cohort size sensitivity analysis.
    
    Args:
        output_path: Path to save results CSV
        config: Simulation configuration
        cohort_sizes: List of cohort sizes to test (per arm)
        n_replicates: Number of Monte Carlo replicates per cohort size
    """
    if cohort_sizes is None:
        cohort_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    
    print("=" * 60)
    print("Cohort Size Sensitivity Analysis")
    print("=" * 60)
    print(f"Cohort sizes: {cohort_sizes}")
    print(f"Replicates per size: {n_replicates}")
    print()
    
    results = []
    
    for n_per_arm in cohort_sizes:
        print(f"Simulating N = {n_per_arm:,} per arm...")
        result = simulate_cohort_size_scenario(n_per_arm, config, n_replicates)
        results.append(result)
        
        print(f"  Mean KCOR: {result['mean_kcor']:.4f}")
        print(f"  SD: {result['sd_kcor']:.4f}")
        print(f"  95% CI width: {result['ci_width_95']:.4f}")
        print(f"  False signal rate: {result['false_signal_rate']:.2%}")
        print()
    
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")
    
    # Determine minimum cohort size threshold
    # Use criterion: false_signal_rate < 5% and CI width < 0.1
    stable_mask = (df['false_signal_rate'] < 0.05) & (df['ci_width_95'] < 0.1)
    if stable_mask.any():
        min_stable_n = int(df[stable_mask]['n_per_arm'].min())
        print(f"\nMinimum stable cohort size: N = {min_stable_n:,} per arm")
        print("(Based on false_signal_rate < 5% and CI width < 0.1)")
    else:
        print("\nNo cohort size met stability criteria with current thresholds")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate cohort size sensitivity analysis"
    )
    parser.add_argument(
        "--output-results", "-o",
        default="test/sim_grid/out/cohort_size_sensitivity.csv",
        help="Output path for results CSV"
    )
    parser.add_argument(
        "--cohort-sizes", "-n",
        type=str,
        default="1000,2000,5000,10000,20000,50000",
        help="Comma-separated list of cohort sizes (per arm)"
    )
    parser.add_argument(
        "--replicates", "-r",
        type=int,
        default=500,
        help="Number of Monte Carlo replicates per cohort size"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    cohort_sizes = [int(x.strip()) for x in args.cohort_sizes.split(',')]
    
    config = SimConfig(seed=args.seed)
    
    df = generate_cohort_size_analysis(
        args.output_results,
        config,
        cohort_sizes=cohort_sizes,
        n_replicates=args.replicates,
    )
    
    print("\nCohort size sensitivity analysis complete!")


if __name__ == "__main__":
    main()


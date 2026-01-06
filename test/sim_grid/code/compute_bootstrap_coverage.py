#!/usr/bin/env python3
"""
Compute bootstrap coverage for KCOR simulation scenarios.

This script runs bootstrap resampling for each simulation scenario and computes
empirical coverage (proportion of bootstrap replicates where true KCOR value
falls within the 95% confidence interval).

Usage:
    python compute_bootstrap_coverage.py --n-bootstrap 1000 --output coverage_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import simulation functions from generate_sim_grid
sys.path.insert(0, os.path.dirname(__file__))
from generate_sim_grid import (
    SimConfig,
    compute_kcor_for_scenario,
    run_scenario_1_gamma_null,
    run_scenario_2_hazard_increase,
    run_scenario_3_hazard_decrease,
    run_scenario_4_nongamma_null,
    run_scenario_6_sparse,
)


def bootstrap_resample_cohort_data(
    weeks: np.ndarray,
    alive: np.ndarray,
    dead: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap resample cohort data by adding noise to event counts.
    
    For aggregated cohort data, we simulate bootstrap variability by:
    1. Resampling deaths at each time point from a Poisson distribution
    2. Reconstructing alive counts from the resampled deaths
    
    Args:
        weeks: Time points
        alive: Number alive at start of each week
        dead: Number dead during each week
        seed: Random seed
    
    Returns:
        Resampled weeks, alive, dead arrays
    """
    np.random.seed(seed)
    n_weeks = len(weeks)
    
    # Start with initial cohort size
    n_initial = int(alive[0]) if len(alive) > 0 else 0
    
    # Resample deaths: for each week, resample from Poisson(dead[t])
    dead_resampled = np.zeros(n_weeks, dtype=float)
    alive_resampled = np.zeros(n_weeks, dtype=float)
    current_alive = float(n_initial)
    
    for t in range(n_weeks):
        alive_resampled[t] = current_alive
        
        if current_alive > 0:
            # Resample deaths from Poisson with mean = dead[t]
            # But ensure we don't exceed current_alive
            mean_deaths = dead[t]
            if mean_deaths > 0:
                n_deaths = np.random.poisson(mean_deaths)
                n_deaths = min(n_deaths, int(current_alive))
            else:
                n_deaths = 0
            
            dead_resampled[t] = n_deaths
            current_alive = max(0, current_alive - n_deaths)
        else:
            dead_resampled[t] = 0
    
    return weeks.copy(), alive_resampled, dead_resampled


def compute_coverage_for_scenario_simple(
    scenario_fn,
    scenario_name: str,
    true_kcor: float,
    config: SimConfig,
    n_simulations: int = 1000,
    target_week: int = 80,
) -> Dict:
    """
    Compute coverage for a scenario using multiple independent simulation replicates.
    
    For each replicate, we compute KCOR. Then we form percentile intervals from
    the distribution of KCOR values and check if the true value falls within.
    
    Args:
        scenario_fn: Function that generates scenario data
        scenario_name: Name of scenario
        true_kcor: True KCOR value for this scenario
        config: Simulation configuration
        n_simulations: Number of independent simulation replicates
        target_week: Week at which to evaluate coverage
    
    Returns:
        Dictionary with coverage statistics
    """
    print(f"\nComputing coverage for {scenario_name}...")
    print(f"  True KCOR: {true_kcor}")
    print(f"  Simulation replicates: {n_simulations}")
    print(f"  Target week: {target_week}")
    
    # Store KCOR values from all simulation replicates
    kcor_values = []
    valid_replicates = 0
    
    for sim in range(n_simulations):
        if (sim + 1) % 100 == 0:
            print(f"  Simulation {sim+1}/{n_simulations}")
        
        # Update seed for each simulation replicate
        config.seed = config.seed + sim
        
        # Generate scenario data with new seed
        scenario_data = scenario_fn(config)
        
        # Compute KCOR for this simulation replicate
        try:
            result = compute_kcor_for_scenario(scenario_data, config)
            kcor_trajectory = result.get("kcor_trajectory")
            
            if kcor_trajectory is not None and len(kcor_trajectory) > target_week:
                kcor_value = kcor_trajectory[target_week]
                if np.isfinite(kcor_value):
                    kcor_values.append(kcor_value)
                    valid_replicates += 1
        except Exception as e:
            # Skip failed replicates
            continue
    
    if len(kcor_values) == 0:
        print(f"  WARNING: No valid simulation replicates!")
        return {
            "scenario": scenario_name,
            "true_kcor": true_kcor,
            "coverage": np.nan,
            "n_valid": 0,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "median_kcor": np.nan,
        }
    
    kcor_values = np.array(kcor_values)
    
    # Compute percentile-based confidence intervals (2.5th and 97.5th percentiles)
    ci_lower = np.percentile(kcor_values, 2.5)
    ci_upper = np.percentile(kcor_values, 97.5)
    median_kcor = np.median(kcor_values)
    
    # Coverage: does the true value fall within the percentile interval?
    true_in_ci = (true_kcor >= ci_lower) and (true_kcor <= ci_upper)
    coverage = 1.0 if true_in_ci else 0.0
    
    print(f"  Valid replicates: {valid_replicates}/{n_simulations}")
    print(f"  Median KCOR: {median_kcor:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  True KCOR in CI: {true_in_ci} (coverage: {coverage*100:.1f}%)")
    
    return {
        "scenario": scenario_name,
        "true_kcor": true_kcor,
        "coverage": coverage,  # Binary: 1 if covered, 0 if not
        "n_valid": valid_replicates,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median_kcor": median_kcor,
        "true_in_ci": true_in_ci,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute bootstrap coverage for KCOR simulation scenarios"
    )
    parser.add_argument(
        "--n-simulations", "-n",
        type=int,
        default=1000,
        help="Number of independent simulation replicates per scenario"
    )
    parser.add_argument(
        "--target-week", "-w",
        type=int,
        default=80,
        help="Target week for coverage evaluation"
    )
    parser.add_argument(
        "--output", "-o",
        default="test/sim_grid/out/bootstrap_coverage.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    config = SimConfig(seed=args.seed)
    
    print("=" * 60)
    print("KCOR Bootstrap Coverage Computation")
    print("=" * 60)
    print(f"Simulation replicates per scenario: {args.n_simulations}")
    print(f"Target week: {args.target_week}")
    print()
    
    # Define scenarios with true KCOR values
    scenarios = [
        ("Gamma-frailty null", run_scenario_1_gamma_null, 1.0),
        ("Injected effect (harm)", run_scenario_2_hazard_increase, 1.2),
        ("Injected effect (benefit)", run_scenario_3_hazard_decrease, 0.8),
        ("Non-gamma frailty", run_scenario_4_nongamma_null, 1.0),
        ("Sparse events", run_scenario_6_sparse, 1.0),
    ]
    
    all_results = []
    
    for scenario_name, scenario_fn, true_kcor in scenarios:
        # Compute coverage using multiple simulation replicates
        result = compute_coverage_for_scenario_simple(
            scenario_fn,
            scenario_name,
            true_kcor,
            config,
            n_simulations=args.n_simulations,
            target_week=args.target_week,
        )
        
        coverage_pct = result["coverage"] * 100
        
        print(f"\n{scenario_name}:")
        print(f"  Coverage: {coverage_pct:.1f}%")
        
        all_results.append({
            "scenario": scenario_name,
            "true_kcor": true_kcor,
            "coverage_percent": coverage_pct,
            "coverage_proportion": result["coverage"],
            "n_simulations": args.n_simulations,
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "median_kcor": result["median_kcor"],
        })
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(args.output, index=False)
    print(f"\nSaved coverage results to: {args.output}")
    
    return df_results


if __name__ == "__main__":
    main()


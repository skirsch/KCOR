#!/usr/bin/env python3
"""
KCOR Mortality Sensitivity Analysis Automation

This script automates running the KCOR mortality pipeline with multiple
configurations to test robustness across:
- Different enrollment dates
- Different quiet periods
- Different follow-up horizons
- Different cohort definitions

USAGE:
    python kcor_mortality_sensitivity.py <input.csv> <output_dir> [--config-file CONFIG.yaml]

DEPENDENCIES:
    pip install pandas numpy matplotlib pyyaml
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import itertools
from datetime import datetime

# Import the core pipeline
from kcor_mortality import run_kcor_pipeline


def default_sensitivity_config() -> Dict:
    """Return default sensitivity analysis configuration."""
    return {
        "enrollment_dates": [
            {"year": 2021, "month": 1},
            {"year": 2021, "month": 7},
            {"year": 2022, "month": 1}
        ],
        "quiet_periods": [
            {"min": 3, "max": 10},
            {"min": 6, "max": 15},
            {"min": 9, "max": 18}
        ],
        "follow_up_months": [12, 18, 24],
        "cohort_definitions": [
            {"separate_doses": False},  # Group 3+ doses
            {"separate_doses": True}    # Separate 3,4,5,6
        ]
    }


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML config file (optional)
    
    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config
        except ImportError:
            print("Warning: pyyaml not installed, using default config")
            return default_sensitivity_config()
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}, using defaults")
            return default_sensitivity_config()
    else:
        return default_sensitivity_config()


def generate_configurations(config: Dict) -> List[Dict]:
    """
    Generate all combinations of sensitivity analysis configurations.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for enroll in config["enrollment_dates"]:
        for quiet in config["quiet_periods"]:
            for fu_months in config["follow_up_months"]:
                for cohort_def in config["cohort_definitions"]:
                    configs.append({
                        "enroll_year": enroll["year"],
                        "enroll_month": enroll["month"],
                        "quiet_t_min": quiet["min"],
                        "quiet_t_max": quiet["max"],
                        "max_fu_months": fu_months,
                        "separate_doses": cohort_def["separate_doses"]
                    })
    
    return configs


def run_sensitivity_analysis(
    csv_path: str,
    base_output_dir: str,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Run sensitivity analysis with multiple configurations.
    
    Args:
        csv_path: Path to input CSV file
        base_output_dir: Base output directory
        config: Configuration dictionary (if None, uses defaults)
    
    Returns:
        Summary DataFrame with all results
    """
    if config is None:
        config = default_sensitivity_config()
    
    configurations = generate_configurations(config)
    total_configs = len(configurations)
    
    print("=" * 80)
    print("KCOR Mortality Sensitivity Analysis")
    print("=" * 80)
    print(f"Total configurations to run: {total_configs}")
    print("=" * 80)
    
    # Create sensitivity output directory
    sensitivity_dir = os.path.join(base_output_dir, "sensitivity")
    os.makedirs(sensitivity_dir, exist_ok=True)
    
    results = []
    successful_runs = 0
    failed_runs = 0
    
    for idx, cfg in enumerate(configurations, 1):
        # Create unique identifier for this configuration
        cfg_id = (
            f"{cfg['enroll_year']}-{cfg['enroll_month']:02d}_"
            f"quiet{cfg['quiet_t_min']}-{cfg['quiet_t_max']}_"
            f"fu{cfg['max_fu_months']}_"
            f"{'separate' if cfg['separate_doses'] else 'grouped'}"
        )
        
        print(f"\n[{idx}/{total_configs}] Running configuration: {cfg_id}")
        print("-" * 80)
        
        # Create output subdirectory for this configuration
        cfg_output_dir = os.path.join(sensitivity_dir, cfg_id)
        
        try:
            # Run pipeline
            pipeline_results = run_kcor_pipeline(
                csv_path=csv_path,
                output_dir=cfg_output_dir,
                enroll_year=cfg["enroll_year"],
                enroll_month=cfg["enroll_month"],
                max_fu_months=cfg["max_fu_months"],
                quiet_t_min=cfg["quiet_t_min"],
                quiet_t_max=cfg["quiet_t_max"],
                separate_doses=cfg["separate_doses"]
            )
            
            # Extract key results
            kcor_df = pipeline_results["kcor"]
            
            # Get final KCOR values (at max follow-up time)
            for cohort in kcor_df["cohort"].unique():
                cohort_data = kcor_df[kcor_df["cohort"] == cohort]
                if len(cohort_data) > 0:
                    final_row = cohort_data.iloc[-1]
                    
                    results.append({
                        "config_id": cfg_id,
                        "enroll_year": cfg["enroll_year"],
                        "enroll_month": cfg["enroll_month"],
                        "quiet_t_min": cfg["quiet_t_min"],
                        "quiet_t_max": cfg["quiet_t_max"],
                        "max_fu_months": cfg["max_fu_months"],
                        "separate_doses": cfg["separate_doses"],
                        "cohort": cohort,
                        "final_t": final_row["t"],
                        "final_kcor": final_row["kcor_ratio"],
                        "max_kcor": cohort_data["kcor_ratio"].max(),
                        "min_kcor": cohort_data["kcor_ratio"].min(),
                        "mean_kcor": cohort_data["kcor_ratio"].mean()
                    })
            
            successful_runs += 1
            print(f"✓ Configuration {cfg_id} completed successfully")
            
        except Exception as e:
            failed_runs += 1
            print(f"✗ Configuration {cfg_id} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Save summary
    summary_path = os.path.join(sensitivity_dir, "summary_all_configs.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Saved summary to {summary_path}")
    print(f"Successful runs: {successful_runs}/{total_configs}")
    print(f"Failed runs: {failed_runs}/{total_configs}")
    print(f"{'=' * 80}")
    
    return summary_df


def analyze_sensitivity_results(summary_df: pd.DataFrame, output_dir: str):
    """
    Analyze sensitivity results and identify patterns.
    
    Args:
        summary_df: Summary DataFrame from sensitivity analysis
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("Sensitivity Analysis Summary")
    print("=" * 80)
    
    sensitivity_dir = os.path.join(output_dir, "sensitivity")
    
    # Group by cohort and compute statistics
    cohort_stats = []
    for cohort in summary_df["cohort"].unique():
        cohort_data = summary_df[summary_df["cohort"] == cohort]
        
        cohort_stats.append({
            "cohort": cohort,
            "n_configs": len(cohort_data),
            "mean_final_kcor": cohort_data["final_kcor"].mean(),
            "median_final_kcor": cohort_data["final_kcor"].median(),
            "std_final_kcor": cohort_data["final_kcor"].std(),
            "min_final_kcor": cohort_data["final_kcor"].min(),
            "max_final_kcor": cohort_data["final_kcor"].max(),
            "pct_above_1": (cohort_data["final_kcor"] > 1.0).mean() * 100,
            "pct_below_1": (cohort_data["final_kcor"] < 1.0).mean() * 100,
            "pct_near_1": ((cohort_data["final_kcor"] >= 0.95) & 
                          (cohort_data["final_kcor"] <= 1.05)).mean() * 100
        })
    
    stats_df = pd.DataFrame(cohort_stats)
    
    # Save statistics
    stats_path = os.path.join(sensitivity_dir, "cohort_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nCohort statistics saved to {stats_path}")
    print("\n" + stats_df.to_string(index=False))
    
    # Identify consistent patterns
    print("\n" + "=" * 80)
    print("Consistency Analysis")
    print("=" * 80)
    
    for cohort in summary_df["cohort"].unique():
        cohort_data = summary_df[summary_df["cohort"] == cohort]
        above_1 = (cohort_data["final_kcor"] > 1.0).sum()
        below_1 = (cohort_data["final_kcor"] < 1.0).sum()
        total = len(cohort_data)
        
        print(f"\n{cohort}:")
        print(f"  Configurations with KCOR > 1: {above_1}/{total} ({above_1/total*100:.1f}%)")
        print(f"  Configurations with KCOR < 1: {below_1}/{total} ({below_1/total*100:.1f}%)")
        
        if above_1 / total >= 0.8:
            print(f"  → CONSISTENT HARM (KCOR > 1 in ≥80% of configurations)")
        elif below_1 / total >= 0.8:
            print(f"  → CONSISTENT BENEFIT (KCOR < 1 in ≥80% of configurations)")
        elif abs(above_1 - below_1) / total < 0.2:
            print(f"  → MIXED/NEUTRAL (no clear pattern)")
        else:
            print(f"  → INCONSISTENT (results vary across configurations)")
    
    # Flag divergent configurations
    print("\n" + "=" * 80)
    print("Divergent Configurations")
    print("=" * 80)
    
    for cohort in summary_df["cohort"].unique():
        cohort_data = summary_df[summary_df["cohort"] == cohort]
        median_kcor = cohort_data["final_kcor"].median()
        
        # Find configs that deviate significantly from median
        threshold = 0.2  # 20% deviation
        divergent = cohort_data[
            (cohort_data["final_kcor"] > median_kcor * (1 + threshold)) |
            (cohort_data["final_kcor"] < median_kcor * (1 - threshold))
        ]
        
        if len(divergent) > 0:
            print(f"\n{cohort} (median KCOR: {median_kcor:.3f}):")
            for _, row in divergent.iterrows():
                print(f"  {row['config_id']}: KCOR = {row['final_kcor']:.3f}")


def create_sensitivity_plots(summary_df: pd.DataFrame, output_dir: str):
    """
    Create visualization plots for sensitivity analysis.
    
    Args:
        summary_df: Summary DataFrame
        output_dir: Output directory
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sensitivity_dir = os.path.join(output_dir, "sensitivity", "comparison_plots")
        os.makedirs(sensitivity_dir, exist_ok=True)
        
        # Plot 1: KCOR distribution by cohort
        plt.figure(figsize=(12, 6))
        cohorts = summary_df["cohort"].unique()
        for cohort in cohorts:
            cohort_data = summary_df[summary_df["cohort"] == cohort]
            plt.hist(cohort_data["final_kcor"], alpha=0.5, label=cohort, bins=20)
        
        plt.axvline(1.0, color="black", linestyle="--", linewidth=2, label="No effect (KCOR=1)")
        plt.xlabel("Final KCOR Ratio", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Distribution of Final KCOR Ratios Across All Configurations", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sensitivity_dir, "kcor_distribution.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Box plot by cohort
        plt.figure(figsize=(12, 6))
        summary_df.boxplot(column="final_kcor", by="cohort", ax=plt.gca())
        plt.axhline(1.0, color="red", linestyle="--", linewidth=2)
        plt.xlabel("Cohort", fontsize=12)
        plt.ylabel("Final KCOR Ratio", fontsize=12)
        plt.title("Final KCOR Ratios by Cohort (All Configurations)", fontsize=14)
        plt.suptitle("")  # Remove default title
        plt.tight_layout()
        plt.savefig(os.path.join(sensitivity_dir, "kcor_boxplot.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSensitivity plots saved to {sensitivity_dir}")
        
    except ImportError:
        print("Warning: matplotlib/seaborn not available, skipping plots")
    except Exception as e:
        print(f"Warning: Failed to create plots: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KCOR Mortality Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Base output directory")
    parser.add_argument("--config-file", help="Path to YAML configuration file (optional)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return 1
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Run sensitivity analysis
    try:
        summary_df = run_sensitivity_analysis(
            csv_path=args.input_csv,
            base_output_dir=args.output_dir,
            config=config
        )
        
        # Analyze results
        analyze_sensitivity_results(summary_df, args.output_dir)
        
        # Create plots
        create_sensitivity_plots(summary_df, args.output_dir)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


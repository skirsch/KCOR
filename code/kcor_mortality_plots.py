#!/usr/bin/env python3
"""
KCOR Mortality Enhanced Visualizations

This module provides comprehensive visualization functions for KCOR mortality analysis:
- Core plots (KCOR ratios, hazard curves, cumulative hazards)
- Comparison plots (across enrollment dates, age bands)
- Diagnostic plots (risk sets, events, slope fits)
- Sensitivity analysis visualizations

USAGE:
    python kcor_mortality_plots.py <hazard_csv> <output_dir> [--plot-type TYPE]

DEPENDENCIES:
    pip install pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


def plot_hazard_curves(
    haz: pd.DataFrame,
    output_path: str,
    plot_adjusted: bool = True,
    title: Optional[str] = None
):
    """
    Plot hazard curves (raw and/or adjusted) per cohort.
    
    Args:
        haz: Hazard DataFrame with columns: cohort, t, hazard, hazard_adj
        output_path: Path to save plot
        plot_adjusted: If True, plot adjusted hazards; else plot raw hazards
        title: Plot title (optional)
    """
    print(f"Creating hazard curve plot: {output_path}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cohorts = haz["cohort"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cohorts)))
    
    for i, cohort in enumerate(cohorts):
        cohort_data = haz[haz["cohort"] == cohort].sort_values("t")
        
        if plot_adjusted and "hazard_adj" in cohort_data.columns:
            y_values = cohort_data["hazard_adj"]
            label = f"{cohort} (adjusted)"
        else:
            y_values = cohort_data["hazard"]
            label = f"{cohort} (raw)"
        
        ax.plot(cohort_data["t"], y_values, label=label, color=colors[i], 
                marker='o', markersize=3, linewidth=2)
    
    ax.set_xlabel("Months since enrollment", fontsize=12)
    ax.set_ylabel("Hazard" + (" (adjusted)" if plot_adjusted else " (raw)"), fontsize=12)
    ax.set_title(title or "Hazard Curves by Cohort", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for hazards
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_cumulative_hazards(
    haz: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot cumulative hazard curves per cohort.
    
    Args:
        haz: Hazard DataFrame with columns: cohort, t, cum_hazard_adj
        output_path: Path to save plot
        title: Plot title (optional)
    """
    print(f"Creating cumulative hazard plot: {output_path}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cohorts = haz["cohort"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cohorts)))
    
    for i, cohort in enumerate(cohorts):
        cohort_data = haz[haz["cohort"] == cohort].sort_values("t")
        ax.plot(cohort_data["t"], cohort_data["cum_hazard_adj"], 
                label=cohort, color=colors[i], marker='o', markersize=3, linewidth=2)
    
    ax.set_xlabel("Months since enrollment", fontsize=12)
    ax.set_ylabel("Cumulative Hazard (adjusted)", fontsize=12)
    ax.set_title(title or "Cumulative Hazard Curves by Cohort", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_kcor_with_ci(
    kcor_df: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot KCOR ratios with confidence intervals.
    
    Args:
        kcor_df: KCOR DataFrame with columns: cohort, t, kcor_ratio, ci_lower, ci_upper
        output_path: Path to save plot
        title: Plot title (optional)
    """
    print(f"Creating KCOR plot with CI: {output_path}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cohorts = kcor_df["cohort"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cohorts)))
    
    for i, cohort in enumerate(cohorts):
        cohort_data = kcor_df[kcor_df["cohort"] == cohort].sort_values("t")
        
        # Plot confidence interval ribbon
        if "ci_lower" in cohort_data.columns and "ci_upper" in cohort_data.columns:
            ax.fill_between(cohort_data["t"], cohort_data["ci_lower"], cohort_data["ci_upper"],
                           alpha=0.2, color=colors[i], label=f"{cohort} (95% CI)")
        
        # Plot KCOR ratio
        ax.plot(cohort_data["t"], cohort_data["kcor_ratio"], 
               label=cohort, color=colors[i], marker='o', markersize=4, linewidth=2)
    
    ax.axhline(1.0, color="black", linestyle="--", linewidth=2, label="No effect (KCOR=1)")
    ax.set_xlabel("Months since enrollment", fontsize=12)
    ax.set_ylabel("KCOR adjusted cumulative hazard ratio", fontsize=12)
    ax.set_title(title or "KCOR: Vaccinated vs Unvaccinated (with 95% CI)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_slope_diagnostics(
    haz: pd.DataFrame,
    slopes_df: pd.DataFrame,
    quiet_t_min: int,
    quiet_t_max: int,
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot slope fit diagnostics (log-hazard vs time in quiet period).
    
    Args:
        haz: Hazard DataFrame with columns: cohort, t, log_hazard
        slopes_df: Slopes DataFrame with columns: cohort, intercept, slope
        quiet_t_min: Start of quiet period
        quiet_t_max: End of quiet period
        output_path: Path to save plot
        title: Plot title (optional)
    """
    print(f"Creating slope diagnostic plot: {output_path}")
    
    # Filter to quiet period
    quiet_data = haz[(haz["t"] >= quiet_t_min) & (haz["t"] <= quiet_t_max)].copy()
    
    n_cohorts = len(slopes_df)
    n_cols = min(3, n_cohorts)
    n_rows = (n_cohorts + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_cohorts == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (_, slope_row) in enumerate(slopes_df.iterrows()):
        cohort = slope_row["cohort"]
        intercept = slope_row["intercept"]
        slope = slope_row["slope"]
        
        ax = axes[idx]
        cohort_data = quiet_data[quiet_data["cohort"] == cohort].dropna(subset=["log_hazard"])
        
        # Plot data points
        ax.scatter(cohort_data["t"], cohort_data["log_hazard"], 
                  alpha=0.6, s=50, label="Data")
        
        # Plot fitted line
        if len(cohort_data) > 0:
            t_fit = np.linspace(quiet_t_min, quiet_t_max, 100)
            log_hazard_fit = intercept + slope * t_fit
            ax.plot(t_fit, log_hazard_fit, 'r-', linewidth=2, 
                   label=f"Fit (slope={slope:.4f})")
        
        ax.set_xlabel("Month (t)", fontsize=10)
        ax.set_ylabel("log(Hazard)", fontsize=10)
        ax.set_title(f"{cohort}\nSlope={slope:.4f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_cohorts, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title or "Slope Fit Diagnostics (Quiet Period)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_risk_sets(
    haz: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot risk set sizes over time per cohort.
    
    Args:
        haz: Hazard DataFrame with columns: cohort, t, at_risk
        output_path: Path to save plot
        title: Plot title (optional)
    """
    print(f"Creating risk set plot: {output_path}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cohorts = haz["cohort"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cohorts)))
    
    for i, cohort in enumerate(cohorts):
        cohort_data = haz[haz["cohort"] == cohort].sort_values("t")
        ax.plot(cohort_data["t"], cohort_data["at_risk"], 
               label=cohort, color=colors[i], marker='o', markersize=3, linewidth=2)
    
    ax.set_xlabel("Months since enrollment", fontsize=12)
    ax.set_ylabel("Persons at Risk", fontsize=12)
    ax.set_title(title or "Risk Set Sizes Over Time", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_event_counts(
    haz: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot death event counts per month per cohort.
    
    Args:
        haz: Hazard DataFrame with columns: cohort, t, deaths
        output_path: Path to save plot
        title: Plot title (optional)
    """
    print(f"Creating event count plot: {output_path}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cohorts = haz["cohort"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cohorts)))
    
    x_pos = np.arange(len(haz["t"].unique()))
    width = 0.8 / len(cohorts)
    
    for i, cohort in enumerate(cohorts):
        cohort_data = haz[haz["cohort"] == cohort].sort_values("t")
        ax.bar(x_pos + i*width, cohort_data["deaths"], width, 
              label=cohort, color=colors[i], alpha=0.7)
    
    ax.set_xlabel("Month (t)", fontsize=12)
    ax.set_ylabel("Deaths", fontsize=12)
    ax.set_title(title or "Death Events Per Month by Cohort", fontsize=14)
    ax.set_xticks(x_pos + width * (len(cohorts) - 1) / 2)
    ax.set_xticklabels([int(t) for t in sorted(haz["t"].unique())])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_sensitivity_heatmap(
    summary_df: pd.DataFrame,
    output_path: str,
    cohort: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Create heatmap of KCOR values across sensitivity configurations.
    
    Args:
        summary_df: Summary DataFrame from sensitivity analysis
        output_path: Path to save plot
        cohort: Specific cohort to plot (if None, plots all)
        title: Plot title (optional)
    """
    print(f"Creating sensitivity heatmap: {output_path}")
    
    if cohort is not None:
        plot_data = summary_df[summary_df["cohort"] == cohort].copy()
    else:
        plot_data = summary_df.copy()
    
    if len(plot_data) == 0:
        print("Warning: No data for heatmap")
        return
    
    # Create pivot table: enrollment date vs quiet period
    pivot = plot_data.pivot_table(
        values="final_kcor",
        index=["enroll_year", "enroll_month"],
        columns=["quiet_t_min", "quiet_t_max"],
        aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", 
                center=1.0, vmin=0.5, vmax=2.0, ax=ax,
                cbar_kws={"label": "Final KCOR Ratio"})
    
    ax.set_xlabel("Quiet Period", fontsize=12)
    ax.set_ylabel("Enrollment Date", fontsize=12)
    ax.set_title(title or f"Sensitivity Heatmap: KCOR Across Configurations" + 
                (f" ({cohort})" if cohort else ""), fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_age_stratified_comparison(
    results_dict: Dict[str, pd.DataFrame],
    output_path: str,
    title: Optional[str] = None
):
    """
    Plot KCOR ratios across age bands for comparison.
    
    Args:
        results_dict: Dictionary mapping age_band to kcor DataFrame
        output_path: Path to save plot
        title: Plot title (optional)
    """
    print(f"Creating age-stratified comparison plot: {output_path}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for i, (age_band, kcor_df) in enumerate(results_dict.items()):
        for cohort in kcor_df["cohort"].unique():
            cohort_data = kcor_df[kcor_df["cohort"] == cohort].sort_values("t")
            ax.plot(cohort_data["t"], cohort_data["kcor_ratio"],
                   label=f"{cohort} ({age_band})", color=colors[i], 
                   marker='o', markersize=3, linewidth=2, alpha=0.7)
    
    ax.axhline(1.0, color="black", linestyle="--", linewidth=2, label="No effect (KCOR=1)")
    ax.set_xlabel("Months since enrollment", fontsize=12)
    ax.set_ylabel("KCOR adjusted cumulative hazard ratio", fontsize=12)
    ax.set_title(title or "KCOR Ratios by Age Band", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def create_all_diagnostic_plots(
    haz_path: str,
    slopes_path: str,
    output_dir: str,
    quiet_t_min: int = 3,
    quiet_t_max: int = 10,
    title_prefix: str = ""
):
    """
    Create all diagnostic plots from hazard and slopes files.
    
    Args:
        haz_path: Path to kcor_hazard_adjusted.csv
        slopes_path: Path to kcor_slopes.csv
        output_dir: Output directory for plots
        quiet_t_min: Start of quiet period
        quiet_t_max: End of quiet period
        title_prefix: Prefix for plot titles
    """
    print(f"Creating all diagnostic plots in {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    haz = pd.read_csv(haz_path)
    slopes_df = pd.read_csv(slopes_path)
    
    # Create plots
    plot_hazard_curves(haz, os.path.join(output_dir, "hazard_curves_raw.png"), 
                      plot_adjusted=False, title=f"{title_prefix}Hazard Curves (Raw)")
    plot_hazard_curves(haz, os.path.join(output_dir, "hazard_curves_adjusted.png"), 
                      plot_adjusted=True, title=f"{title_prefix}Hazard Curves (Adjusted)")
    plot_cumulative_hazards(haz, os.path.join(output_dir, "cumulative_hazards.png"),
                           title=f"{title_prefix}Cumulative Hazards")
    plot_slope_diagnostics(haz, slopes_df, quiet_t_min, quiet_t_max,
                          os.path.join(output_dir, "slope_diagnostics.png"),
                          title=f"{title_prefix}Slope Fit Diagnostics")
    plot_risk_sets(haz, os.path.join(output_dir, "risk_sets.png"),
                  title=f"{title_prefix}Risk Set Sizes")
    plot_event_counts(haz, os.path.join(output_dir, "event_counts.png"),
                     title=f"{title_prefix}Event Counts")
    
    print(f"All diagnostic plots saved to {output_dir}")


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="KCOR Mortality Enhanced Visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("hazard_csv", help="Path to kcor_hazard_adjusted.csv")
    parser.add_argument("output_dir", help="Output directory for plots")
    parser.add_argument("--slopes-csv", help="Path to kcor_slopes.csv (for diagnostics)")
    parser.add_argument("--kcor-csv", help="Path to kcor_ratios.csv (for KCOR plots)")
    parser.add_argument("--quiet-min", type=int, default=3, help="Quiet period start")
    parser.add_argument("--quiet-max", type=int, default=10, help="Quiet period end")
    parser.add_argument("--plot-type", choices=["all", "hazards", "kcor", "diagnostics"],
                       default="all", help="Type of plots to create")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hazard_csv):
        print(f"Error: Input file not found: {args.hazard_csv}")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        haz = pd.read_csv(args.hazard_csv)
        
        if args.plot_type in ["all", "hazards", "diagnostics"]:
            if args.slopes_csv and os.path.exists(args.slopes_csv):
                slopes_df = pd.read_csv(args.slopes_csv)
                plot_slope_diagnostics(haz, slopes_df, args.quiet_min, args.quiet_max,
                                      os.path.join(args.output_dir, "slope_diagnostics.png"))
            
            plot_hazard_curves(haz, os.path.join(args.output_dir, "hazard_curves.png"))
            plot_cumulative_hazards(haz, os.path.join(args.output_dir, "cumulative_hazards.png"))
            plot_risk_sets(haz, os.path.join(args.output_dir, "risk_sets.png"))
            plot_event_counts(haz, os.path.join(args.output_dir, "event_counts.png"))
        
        if args.plot_type in ["all", "kcor"] and args.kcor_csv and os.path.exists(args.kcor_csv):
            kcor_df = pd.read_csv(args.kcor_csv)
            plot_kcor_with_ci(kcor_df, os.path.join(args.output_dir, "kcor_with_ci.png"))
        
        print(f"\nPlots saved to {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


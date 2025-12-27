#!/usr/bin/env python3
"""
KCOR Simulation Grid Figure Generator

Generates publication-ready figures for the simulation grid:
1. fig_sim_grid_overview.png - KCOR(t) trajectories across scenarios
2. fig_sim_grid_diagnostics.png - Diagnostic summaries

Usage:
    python plot_sim_grid.py --input-results RESULTS.xlsx --input-diagnostics DIAG.csv \
                            --output-overview FIG1.png --output-diagnostics FIG2.png
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Use Agg backend for headless operation (must be before pyplot import)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================================
# Plot Configuration
# ============================================================================

# Use a clean, publication-ready style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Color palette for scenarios
SCENARIO_COLORS = {
    'gamma_null': '#2E86AB',        # Blue
    'hazard_increase': '#E94F37',   # Red
    'hazard_decrease': '#44AF69',   # Green
    'nongamma_null': '#F18F01',     # Orange
    'contamination': '#A23B72',     # Purple
    'sparse': '#6C757D',            # Gray
}

SCENARIO_LABELS = {
    'gamma_null': 'Gamma-Frailty Null',
    'hazard_increase': 'Injected Hazard Increase',
    'hazard_decrease': 'Injected Hazard Decrease',
    'nongamma_null': 'Non-Gamma Frailty',
    'contamination': 'Quiet-Window Contamination',
    'sparse': 'Sparse Events',
}

# Effect window for shading
EFFECT_WINDOW = (20, 80)
QUIET_WINDOW = (20, 80)


# ============================================================================
# Figure 1: KCOR Trajectory Overview
# ============================================================================

def plot_kcor_overview(
    results_path: str,
    output_path: str,
    dpi: int = 300,
):
    """
    Generate 2x3 panel figure showing KCOR(t) trajectories.
    
    Args:
        results_path: Path to sim_grid_results.xlsx
        output_path: Output path for PNG
        dpi: Resolution
    """
    # Load results
    xls = pd.ExcelFile(results_path)
    
    # Define scenario order for 2x3 grid
    scenarios = [
        'gamma_null',
        'hazard_increase', 
        'hazard_decrease',
        'nongamma_null',
        'contamination',
        'sparse',
    ]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        # Load KCOR trajectory
        sheet_name = f"{scenario[:25]}_kcor"
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception:
            # Try alternative naming
            matching = [s for s in xls.sheet_names if scenario[:10] in s.lower() and 'kcor' in s.lower()]
            if matching:
                df = pd.read_excel(xls, sheet_name=matching[0])
            else:
                ax.text(0.5, 0.5, f"No data for\n{scenario}", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
                continue
        
        weeks = df['Week'].values
        kcor = df['KCOR'].values
        
        # Plot KCOR trajectory
        color = SCENARIO_COLORS.get(scenario, '#333333')
        ax.plot(weeks, kcor, color=color, linewidth=1.5, label='KCOR(t)')
        
        # Reference line at KCOR = 1
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='KCOR = 1')
        
        # Add ±5% band for null scenarios
        if scenario in ['gamma_null', 'nongamma_null', 'contamination', 'sparse']:
            ax.axhspan(0.95, 1.05, alpha=0.1, color='green', label='±5% band')
        
        # Shade effect window for effect scenarios
        if scenario in ['hazard_increase', 'hazard_decrease']:
            ax.axvspan(EFFECT_WINDOW[0], EFFECT_WINDOW[1], alpha=0.15, color=color, label='Effect window')
        
        # Shade quiet window for contamination scenario
        if scenario == 'contamination':
            ax.axvspan(30, 50, alpha=0.2, color='red', label='Shock window')
        
        # Formatting
        ax.set_xlim(0, 120)
        ax.set_ylim(0.7, 1.5)
        ax.set_xlabel('Weeks since cohort entry')
        ax.set_ylabel('KCOR(t)')
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontweight='bold')
        
        # Add scenario number
        ax.text(0.02, 0.98, f"({idx+1})", transform=ax.transAxes, fontsize=10, 
                fontweight='bold', va='top', ha='left')
        
        # Compute and display median KCOR
        diagnostic_mask = (weeks >= 20) & (weeks <= 100)
        kcor_diagnostic = kcor[diagnostic_mask]
        kcor_diagnostic = kcor_diagnostic[np.isfinite(kcor_diagnostic)]
        if len(kcor_diagnostic) > 0:
            median_kcor = np.median(kcor_diagnostic)
            ax.text(0.98, 0.02, f"Median: {median_kcor:.3f}", transform=ax.transAxes,
                    fontsize=8, va='bottom', ha='right', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title
    fig.suptitle('KCOR Simulation Grid: Operating Characteristics', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved KCOR overview figure: {output_path}")


# ============================================================================
# Figure 2: Diagnostic Summary
# ============================================================================

def plot_diagnostics(
    diagnostics_path: str,
    output_path: str,
    dpi: int = 300,
):
    """
    Generate diagnostic summary figure with 3 panels:
    (i) RMSE by scenario
    (ii) θ̂ by scenario  
    (iii) R² by scenario
    
    Args:
        diagnostics_path: Path to sim_grid_diagnostics.csv
        output_path: Output path for PNG
        dpi: Resolution
    """
    # Load diagnostics
    df = pd.read_csv(diagnostics_path)
    
    # Define scenario order
    scenario_order = [
        'gamma_null',
        'hazard_increase',
        'hazard_decrease',
        'nongamma_null',
        'contamination',
        'sparse',
    ]
    
    # Prepare data - aggregate by scenario (mean across doses)
    summary = df.groupby('scenario').agg({
        'rmse': 'mean',
        'theta_hat': 'mean',
        'r_squared': 'mean',
    }).reset_index()
    
    # Reorder
    summary['order'] = summary['scenario'].map({s: i for i, s in enumerate(scenario_order)})
    summary = summary.sort_values('order')
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    # Get colors and labels
    colors = [SCENARIO_COLORS.get(s, '#333333') for s in summary['scenario']]
    labels = [SCENARIO_LABELS.get(s, s) for s in summary['scenario']]
    x = np.arange(len(summary))
    
    # Panel (i): RMSE
    ax1 = axes[0]
    bars1 = ax1.bar(x, summary['rmse'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('RMSE (Cumulative Hazard)')
    ax1.set_title('(i) Fit Error', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"S{i+1}" for i in x], fontsize=9)
    ax1.set_xlabel('Scenario')
    
    # Add threshold line for "good" fit
    ax1.axhline(y=0.01, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Good fit threshold')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Panel (ii): θ̂
    ax2 = axes[1]
    bars2 = ax2.bar(x, summary['theta_hat'], color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Fitted θ̂ (Frailty Variance)')
    ax2.set_title('(ii) Estimated Frailty', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{i+1}" for i in x], fontsize=9)
    ax2.set_xlabel('Scenario')
    
    # Panel (iii): R²
    ax3 = axes[2]
    bars3 = ax3.bar(x, summary['r_squared'], color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('R² (Post-Normalization Linearity)')
    ax3.set_title('(iii) Linearity', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"S{i+1}" for i in x], fontsize=9)
    ax3.set_xlabel('Scenario')
    ax3.set_ylim(0, 1.05)
    
    # Add threshold line for "good" linearity
    ax3.axhline(y=0.99, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Good linearity')
    ax3.legend(loc='lower right', fontsize=8)
    
    # Add legend for scenario colors at bottom
    legend_patches = [mpatches.Patch(color=SCENARIO_COLORS.get(s, '#333333'), 
                                     label=f"S{i+1}: {SCENARIO_LABELS.get(s, s)[:25]}") 
                      for i, s in enumerate(scenario_order)]
    
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.02), frameon=True)
    
    # Add overall title
    fig.suptitle('KCOR Simulation Grid: Diagnostic Summary', fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.98])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved diagnostics figure: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate KCOR simulation grid figures"
    )
    parser.add_argument(
        "--input-results", "-r",
        default="test/sim_grid/out/sim_grid_results.xlsx",
        help="Path to sim_grid_results.xlsx"
    )
    parser.add_argument(
        "--input-diagnostics", "-d",
        default="test/sim_grid/out/sim_grid_diagnostics.csv",
        help="Path to sim_grid_diagnostics.csv"
    )
    parser.add_argument(
        "--output-overview", "-o",
        default="test/sim_grid/out/fig_sim_grid_overview.png",
        help="Output path for overview figure"
    )
    parser.add_argument(
        "--output-diagnostics", "-g",
        default="test/sim_grid/out/fig_sim_grid_diagnostics.png",
        help="Output path for diagnostics figure"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure resolution"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KCOR Simulation Grid Figure Generator")
    print("=" * 60)
    
    # Generate Figure 1: KCOR Overview
    print("\nGenerating Figure 1: KCOR Trajectory Overview...")
    plot_kcor_overview(
        args.input_results,
        args.output_overview,
        args.dpi,
    )
    
    # Generate Figure 2: Diagnostics
    print("\nGenerating Figure 2: Diagnostic Summary...")
    plot_diagnostics(
        args.input_diagnostics,
        args.output_diagnostics,
        args.dpi,
    )
    
    print("\nFigure generation complete!")


if __name__ == "__main__":
    main()


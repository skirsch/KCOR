#!/usr/bin/env python3
"""
S7 Simulation Figure Generator

Generates publication-ready figures for S7 simulation results:
1. fig_s7_overview.png - KCOR(t) trajectories showing quiet window and effect window
2. fig_s7_diagnostics.png - Diagnostic summaries

Usage:
    python plot_s7_sim.py --input-results RESULTS.xlsx --input-diagnostics DIAG.csv \
                          --output-overview FIG1.png --output-diagnostics FIG2.png
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Use Agg backend for headless operation
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Publication-ready style
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


def plot_s7_overview(
    results_path: str,
    output_path: str,
    dpi: int = 300,
):
    """
    Generate figure showing S7 KCOR trajectories with quiet and effect windows.
    
    Shows key scenarios: step harm (early), step benefit (early), overlap variant.
    """
    # Load results
    df = pd.read_excel(results_path, sheet_name='S7_results')
    df_diag = pd.read_csv(results_path.replace('_results.xlsx', '_diagnostics.csv'))
    
    # Select key scenarios to plot
    key_scenarios = [
        'S7_step_r1.2_early',  # Harm, early
        'S7_step_r0.8_early',  # Benefit, early
        'S7_overlap_step_r1.2',  # Overlap variant
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, scenario in enumerate(key_scenarios):
        ax = axes[idx]
        df_scen = df[df['scenario'] == scenario].copy()
        
        if len(df_scen) == 0:
            continue
        
        # Get window info from diagnostics
        diag_row = df_diag[df_diag['scenario'] == scenario].iloc[0]
        quiet_win = [int(x) for x in diag_row['quiet_window'].split('-')]
        effect_win = [int(x) for x in diag_row['effect_window'].split('-')]
        
        # Plot KCOR trajectory
        weeks = df_scen['week'].values
        kcor = df_scen['KCOR'].values
        
        ax.plot(weeks, kcor, 'b-', linewidth=1.5, label='KCOR(t)')
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Shade quiet window
        ax.axvspan(quiet_win[0], quiet_win[1], alpha=0.2, color='green', label='Quiet window')
        
        # Shade effect window
        if not diag_row['overlap_variant']:
            ax.axvspan(effect_win[0], effect_win[1], alpha=0.2, color='red', label='Effect window')
        
        ax.set_xlabel('Week since enrollment')
        ax.set_ylabel('KCOR(t)')
        ax.set_title(scenario.replace('_', ' ').replace('S7 ', ''))
        ax.set_ylim([0.7, 1.3])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Generated S7 overview figure: {output_path}", file=sys.stderr)


def plot_s7_diagnostics(
    diagnostics_path: str,
    output_path: str,
    dpi: int = 300,
):
    """
    Generate diagnostic summary figure for S7 scenarios.
    """
    df = pd.read_csv(diagnostics_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Fitted theta values
    ax = axes[0, 0]
    scenarios = df['scenario'].values
    theta0 = df['theta0_hat'].values
    theta1 = df['theta1_hat'].values
    
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, theta0, width, label='Cohort 0 (θ₀)', alpha=0.7)
    ax.bar(x + width/2, theta1, width, label='Cohort 1 (θ₁)', alpha=0.7)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Fitted θ')
    ax.set_title('Fitted Frailty Variance')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: RMSE
    ax = axes[0, 1]
    rmse0 = df['rmse0'].values
    rmse1 = df['rmse1'].values
    ax.bar(x - width/2, rmse0, width, label='Cohort 0', alpha=0.7)
    ax.bar(x + width/2, rmse1, width, label='Cohort 1', alpha=0.7)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('RMSE (H-space)')
    ax.set_title('Fit Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Success flags
    ax = axes[1, 0]
    success0 = df['success0'].astype(int).values
    success1 = df['success1'].astype(int).values
    ax.bar(x - width/2, success0, width, label='Cohort 0', alpha=0.7)
    ax.bar(x + width/2, success1, width, label='Cohort 1', alpha=0.7)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Fit Success (1=yes)')
    ax.set_title('Fit Convergence')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=7)
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Effect multiplier vs expected KCOR deviation
    ax = axes[1, 1]
    effect_r = df['effect_multiplier'].values
    # For non-overlap scenarios, compute median KCOR during effect window
    # (simplified - would need full results to compute properly)
    ax.scatter(effect_r, np.ones_like(effect_r), alpha=0.6, s=50)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Effect Multiplier r')
    ax.set_ylabel('Expected KCOR')
    ax.set_title('Effect Detection')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Generated S7 diagnostics figure: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate S7 simulation figures")
    parser.add_argument("--input-results", required=True, help="Input Excel file with results")
    parser.add_argument("--input-diagnostics", required=True, help="Input CSV file with diagnostics")
    parser.add_argument("--output-overview", required=True, help="Output PNG for overview figure")
    parser.add_argument("--output-diagnostics", required=True, help="Output PNG for diagnostics figure")
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution")
    
    args = parser.parse_args()
    
    plot_s7_overview(args.input_results, args.output_overview, args.dpi)
    plot_s7_diagnostics(args.input_diagnostics, args.output_diagnostics, args.dpi)


if __name__ == "__main__":
    main()


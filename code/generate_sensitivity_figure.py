"""
Generate sensitivity analysis overview figure for KCOR methods paper.

This script reads the sensitivity analysis output (KCOR_SA.xlsx) and generates
a figure showing the distribution of KCOR values across the parameter grid.

Output: Heatmap or distribution plot showing KCOR robustness to parameter choices.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_sensitivity_output(xlsx_path: str) -> dict:
    """
    Read sensitivity analysis Excel file and extract KCOR grids.
    
    The SA output has sheets named like '2021_24_1_vs_0', '2021_24_2_vs_0', etc.
    Each sheet contains a grid of KCOR values with:
    - Rows: baseline weeks (normalization week count)
    - Columns: quiet-start offsets (weeks from 2022-24)
    """
    xls = pd.ExcelFile(xlsx_path)
    
    results = {}
    
    for sheet_name in xls.sheet_names:
        # Parse sheet name to get cohort and dose pair
        parts = sheet_name.split('_')
        if len(parts) < 4:
            continue
        
        # Expected format: YYYY_WW_D1_vs_D2
        try:
            cohort = f"{parts[0]}_{parts[1]}"
            dose_num = int(parts[2])
            dose_den = int(parts[4]) if parts[3] == 'vs' else None
        except (ValueError, IndexError):
            continue
        
        if dose_den is None:
            continue
        
        df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
        
        # Store the grid
        key = f"{cohort}: Dose {dose_num} vs {dose_den}"
        results[key] = {
            'df': df,
            'cohort': cohort,
            'dose_num': dose_num,
            'dose_den': dose_den,
            'sheet_name': sheet_name,
        }
    
    return results


def plot_sensitivity_heatmaps(
    results: dict,
    output_path: str,
) -> None:
    """
    Create heatmap figure showing KCOR values across parameter grid.
    """
    n_grids = len(results)
    if n_grids == 0:
        print("No sensitivity data to plot")
        return
    
    # Determine figure layout
    n_cols = min(3, n_grids)
    n_rows = (n_grids + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    
    if n_grids == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Compute unified color scale across all panels
    all_values = []
    for key, data in results.items():
        df = data['df']
        values = df.values.flatten()
        values = values[~np.isnan(values)]
        all_values.extend(values)
    
    if len(all_values) > 0:
        global_vmin = min(all_values)
        global_vmax = max(all_values)
        # Center colormap at 1.0, symmetric around 1.0
        if global_vmin < 1.0 < global_vmax:
            max_dev = max(abs(global_vmin - 1.0), abs(global_vmax - 1.0))
            unified_vmin = 1.0 - max_dev
            unified_vmax = 1.0 + max_dev
        else:
            unified_vmin = global_vmin
            unified_vmax = global_vmax
    else:
        unified_vmin = 0.95
        unified_vmax = 1.05
    
    # Plot heatmaps with unified color scale
    for idx, (key, data) in enumerate(results.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        df = data['df']
        
        # Use unified color scale for all panels
        # Only show colorbar on the last panel
        cbar = (idx == n_grids - 1)
        
        sns.heatmap(
            df, 
            ax=ax, 
            cmap='RdBu_r',
            center=1.0,
            vmin=unified_vmin,
            vmax=unified_vmax,
            annot=True,
            fmt='.3f',
            annot_kws={'size': 10},
            cbar=cbar,
            cbar_kws={'label': 'KCOR(t) value', 'shrink': 0.8} if cbar else None,
        )
        
        ax.set_title(key, fontsize=12, fontweight='bold')
        ax.set_xlabel('Quiet-window start offset (weeks)', fontsize=12)
        ax.set_ylabel('Baseline weeks', fontsize=12)
        ax.tick_params(labelsize=10)
        
        # Simplify x-axis labels: convert "offset_-12w" format to "-12, -8, -4, 0, +4, +8, +12"
        # Get current tick labels and convert them
        x_tick_labels = ax.get_xticklabels()
        new_x_labels = []
        for label in x_tick_labels:
            text = label.get_text()
            # Extract numeric value from column names like "offset_-12w"
            if 'offset_' in text and 'w' in text:
                # Extract number from "offset_-12w" format
                num_str = text.replace('offset_', '').replace('w', '')
                try:
                    num = int(num_str)
                    # Format: negative numbers as "-12", zero as "0", positive as "+4"
                    if num > 0:
                        new_x_labels.append(f'+{num}')
                    else:
                        new_x_labels.append(str(num))
                except ValueError:
                    new_x_labels.append(text)
            else:
                # Already in a different format, try to parse as number
                try:
                    num = int(text.replace('+', ''))
                    if num > 0:
                        new_x_labels.append(f'+{num}')
                    else:
                        new_x_labels.append(str(num))
                except ValueError:
                    new_x_labels.append(text)
        
        ax.set_xticklabels(new_x_labels)
    
    # Hide unused axes
    for idx in range(n_grids, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add legend explaining axes (add as text annotation)
    # Place legend in the first subplot's corner
    if n_grids > 0:
        first_ax = axes[0, 0] if n_rows > 1 else axes[0]
        legend_text = (
            'X-axis: Quiet-window start offset (weeks)\n'
            'Y-axis: Baseline weeks\n'
            'Color: KCOR(t) value (centered at 1.0)'
        )
        first_ax.text(0.02, 0.98, legend_text, transform=first_ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle('Sensitivity Analysis: KCOR Robustness to Parameter Choices', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved figure: {output_path}")


def plot_sensitivity_distribution(
    results: dict,
    output_path: str,
) -> None:
    """
    Create distribution plot showing KCOR values across all parameter combinations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    all_values = []
    labels = []
    
    for key, data in results.items():
        df = data['df']
        values = df.values.flatten()
        values = values[~np.isnan(values)]
        all_values.append(values)
        labels.append(key)
    
    # Left panel: Box plots
    ax = axes[0]
    ax.boxplot(all_values, labels=[l.split(':')[1].strip() for l in labels], vert=True)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='KCOR = 1')
    ax.set_ylabel('KCOR')
    ax.set_xlabel('Dose comparison')
    ax.set_title('Distribution of KCOR across parameter grid')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right panel: Histogram of all values combined
    ax = axes[1]
    all_flat = np.concatenate(all_values)
    ax.hist(all_flat, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='KCOR = 1')
    ax.axvline(np.mean(all_flat), color='blue', linestyle='-', linewidth=2, 
               label=f'Mean = {np.mean(all_flat):.3f}')
    ax.set_xlabel('KCOR')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall KCOR distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    stats_text = (
        f"N = {len(all_flat)}\n"
        f"Mean = {np.mean(all_flat):.4f}\n"
        f"Std = {np.std(all_flat):.4f}\n"
        f"Min = {np.min(all_flat):.4f}\n"
        f"Max = {np.max(all_flat):.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Sensitivity Analysis: KCOR Robustness', fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity analysis overview figure"
    )
    parser.add_argument(
        "--input", "-i",
        default="test/sensitivity/out/KCOR_SA.xlsx",
        help="Input sensitivity analysis Excel file"
    )
    parser.add_argument(
        "--output", "-o",
        default="documentation/preprint/figures/fig_sensitivity_overview.png",
        help="Output figure path"
    )
    parser.add_argument(
        "--style", choices=['heatmap', 'distribution', 'both'],
        default='both',
        help="Figure style: heatmap, distribution, or both"
    )
    
    args = parser.parse_args()
    
    # Read sensitivity data
    results = read_sensitivity_output(args.input)
    
    if not results:
        print(f"Error: No valid sensitivity data found in {args.input}")
        return 1
    
    print(f"Found {len(results)} dose comparisons in sensitivity data")
    
    # Generate figures
    output_base = Path(args.output)
    
    if args.style in ['heatmap', 'both']:
        heatmap_path = str(output_base.with_suffix('.heatmap.png')) if args.style == 'both' else args.output
        plot_sensitivity_heatmaps(results, heatmap_path)
    
    if args.style in ['distribution', 'both']:
        dist_path = str(output_base.with_suffix('.dist.png')) if args.style == 'both' else args.output
        plot_sensitivity_distribution(results, dist_path)
    
    if args.style == 'both':
        # Also create a combined figure
        plot_sensitivity_heatmaps(results, args.output)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
Plot positive control results showing KCOR deviation under injected effects.

This script reads KCOR processed output and generates a figure showing:
- Top panel: Pre-normalization hazard curves for control and treatment cohorts
- Bottom panel: KCOR(t) showing deviation from 1 under injected effect

Output: Figure for the methods paper showing positive control validation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_kcor_output(xlsx_path: str) -> dict:
    """
    Read KCOR processed output and extract relevant data.
    
    Returns dict with scenario data including KCOR time series.
    """
    xls = pd.ExcelFile(xlsx_path)
    
    results = {}
    
    for sheet_name in xls.sheet_names:
        if '_summary' in sheet_name:
            continue
        
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Determine scenario from sheet name
        if 'harm' in sheet_name.lower():
            scenario = 'harm'
        elif 'benefit' in sheet_name.lower():
            scenario = 'benefit'
        else:
            scenario = sheet_name
        
        # Extract KCOR data for all-ages cohort (YoB=-2 or aggregate)
        # Look for key columns
        required_cols = ['t', 'KCOR', 'Dose']
        if not all(c in df.columns for c in ['t', 'Dose']):
            # Try alternative column names
            if 'EventTime' in df.columns:
                df['t'] = df['EventTime']
        
        if 't' not in df.columns:
            continue
            
        # Get KCOR column (may be named differently)
        kcor_col = None
        for col in ['KCOR', 'kcor', 'KCOR_ratio']:
            if col in df.columns:
                kcor_col = col
                break
        
        if kcor_col is None:
            # Try to compute from cumulative hazards if available
            if 'CH_adj' in df.columns or 'cumhaz_adj' in df.columns:
                ch_col = 'CH_adj' if 'CH_adj' in df.columns else 'cumhaz_adj'
                # Group by Dose and compute ratio
                continue
        
        results[scenario] = {
            'df': df,
            'sheet_name': sheet_name,
        }
    
    return results


def compute_kcor_from_raw_data(xlsx_path: str) -> dict:
    """
    Compute KCOR directly from raw hazard data if processed output is not available.
    
    This function reads the positive control Excel file and computes KCOR
    using simple cumulative hazard ratios.
    """
    xls = pd.ExcelFile(xlsx_path)
    
    results = {}
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        if 'Dose' not in df.columns:
            continue
        
        # Determine scenario from sheet name
        if 'harm' in sheet_name.lower():
            scenario = 'harm'
            expected_direction = '> 1'
            multiplier = 1.2
        elif 'benefit' in sheet_name.lower():
            scenario = 'benefit'
            expected_direction = '< 1'
            multiplier = 0.8
        else:
            continue
        
        # Aggregate by week and dose (sum across sex)
        agg = df.groupby(['ISOweekDied', 'Dose']).agg({
            'Alive': 'sum',
            'Dead': 'sum'
        }).reset_index()
        
        # Compute weekly hazard: h = -log(1 - Dead/Alive)
        agg['hazard'] = -np.log(1 - agg['Dead'] / agg['Alive'].clip(lower=1))
        agg['hazard'] = agg['hazard'].clip(lower=0)  # Ensure non-negative
        
        # Get unique weeks in order
        weeks = agg['ISOweekDied'].unique()
        week_to_idx = {w: i for i, w in enumerate(weeks)}
        agg['week_idx'] = agg['ISOweekDied'].map(week_to_idx)
        
        # Separate by dose
        dose0 = agg[agg['Dose'] == 0].sort_values('week_idx')
        dose1 = agg[agg['Dose'] == 1].sort_values('week_idx')
        
        # Compute cumulative hazards
        ch0 = dose0['hazard'].cumsum().values
        ch1 = dose1['hazard'].cumsum().values
        
        # Compute KCOR = CH1 / CH0 (treatment vs control)
        # Avoid division by zero
        kcor = np.where(ch0 > 0, ch1 / ch0, np.nan)
        
        # Normalize to 1 at baseline (first valid point after skip)
        skip_weeks = 4
        if len(kcor) > skip_weeks and np.isfinite(kcor[skip_weeks]):
            kcor = kcor / kcor[skip_weeks]
        
        results[scenario] = {
            'weeks': np.arange(len(weeks)),
            'kcor': kcor,
            'hazard_dose0': dose0['hazard'].values,
            'hazard_dose1': dose1['hazard'].values,
            'ch_dose0': ch0,
            'ch_dose1': ch1,
            'expected_direction': expected_direction,
            'multiplier': multiplier,
            'injection_window': (20, 80),
        }
    
    return results


def plot_positive_control(
    results: dict,
    output_path: str,
    injection_window: tuple = (20, 80),
) -> None:
    """
    Create positive control figure showing KCOR deviation under injected effect.
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Use 2x2 grid: left column for harm, right column for benefit
    # Top row: hazards, bottom row: KCOR
    
    scenarios = ['harm', 'benefit']
    colors = {'harm': 'tab:red', 'benefit': 'tab:blue'}
    
    for i, scenario in enumerate(scenarios):
        if scenario not in results:
            continue
        
        data = results[scenario]
        weeks = data['weeks']
        kcor = data['kcor']
        
        # Top panel: Hazard curves
        ax_top = fig.add_subplot(2, 2, i + 1)
        ax_top.plot(weeks, data['hazard_dose0'], label='Dose 0 (control)', 
                    linewidth=2, color='tab:gray')
        ax_top.plot(weeks, data['hazard_dose1'], label='Dose 1 (treatment)', 
                    linewidth=2, color=colors[scenario])
        
        # Shade injection window
        ax_top.axvspan(injection_window[0], injection_window[1], 
                       alpha=0.15, color=colors[scenario],
                       label=f'Injection window (r={data["multiplier"]})')
        
        ax_top.set_xlabel('Week since enrollment')
        ax_top.set_ylabel('Weekly hazard')
        ax_top.set_title(f'{scenario.capitalize()} scenario: Hazard curves')
        ax_top.legend(loc='upper right', fontsize=8)
        ax_top.grid(True, alpha=0.3)
        
        # Bottom panel: KCOR
        ax_bot = fig.add_subplot(2, 2, i + 3)
        ax_bot.plot(weeks, kcor, linewidth=2, color=colors[scenario],
                    label=f'KCOR (expected {data["expected_direction"]})')
        ax_bot.axhline(1.0, color='black', linestyle='--', linewidth=1, 
                       label='KCOR = 1 (null)')
        
        # Shade injection window
        ax_bot.axvspan(injection_window[0], injection_window[1], 
                       alpha=0.15, color=colors[scenario])
        
        # Add expected direction indicator
        if scenario == 'harm':
            direction_y = np.nanmax(kcor[injection_window[0]:injection_window[1]]) if len(kcor) > injection_window[1] else 1.1
        else:
            direction_y = np.nanmin(kcor[injection_window[0]:injection_window[1]]) if len(kcor) > injection_window[1] else 0.9
        
        ax_bot.set_xlabel('Week since enrollment')
        ax_bot.set_ylabel('KCOR(t)')
        ax_bot.set_title(f'{scenario.capitalize()} scenario: KCOR deviation from 1')
        ax_bot.legend(loc='best', fontsize=8)
        ax_bot.grid(True, alpha=0.3)
        
        # Set y-axis limits to show deviation clearly
        ymin, ymax = ax_bot.get_ylim()
        if scenario == 'harm':
            ax_bot.set_ylim(0.95, max(ymax, 1.15))
        else:
            ax_bot.set_ylim(min(ymin, 0.85), 1.05)
    
    fig.suptitle('Positive Control: KCOR detects injected hazard effects', 
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot positive control KCOR results"
    )
    parser.add_argument(
        "--input", "-i",
        default="test/positive_control/data/KCOR_positive_control.xlsx",
        help="Input Excel file (KCOR processed output or raw positive control data)"
    )
    parser.add_argument(
        "--output", "-o",
        default="test/positive_control/analysis/fig_pos_control_injected.png",
        help="Output figure path"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Compute KCOR from raw data (if processed output not available)"
    )
    
    args = parser.parse_args()
    
    # Try to read processed output first, fall back to raw computation
    try:
        if args.raw:
            raise ValueError("Forced raw mode")
        results = read_kcor_output(args.input)
        if not results:
            raise ValueError("No valid scenarios in processed output")
    except Exception:
        print("Computing KCOR from raw data...")
        results = compute_kcor_from_raw_data(args.input)
    
    if not results:
        print("Error: No valid data found")
        return 1
    
    plot_positive_control(results, args.output)
    
    # Print summary
    print("\nPositive control results:")
    for scenario, data in results.items():
        if 'kcor' in data:
            # Get KCOR value at end of injection window
            end_idx = min(data['injection_window'][1], len(data['kcor']) - 1)
            kcor_at_end = data['kcor'][end_idx] if end_idx < len(data['kcor']) else np.nan
            print(f"  {scenario.upper()}: KCOR at week {end_idx} = {kcor_at_end:.4f} (expected {data['expected_direction']})")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


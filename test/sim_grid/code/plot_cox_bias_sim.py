#!/usr/bin/env python3
"""
Plot Cox Bias Demonstration Results

Creates figures showing Cox HR vs theta and KCOR vs theta from the Cox bias
demonstration simulation.

Usage:
    python plot_cox_bias_sim.py --input-results RESULTS.csv --output-hr-figure HR_FIG.png --output-kcor-figure KCOR_FIG.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cox_hr_vs_theta(
    csv_path: str,
    out_path: str
) -> None:
    """
    Plot Cox hazard ratio vs frailty variance theta.
    
    Args:
        csv_path: Path to CSV with simulation results
        out_path: Output figure path
    """
    df = pd.read_csv(csv_path).sort_values("theta_B")

    theta = df["theta_B"].to_numpy()
    hr = df["cox_HR"].to_numpy()
    lo = df["cox_CI_low"].to_numpy()
    hi = df["cox_CI_high"].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(theta, hr, marker="o", linewidth=2, markersize=8, label="Cox HR")
    plt.fill_between(theta, lo, hi, alpha=0.2, label="95% CI")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="True HR = 1")
    plt.xlabel("Frailty variance θ (cohort B)", fontsize=12)
    plt.ylabel("Cox hazard ratio (B vs A)", fontsize=12)
    plt.title("Cox regression under synthetic null: HR deviates from 1 with increasing frailty", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}", file=sys.stderr)


def plot_kcor_vs_theta(
    csv_path: str,
    out_path: str
) -> None:
    """
    Plot KCOR asymptote vs frailty variance theta.
    
    Args:
        csv_path: Path to CSV with simulation results
        out_path: Output figure path
    """
    df = pd.read_csv(csv_path).sort_values("theta_B")

    theta = df["theta_B"].to_numpy()
    kcor_end = df["kcor_asymptote"].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(theta, kcor_end, marker="o", linewidth=2, markersize=8, label="KCOR asymptote")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="True KCOR = 1")
    plt.xlabel("Frailty variance θ (cohort B)", fontsize=12)
    plt.ylabel("KCOR asymptote (end of follow-up)", fontsize=12)
    plt.title("KCOR under synthetic null: remains centered near 1 across θ", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Cox bias demonstration results"
    )
    parser.add_argument(
        "--input-results",
        type=str,
        required=True,
        help="Input CSV file with simulation results"
    )
    parser.add_argument(
        "--output-hr-figure",
        type=str,
        default="fig_cox_bias_hr_vs_theta.png",
        help="Output path for Cox HR vs theta figure"
    )
    parser.add_argument(
        "--output-kcor-figure",
        type=str,
        default="fig_cox_bias_kcor_vs_theta.png",
        help="Output path for KCOR vs theta figure"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.input_results):
        print(f"Error: Input file not found: {args.input_results}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    for out_path in [args.output_hr_figure, args.output_kcor_figure]:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

    plot_cox_hr_vs_theta(args.input_results, args.output_hr_figure)
    plot_kcor_vs_theta(args.input_results, args.output_kcor_figure)


if __name__ == "__main__":
    main()


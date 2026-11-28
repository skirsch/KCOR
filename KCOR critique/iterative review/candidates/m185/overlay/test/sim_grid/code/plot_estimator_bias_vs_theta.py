#!/usr/bin/env python3
"""
Plot deviations from unity for standard Cox and known-frailty KCOR.

Reads test/sim_grid/out/cox_bias_results.csv (from generate_cox_bias_sim.py).
Writes documentation/preprint/figures/fig_estimator_bias_vs_theta.png

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out" / "cox_bias_results.csv",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "documentation"
        / "preprint"
        / "figures"
        / "fig_estimator_bias_vs_theta.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    theta = df["theta_B"].to_numpy(dtype=float)
    cox_hr = df["cox_HR"].to_numpy(dtype=float)
    kcor = df["kcor_asymptote"].to_numpy(dtype=float)

    cox_bias = cox_hr - 1.0
    kcor_bias = kcor - 1.0

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(0.0, color="0.45", lw=0.9, ls="--", zorder=0)
    ax.plot(theta, cox_bias, "o-", color="tab:red", lw=2, ms=6, label="Cox HR")
    ax.plot(theta, kcor_bias, "^-", color="tab:green", lw=2, ms=6, label="Week-52 KCOR")
    ax.set_xlabel(r"Frailty variance $\theta_0$ (cohort B)")
    ax.set_ylabel("Difference from 1")
    ax.set_title("Synthetic null: no treatment effect at fixed frailty")
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_png}", flush=True)


if __name__ == "__main__":
    main()

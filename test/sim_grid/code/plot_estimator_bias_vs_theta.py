#!/usr/bin/env python3
"""
Plot relative bias vs theta_0 for Cox, calibrated shared-frailty proxy, and KCOR asymptote.

Reads test/sim_grid/out/cox_bias_results.csv (from generate_cox_bias_sim.py).
Writes documentation/preprint/figures/fig_estimator_bias_vs_theta.png

Shared-frailty series is an APPROXIMATE REFERENCE from Table joint_frailty null row (see manuscript caption), not a refit at each theta.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Proportional bias reduction: gamma-frailty null row of tbl:joint_frailty_comparison
# Standard Cox HR 0.87 vs shared-frailty Cox 0.94 -> shrink factor on (HR-1)
_COX_NULL = 0.87
_SF_NULL = 0.94
_KAPPA = (_SF_NULL - 1.0) / (_COX_NULL - 1.0) if abs(_COX_NULL - 1.0) > 1e-12 else 0.5


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
    sf_hr = 1.0 + _KAPPA * (cox_hr - 1.0)
    sf_bias = sf_hr - 1.0

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(0.0, color="0.45", lw=0.9, ls="--", zorder=0)
    ax.plot(theta, cox_bias, "o-", color="tab:red", lw=2, ms=6, label="Cox HR (bias)")
    ax.plot(
        theta,
        sf_bias,
        "s--",
        color="tab:purple",
        lw=2,
        ms=5,
        label="Shared-frailty approx. (bias)",
    )
    ax.plot(theta, kcor_bias, "^-", color="tab:green", lw=2, ms=6, label=r"KCOR asymptote (bias)")
    ax.set_xlabel(r"Frailty variance $\theta_0$ (cohort B)")
    ax.set_ylabel("Relative bias (estimate $-$ true target)")
    ax.set_title("Synthetic null: true HR = 1, true KCOR asymptote = 1")
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_png}", flush=True)


if __name__ == "__main__":
    main()

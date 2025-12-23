"""
Generate a simple 3‑panel schematic for the KCOR v6 paper.

Outputs (recommended):
- documentation/preprint/figs/kcor_v6_schematic.png
- documentation/preprint/figs_full/kcor_v6_schematic.png (higher DPI)
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _h_obs_from_k_theta_t(k: float, theta: float, t: np.ndarray) -> np.ndarray:
    # Gamma-frailty identity in cumulative-hazard space (default baseline H0(t)=k t)
    # H_obs(t) = (1/theta) * log(1 + theta * k * t)
    if theta == 0.0:
        return k * t
    return (1.0 / theta) * np.log(1.0 + theta * k * t)


def _h0_from_hobs_theta(H_obs: np.ndarray, theta: float) -> np.ndarray:
    # Exact inversion: H0(t) = (exp(theta * H_obs(t)) - 1) / theta
    if theta == 0.0:
        return H_obs
    return (np.exp(theta * H_obs) - 1.0) / theta


def generate(out_path: str, full_out_path: str | None = None) -> None:
    import matplotlib.pyplot as plt

    # --- Panel 1: Individual hazards with multiplicative frailty
    t1 = np.linspace(0, 50, 200)
    h0 = 0.01
    z_vals = [0.5, 1.0, 2.0]

    # --- Panels 2–3: Cohort depletion curvature + inversion
    t2 = np.linspace(0, 60, 241)
    k = 0.02
    theta_vals = [0.25, 1.5]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))

    # Panel 1
    ax = axes[0]
    for z in z_vals:
        ax.plot(t1, z * h0 * np.ones_like(t1), linewidth=2, label=f"z={z:g}")
    ax.set_title("Individuals: $h_i(t)=z_i h_0(t)$")
    ax.set_xlabel("event time $t$")
    ax.set_ylabel("hazard")
    ax.set_ylim(0, max(z_vals) * h0 * 1.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel 2
    ax = axes[1]
    for th in theta_vals:
        H_obs = _h_obs_from_k_theta_t(k=k, theta=th, t=t2)
        ax.plot(t2, H_obs, linewidth=2, label=rf"$\theta={th:g}$")
    ax.set_title("Cohorts: $H^{\\mathrm{obs}}(t)$ curvature")
    ax.set_xlabel("event time $t$")
    ax.set_ylabel("cumulative hazard")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel 3
    ax = axes[2]
    for th in theta_vals:
        H_obs = _h_obs_from_k_theta_t(k=k, theta=th, t=t2)
        H0 = _h0_from_hobs_theta(H_obs=H_obs, theta=th)
        ax.plot(t2, H0, linewidth=2, label=rf"$\theta={th:g}$")
    ax.plot(t2, k * t2, linewidth=1.5, linestyle="--", color="black", alpha=0.6, label=r"$k t$")
    ax.set_title("After inversion: $H_0(t)$ (aligned)")
    ax.set_xlabel("event time $t$")
    ax.set_ylabel("baseline cumulative hazard")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")

    if full_out_path is not None:
        os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
        fig.savefig(full_out_path, dpi=400, bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="documentation/preprint/figs/kcor_v6_schematic.png",
        help="Output PNG path (default: documentation/preprint/figs/kcor_v6_schematic.png)",
    )
    parser.add_argument(
        "--out-full",
        default="documentation/preprint/figs_full/kcor_v6_schematic.png",
        help="Optional higher-DPI output PNG path (default: documentation/preprint/figs_full/kcor_v6_schematic.png)",
    )
    args = parser.parse_args()

    out_full = args.out_full if args.out_full else None
    generate(out_path=args.out, full_out_path=out_full)


if __name__ == "__main__":
    main()



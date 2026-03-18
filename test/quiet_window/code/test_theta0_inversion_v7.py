#!/usr/bin/env python3
"""Acceptance test: forward-then-invert should recover theta0 on chosen branch."""

import math
import numpy as np


def theta_t_from_theta0(theta0: float, h_t: np.ndarray) -> np.ndarray:
    return float(theta0) / np.square(1.0 + float(theta0) * h_t)


def theta0_from_anchor(theta_t: float, h_t: float) -> float:
    """Stable branch used by KCOR v7."""
    disc = 1.0 - 4.0 * theta_t * h_t
    if disc < 0.0:
        return np.nan
    denom = 1.0 - 2.0 * theta_t * h_t + math.sqrt(max(disc, 0.0))
    if abs(denom) < 1e-12:
        return np.nan
    return (2.0 * theta_t) / denom


def run_branch_recovery_test() -> None:
    theta0_grid = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1.0], dtype=float)
    # The chosen branch is the continuous small-H branch; enforce theta0*H < 1.
    # This keeps the branch mapping single-valued for this acceptance test.
    tol = 1e-8

    max_err = 0.0
    tested = 0
    skipped = 0
    for theta0 in theta0_grid:
        h_max = 0.95 / theta0
        h_grid = np.linspace(0.0, h_max, 100, dtype=float)
        theta_t_vals = theta_t_from_theta0(theta0, h_grid)
        for h_t, theta_t in zip(h_grid, theta_t_vals):
            inv = theta0_from_anchor(float(theta_t), float(h_t))
            if not np.isfinite(inv):
                skipped += 1
                continue
            tested += 1
            err = abs(inv - theta0)
            max_err = max(max_err, err)
            if err > tol:
                raise AssertionError(
                    f"Branch recovery failed: theta0={theta0}, H={h_t}, recovered={inv}, err={err}"
                )

    print(
        f"PASS: branch recovery test completed; tested={tested}, skipped={skipped}, max_abs_error={max_err:.3e}"
    )


if __name__ == "__main__":
    run_branch_recovery_test()

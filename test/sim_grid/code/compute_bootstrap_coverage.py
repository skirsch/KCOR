#!/usr/bin/env python3
"""
Compute empirical bootstrap coverage for KCOR simulation scenarios.

For each independent simulated dataset, this script builds a percentile CI from
bootstrap replicates of that dataset, then reports the fraction of datasets whose
CI contains the prespecified true KCOR at ``target_week`` (Monte Carlo estimate of
coverage). This is not the same as pooling KCOR across simulations into one global
interval.

Usage:
    python compute_bootstrap_coverage.py --n-simulations 500 --n-bootstrap 200 \\
        --output coverage_results.csv
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import simulation functions from generate_sim_grid
sys.path.insert(0, os.path.dirname(__file__))
from generate_sim_grid import (
    SimConfig,
    compute_kcor_for_scenario,
    run_scenario_1_gamma_null,
    run_scenario_2_hazard_increase,
    run_scenario_3_hazard_decrease,
    run_scenario_4_nongamma_null,
    run_scenario_6_sparse,
)


def bootstrap_resample_cohort_data(
    weeks: np.ndarray,
    alive: np.ndarray,
    dead: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample cohort event counts to approximate bootstrap variability.

    Weekly deaths are drawn from a Poisson distribution with mean equal to the
    observed count, then alive counts are reconstructed forward in time. This is
    a variance / approximation device for aggregated counts; it is **not** exact
    nonparametric resampling of individual discrete survival paths (e.g. not
    Binomial(alive[t], p_death) resampling).

    Args:
        weeks: Time points
        alive: Number alive at start of each week
        dead: Number dead during each week
        seed: Seed for ``numpy.random.default_rng``

    Returns:
        Resampled weeks, alive, dead arrays (weeks is a copy of the input axis).
    """
    rng = np.random.default_rng(seed)
    n_weeks = len(weeks)

    n_initial = int(alive[0]) if len(alive) > 0 else 0

    dead_resampled = np.zeros(n_weeks, dtype=float)
    alive_resampled = np.zeros(n_weeks, dtype=float)
    current_alive = float(n_initial)

    for t in range(n_weeks):
        alive_resampled[t] = current_alive

        if current_alive > 0:
            mean_deaths = float(dead[t])
            if mean_deaths > 0:
                n_deaths = int(rng.poisson(mean_deaths))
                n_deaths = min(n_deaths, int(current_alive))
            else:
                n_deaths = 0

            dead_resampled[t] = n_deaths
            current_alive = max(0.0, current_alive - n_deaths)
        else:
            dead_resampled[t] = 0

    return weeks.copy(), alive_resampled, dead_resampled


def _extract_kcor_at_week(result: Dict, week: int) -> Optional[float]:
    traj = result.get("kcor_trajectory")
    if traj is None or len(traj) == 0:
        return None
    eval_week = min(week, len(traj) - 1)
    if eval_week < 0:
        return None
    val = traj[eval_week]
    if val is None or not np.isfinite(val):
        return None
    return float(val)


def _bootstrap_kcor_for_dataset(
    scenario_data: Dict,
    cfg: SimConfig,
    n_boot: int,
    sim_index: int,
    base_seed: int,
    target_week: int,
) -> List[float]:
    """Bootstrap KCOR at ``target_week`` for one simulated dataset."""
    kcor_boot: List[float] = []
    cohorts = scenario_data.get("cohorts")
    if not cohorts:
        return kcor_boot

    for b in range(n_boot):
        scenario_boot = copy.deepcopy(scenario_data)
        for idx, cohort in enumerate(scenario_boot["cohorts"]):
            weeks = np.asarray(cohort["weeks"])
            alive = np.asarray(cohort["alive"], dtype=float)
            dead = np.asarray(cohort["dead"], dtype=float)
            boot_seed = base_seed + sim_index * 100_000 + b * 1_000 + idx
            w_b, al_b, de_b = bootstrap_resample_cohort_data(
                weeks, alive, dead, seed=boot_seed
            )
            cohort["weeks"] = w_b
            cohort["alive"] = al_b
            cohort["dead"] = de_b

        try:
            result_b = compute_kcor_for_scenario(scenario_boot, cfg)
            kcor_b = _extract_kcor_at_week(result_b, target_week)
            if kcor_b is not None:
                kcor_boot.append(kcor_b)
        except Exception:
            continue

    return kcor_boot


def compute_coverage_for_scenario_simple(
    scenario_fn,
    scenario_name: str,
    true_kcor: float,
    config: SimConfig,
    n_simulations: int = 500,
    n_bootstrap: int = 200,
    target_week: int = 80,
) -> Dict:
    """
    Empirical bootstrap coverage for KCOR at ``target_week``.

    For each simulated dataset: point KCOR, bootstrap CI from that dataset only,
    indicator whether ``true_kcor`` lies in the CI. Coverage is the mean of those
    indicators (Monte Carlo estimate of P(true_kcor in CI(dataset))).

    ``true_kcor`` must be the estimand aligned with ``target_week`` (same horizon
    as the extracted KCOR trajectory index).
    """
    print(f"\nComputing empirical bootstrap coverage for {scenario_name}...")
    print(f"  True KCOR (at target week): {true_kcor}")
    print(f"  Simulation replicates: {n_simulations}")
    print(f"  Bootstrap replicates per simulation: {n_bootstrap}")
    print(f"  Target week: {target_week}")

    covered_flags: List[bool] = []
    point_estimates: List[float] = []
    ci_lowers: List[float] = []
    ci_uppers: List[float] = []
    n_valid_simulations = 0
    n_failed_simulations = 0

    base_seed = int(config.seed)
    # At least half of B successful draws, and at least 20 when B is large;
    # cap at n_bootstrap so small --n-bootstrap smoke tests remain feasible.
    min_boot_ok = min(n_bootstrap, max(20, int(0.5 * n_bootstrap)))

    for sim in range(n_simulations):
        if (sim + 1) % 25 == 0 or sim == 0:
            print(f"  Simulation {sim + 1}/{n_simulations}")

        sim_seed = base_seed + sim
        config.seed = sim_seed

        try:
            scenario_data = scenario_fn(config)

            result = compute_kcor_for_scenario(scenario_data, config)
            point_kcor = _extract_kcor_at_week(result, target_week)
            if point_kcor is None:
                n_failed_simulations += 1
                continue

            kcor_boot = _bootstrap_kcor_for_dataset(
                scenario_data,
                config,
                n_boot=n_bootstrap,
                sim_index=sim,
                base_seed=base_seed,
                target_week=target_week,
            )

            if len(kcor_boot) < min_boot_ok:
                n_failed_simulations += 1
                continue

            kcor_boot_arr = np.asarray(kcor_boot, dtype=float)
            ci_lower = float(np.percentile(kcor_boot_arr, 2.5))
            ci_upper = float(np.percentile(kcor_boot_arr, 97.5))

            covered = (true_kcor >= ci_lower) and (true_kcor <= ci_upper)

            covered_flags.append(bool(covered))
            point_estimates.append(point_kcor)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            n_valid_simulations += 1

        except Exception:
            n_failed_simulations += 1
            continue

    if n_valid_simulations == 0:
        print("  WARNING: No valid simulations!")
        return {
            "scenario": scenario_name,
            "true_kcor": true_kcor,
            "coverage": np.nan,
            "coverage_percent": np.nan,
            "coverage_se": np.nan,
            "n_valid": 0,
            "n_failed": n_failed_simulations,
            "median_point_estimate": np.nan,
            "mean_ci_lower": np.nan,
            "mean_ci_upper": np.nan,
            "mean_ci_width": np.nan,
        }

    covered_arr = np.asarray(covered_flags, dtype=bool)
    point_arr = np.asarray(point_estimates, dtype=float)
    lower_arr = np.asarray(ci_lowers, dtype=float)
    upper_arr = np.asarray(ci_uppers, dtype=float)

    coverage = float(np.mean(covered_arr))
    coverage_percent = 100.0 * coverage
    mean_ci_width = float(np.mean(upper_arr - lower_arr))
    coverage_se = float(np.sqrt(coverage * (1.0 - coverage) / n_valid_simulations))

    print(f"  Valid simulations: {n_valid_simulations}/{n_simulations}")
    print(f"  Failed simulations: {n_failed_simulations}")
    print(f"  Median point estimate: {np.median(point_arr):.4f}")
    print(f"  Mean CI: [{np.mean(lower_arr):.4f}, {np.mean(upper_arr):.4f}]")
    print(f"  Mean CI width: {mean_ci_width:.4f}")
    print(f"  Empirical coverage: {coverage_percent:.1f}% (SE ~ {100.0 * coverage_se:.2f}%)")

    return {
        "scenario": scenario_name,
        "true_kcor": true_kcor,
        "coverage": coverage,
        "coverage_percent": coverage_percent,
        "coverage_se": coverage_se,
        "n_valid": n_valid_simulations,
        "n_failed": n_failed_simulations,
        "median_point_estimate": float(np.median(point_arr)),
        "mean_ci_lower": float(np.mean(lower_arr)),
        "mean_ci_upper": float(np.mean(upper_arr)),
        "mean_ci_width": mean_ci_width,
    }


def _bootstrap_theta_hats_for_dataset(
    scenario_data: Dict,
    cfg: SimConfig,
    n_boot: int,
    sim_index: int,
    base_seed: int,
) -> Dict[int, List[float]]:
    """Bootstrap ``theta_hat`` per cohort dose for one simulated dataset."""
    theta_boot: Dict[int, List[float]] = {}
    cohorts = scenario_data.get("cohorts")
    if not cohorts:
        return theta_boot

    for b in range(n_boot):
        scenario_boot = copy.deepcopy(scenario_data)
        for idx, cohort in enumerate(scenario_boot["cohorts"]):
            weeks = np.asarray(cohort["weeks"])
            alive = np.asarray(cohort["alive"], dtype=float)
            dead = np.asarray(cohort["dead"], dtype=float)
            boot_seed = base_seed + sim_index * 100_000 + b * 1_000 + idx + 17
            w_b, al_b, de_b = bootstrap_resample_cohort_data(
                weeks, alive, dead, seed=boot_seed
            )
            cohort["weeks"] = w_b
            cohort["alive"] = al_b
            cohort["dead"] = de_b

        try:
            result_b = compute_kcor_for_scenario(scenario_boot, cfg)
            for cr in result_b.get("cohort_results", []):
                dose = int(cr["dose"])
                th = cr.get("theta_hat")
                if th is not None and np.isfinite(th):
                    theta_boot.setdefault(dose, []).append(float(th))
        except Exception:
            continue

    return theta_boot


def compute_coverage_theta_per_arm(
    scenario_fn,
    scenario_name: str,
    config: SimConfig,
    n_simulations: int = 500,
    n_bootstrap: int = 200,
) -> Dict:
    """
    Empirical bootstrap coverage for ``theta_hat`` by cohort dose.

    Pairs ``theta_hat`` with ``theta_true`` using the ``dose`` key on each cohort
    and on ``cohort_results``, not list position. Doses with non-finite
    ``theta_true`` (e.g. non-gamma scenario) are reported as NaN.
    """
    print(f"\nComputing theta bootstrap coverage for {scenario_name}...")
    print(f"  Simulation replicates: {n_simulations}")
    print(f"  Bootstrap replicates per simulation: {n_bootstrap}")

    base_seed = int(config.seed)
    min_boot_ok = min(n_bootstrap, max(20, int(0.5 * n_bootstrap)))

    covered_d0: List[bool] = []
    covered_d1: List[bool] = []
    n_valid_d0 = 0
    n_valid_d1 = 0
    n_failed_simulations = 0

    for sim in range(n_simulations):
        if (sim + 1) % 25 == 0 or sim == 0:
            print(f"  Simulation {sim + 1}/{n_simulations}")

        sim_seed = base_seed + sim
        config.seed = sim_seed

        try:
            scenario_data = scenario_fn(config)
            theta_true_by_dose: Dict[int, float] = {}
            for c in scenario_data.get("cohorts", []):
                d = int(c["dose"])
                tt = c.get("theta_true", np.nan)
                theta_true_by_dose[d] = float(tt) if np.isfinite(tt) else np.nan

            compute_kcor_for_scenario(scenario_data, config)

            theta_boot = _bootstrap_theta_hats_for_dataset(
                scenario_data,
                config,
                n_boot=n_bootstrap,
                sim_index=sim,
                base_seed=base_seed,
            )

            for dose in (0, 1):
                tt = theta_true_by_dose.get(dose, np.nan)
                if not np.isfinite(tt):
                    continue

                boots = theta_boot.get(dose, [])
                if len(boots) < min_boot_ok:
                    continue

                arr = np.asarray(boots, dtype=float)
                ci_lo = float(np.percentile(arr, 2.5))
                ci_hi = float(np.percentile(arr, 97.5))
                covered = (tt >= ci_lo) and (tt <= ci_hi)

                if dose == 0:
                    covered_d0.append(bool(covered))
                    n_valid_d0 += 1
                else:
                    covered_d1.append(bool(covered))
                    n_valid_d1 += 1

        except Exception:
            n_failed_simulations += 1
            continue

    def _summarize(
        covered: List[bool], n_valid: int
    ) -> Tuple[float, float, float]:
        if n_valid == 0:
            return np.nan, np.nan, np.nan
        p = float(np.mean(covered))
        se = float(np.sqrt(p * (1.0 - p) / n_valid))
        return p, 100.0 * p, se

    cov0, pct0, se0 = _summarize(covered_d0, n_valid_d0)
    cov1, pct1, se1 = _summarize(covered_d1, n_valid_d1)

    print(
        f"  Theta coverage dose 0: {pct0 if np.isfinite(pct0) else float('nan'):.1f}% "
        f"(n_valid={n_valid_d0})"
    )
    print(
        f"  Theta coverage dose 1: {pct1 if np.isfinite(pct1) else float('nan'):.1f}% "
        f"(n_valid={n_valid_d1})"
    )

    return {
        "scenario": scenario_name,
        "coverage_theta_d0": cov0,
        "coverage_theta_d0_percent": pct0,
        "coverage_theta_d0_se": se0,
        "n_valid_theta_d0": n_valid_d0,
        "coverage_theta_d1": cov1,
        "coverage_theta_d1_percent": pct1,
        "coverage_theta_d1_se": se1,
        "n_valid_theta_d1": n_valid_d1,
        "n_failed_simulations": n_failed_simulations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute empirical bootstrap coverage for KCOR simulation scenarios"
    )
    parser.add_argument(
        "--n-simulations",
        "-n",
        type=int,
        default=500,
        help="Number of independent simulation replicates per scenario",
    )
    parser.add_argument(
        "--n-bootstrap",
        "-b",
        type=int,
        default=200,
        help="Number of bootstrap replicates per simulation replicate",
    )
    parser.add_argument(
        "--target-week",
        "-w",
        type=int,
        default=80,
        help="Target week for KCOR coverage evaluation",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test/sim_grid/out/bootstrap_coverage.csv",
        help="Output CSV file path (KCOR coverage)",
    )
    parser.add_argument(
        "--output-theta",
        default="test/sim_grid/out/bootstrap_coverage_theta.csv",
        help="Output CSV path for theta coverage by arm",
    )
    parser.add_argument(
        "--skip-theta",
        action="store_true",
        help="Do not run theta coverage (faster)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # true_kcor: KCOR(t) at args.target_week under each DGP (not necessarily the
    # raw hazard multiplier r during the effect window). Harm/benefit values match
    # deterministic KCOR at week 80 from generate_sim_grid scenarios 2–3.
    scenarios = [
        ("Gamma-frailty null", run_scenario_1_gamma_null, 1.0),
        ("Injected effect (harm)", run_scenario_2_hazard_increase, 1.1511348305379185),
        ("Injected effect (benefit)", run_scenario_3_hazard_decrease, 0.9625549099317127),
        ("Non-gamma frailty", run_scenario_4_nongamma_null, 1.0),
        ("Sparse events", run_scenario_6_sparse, 1.0),
    ]

    print("=" * 60)
    print("KCOR empirical bootstrap coverage")
    print("=" * 60)
    print(f"Simulation replicates per scenario: {args.n_simulations}")
    print(f"Bootstrap replicates per simulation: {args.n_bootstrap}")
    print(f"Target week: {args.target_week}")
    print()

    all_kcor: List[Dict] = []
    all_theta: List[Dict] = []

    for scenario_name, scenario_fn, true_kcor in scenarios:
        config = SimConfig(seed=args.seed)
        result_k = compute_coverage_for_scenario_simple(
            scenario_fn,
            scenario_name,
            true_kcor,
            config,
            n_simulations=args.n_simulations,
            n_bootstrap=args.n_bootstrap,
            target_week=args.target_week,
        )
        row = {
            "scenario": scenario_name,
            "true_kcor": true_kcor,
            "target_week": args.target_week,
            "n_bootstrap": args.n_bootstrap,
            "n_simulations": args.n_simulations,
            "coverage_proportion": result_k["coverage"],
            "coverage_percent": result_k["coverage_percent"],
            "coverage_se": result_k["coverage_se"],
            "n_valid": result_k["n_valid"],
            "n_failed": result_k["n_failed"],
            "median_point_estimate": result_k["median_point_estimate"],
            "mean_ci_lower": result_k["mean_ci_lower"],
            "mean_ci_upper": result_k["mean_ci_upper"],
            "mean_ci_width": result_k["mean_ci_width"],
        }
        all_kcor.append(row)

        if not args.skip_theta:
            config_t = SimConfig(seed=args.seed)
            result_t = compute_coverage_theta_per_arm(
                scenario_fn,
                scenario_name,
                config_t,
                n_simulations=args.n_simulations,
                n_bootstrap=args.n_bootstrap,
            )
            all_theta.append(result_t)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_kcor = pd.DataFrame(all_kcor)
    df_kcor.to_csv(args.output, index=False)
    print(f"\nSaved KCOR coverage results to: {args.output}")

    if not args.skip_theta:
        out_theta_dir = os.path.dirname(args.output_theta)
        if out_theta_dir:
            os.makedirs(out_theta_dir, exist_ok=True)
        df_theta = pd.DataFrame(all_theta)
        df_theta.to_csv(args.output_theta, index=False)
        print(f"Saved theta coverage results to: {args.output_theta}")

    return df_kcor


if __name__ == "__main__":
    main()

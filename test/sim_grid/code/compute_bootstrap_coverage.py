#!/usr/bin/env python3
"""
Compute empirical bootstrap coverage for KCOR simulation scenarios.

For each independent simulated dataset, this script builds a percentile CI from
bootstrap replicates of that dataset, then reports the fraction of datasets whose
CI contains the DGP KCOR at ``target_week`` (Monte Carlo estimate of coverage).
``true_kcor`` is always taken from a **single reference DGP draw** at a fixed
documented ``truth_seed`` (see ``BOOTSTRAP_COVERAGE_TRUTH_SEED``), not from raw
hazard multipliers.

The Poisson resampling in ``bootstrap_resample_cohort_data`` is an **approximate**
bootstrap for aggregated counts; see that function's docstring.

Usage (do not run large jobs without local approval; defaults are heavy)::

    python compute_bootstrap_coverage.py --n-simulations 500 --n-bootstrap 200 \\
        --target-week 80 \\
        --output test/sim_grid/out/bootstrap_coverage.csv \\
        --output-replicates test/sim_grid/out/bootstrap_coverage_replicates.csv

    python test/sim_grid/code/plot_bootstrap_coverage.py --pdf
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# Fixed seed for the reference DGP draw used to define true KCOR (and theta truth
# audit). Documented; do not substitute manuscript hazard shortcuts.
BOOTSTRAP_COVERAGE_TRUTH_SEED = 0
SCRIPT_VERSION = "2.1"
CI_METHOD_LABEL = "percentile_2.5_97.5"
FLOAT_EQ_TOL = 1e-9
COVERAGE_WARN_EPS = 1e-6


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


def dgp_kcor_at_target_week(
    scenario_fn: Callable[[SimConfig], Dict],
    config: SimConfig,
    target_week: int,
    truth_seed: int,
) -> float:
    """
    KCOR at ``target_week`` from one reference DGP realization (``truth_seed``).

    Uses the same ``scenario_fn`` and ``compute_kcor_for_scenario`` path as the
    Monte Carlo loop, with ``config.seed`` set to ``truth_seed`` only for this draw.
    """
    cfg = copy.deepcopy(config)
    cfg.seed = int(truth_seed)
    scenario_data = scenario_fn(cfg)
    result = compute_kcor_for_scenario(scenario_data, cfg)
    val = _extract_kcor_at_week(result, target_week)
    if val is None or not np.isfinite(val):
        raise RuntimeError(
            f"DGP KCOR at target_week={target_week} missing/non-finite for "
            f"{getattr(scenario_fn, '__name__', repr(scenario_fn))}, "
            f"truth_seed={truth_seed}."
        )
    return float(val)


def dgp_theta_true_by_dose(
    scenario_fn: Callable[[SimConfig], Dict],
    config: SimConfig,
    truth_seed: int,
) -> Dict[int, float]:
    """``theta_true`` by cohort dose from the same reference DGP draw as KCOR truth."""
    cfg = copy.deepcopy(config)
    cfg.seed = int(truth_seed)
    scenario_data = scenario_fn(cfg)
    out: Dict[int, float] = {}
    for c in scenario_data.get("cohorts", []):
        d = int(c["dose"])
        tt = c.get("theta_true", np.nan)
        out[d] = float(tt) if np.isfinite(tt) else float("nan")
    return out


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
    scenario_fn: Callable[[SimConfig], Dict],
    scenario_name: str,
    true_kcor: float,
    config: SimConfig,
    n_simulations: int = 500,
    n_bootstrap: int = 200,
    target_week: int = 80,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Empirical bootstrap coverage for KCOR at ``target_week``.

    For each simulated dataset: point KCOR, bootstrap CI from that dataset only,
    indicator whether ``true_kcor`` lies in the CI. Coverage is the mean of those
    indicators (Monte Carlo estimate of P(true_kcor in CI(dataset))).
    """
    print(f"\nComputing empirical bootstrap coverage for {scenario_name}...", flush=True)
    print(f"  True KCOR (DGP at target week, truth seed): {true_kcor}", flush=True)
    print(f"  Simulation replicates: {n_simulations}", flush=True)
    print(f"  Bootstrap replicates per simulation: {n_bootstrap}", flush=True)
    print(f"  Target week: {target_week}", flush=True)

    covered_flags: List[bool] = []
    point_estimates: List[float] = []
    ci_lowers: List[float] = []
    ci_uppers: List[float] = []
    n_valid_simulations = 0
    n_failed_simulations = 0
    replicate_rows: List[Dict[str, Any]] = []

    base_seed = int(config.seed)
    min_boot_ok = min(n_bootstrap, max(20, int(0.5 * n_bootstrap)))

    for sim in range(n_simulations):
        if (sim + 1) % 25 == 0 or sim == 0:
            print(f"  Simulation {sim + 1}/{n_simulations}", flush=True)

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

            n_boot_valid = len(kcor_boot)
            n_boot_failed = n_bootstrap - n_boot_valid

            if n_boot_valid < min_boot_ok:
                n_failed_simulations += 1
                continue

            kcor_boot_arr = np.asarray(kcor_boot, dtype=float)
            ci_lower = float(np.percentile(kcor_boot_arr, 2.5))
            ci_upper = float(np.percentile(kcor_boot_arr, 97.5))
            ci_width = float(ci_upper - ci_lower)

            covered = (true_kcor >= ci_lower) and (true_kcor <= ci_upper)

            covered_flags.append(bool(covered))
            point_estimates.append(point_kcor)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            n_valid_simulations += 1

            replicate_rows.append(
                {
                    "scenario_name": scenario_name,
                    "simulation_id": sim + 1,
                    "target_week": target_week,
                    "true_kcor": true_kcor,
                    "point_estimate": point_kcor,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "covered": covered,
                    "ci_width": ci_width,
                    "n_boot_valid": n_boot_valid,
                    "n_boot_failed": n_boot_failed,
                }
            )

        except Exception:
            n_failed_simulations += 1
            continue

    if n_valid_simulations == 0:
        print("  WARNING: No valid simulations!", flush=True)
        summary = {
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
        return summary, replicate_rows

    covered_arr = np.asarray(covered_flags, dtype=bool)
    point_arr = np.asarray(point_estimates, dtype=float)
    lower_arr = np.asarray(ci_lowers, dtype=float)
    upper_arr = np.asarray(ci_uppers, dtype=float)

    coverage = float(np.mean(covered_arr))
    coverage_percent = 100.0 * coverage
    mean_ci_width = float(np.mean(upper_arr - lower_arr))
    coverage_se = float(np.sqrt(coverage * (1.0 - coverage) / n_valid_simulations))

    print(f"  Valid simulations: {n_valid_simulations}/{n_simulations}", flush=True)
    print(f"  Failed simulations: {n_failed_simulations}", flush=True)
    print(f"  Median point estimate: {np.median(point_arr):.4f}", flush=True)
    print(f"  Mean CI: [{np.mean(lower_arr):.4f}, {np.mean(upper_arr):.4f}]", flush=True)
    print(f"  Mean CI width: {mean_ci_width:.4f}", flush=True)
    print(
        f"  Empirical coverage: {coverage_percent:.1f}% (SE ~ {100.0 * coverage_se:.2f}%)",
        flush=True,
    )

    summary = {
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
    return summary, replicate_rows


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
    scenario_fn: Callable[[SimConfig], Dict],
    scenario_name: str,
    config: SimConfig,
    n_simulations: int = 500,
    n_bootstrap: int = 200,
    target_week: int = 80,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Empirical bootstrap coverage for ``theta_hat`` by cohort dose.

    Pairs ``theta_hat`` with ``theta_true`` using the ``dose`` key on each cohort
    and on ``cohort_results``, not list position. Doses with non-finite
    ``theta_true`` (e.g. non-gamma scenario) are skipped.
    """
    print(f"\nComputing theta bootstrap coverage for {scenario_name}...", flush=True)
    print(f"  Simulation replicates: {n_simulations}", flush=True)
    print(f"  Bootstrap replicates per simulation: {n_bootstrap}", flush=True)
    print(f"  target_week (metadata column): {target_week}", flush=True)

    base_seed = int(config.seed)
    min_boot_ok = min(n_bootstrap, max(20, int(0.5 * n_bootstrap)))

    covered_d0: List[bool] = []
    covered_d1: List[bool] = []
    widths_d0: List[float] = []
    widths_d1: List[float] = []
    n_valid_d0 = 0
    n_valid_d1 = 0
    n_failed_simulations = 0
    replicate_rows: List[Dict[str, Any]] = []

    for sim in range(n_simulations):
        if (sim + 1) % 25 == 0 or sim == 0:
            print(f"  Simulation {sim + 1}/{n_simulations}", flush=True)

        sim_seed = base_seed + sim
        config.seed = sim_seed

        try:
            scenario_data = scenario_fn(config)
            theta_true_by_dose: Dict[int, float] = {}
            for c in scenario_data.get("cohorts", []):
                d = int(c["dose"])
                tt = c.get("theta_true", np.nan)
                theta_true_by_dose[d] = float(tt) if np.isfinite(tt) else np.nan

            result = compute_kcor_for_scenario(scenario_data, config)
            theta_hat_by_dose: Dict[int, float] = {}
            for cr in result.get("cohort_results", []):
                d = int(cr["dose"])
                th = cr.get("theta_hat")
                if th is not None and np.isfinite(th):
                    theta_hat_by_dose[d] = float(th)

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

                th_hat = theta_hat_by_dose.get(dose, np.nan)
                if not np.isfinite(th_hat):
                    continue

                boots = theta_boot.get(dose, [])
                n_boot_valid = len(boots)
                n_boot_failed = n_bootstrap - n_boot_valid
                if n_boot_valid < min_boot_ok:
                    continue

                arr = np.asarray(boots, dtype=float)
                ci_lo = float(np.percentile(arr, 2.5))
                ci_hi = float(np.percentile(arr, 97.5))
                ci_width = float(ci_hi - ci_lo)
                covered = (tt >= ci_lo) and (tt <= ci_hi)

                replicate_rows.append(
                    {
                        "scenario_name": scenario_name,
                        "simulation_id": sim + 1,
                        "target_week": target_week,
                        "dose": dose,
                        "theta_true": tt,
                        "theta_hat": th_hat,
                        "point_estimate": th_hat,
                        "ci_lower": ci_lo,
                        "ci_upper": ci_hi,
                        "covered": covered,
                        "ci_width": ci_width,
                        "n_boot_valid": n_boot_valid,
                        "n_boot_failed": n_boot_failed,
                    }
                )

                if dose == 0:
                    covered_d0.append(bool(covered))
                    widths_d0.append(ci_width)
                    n_valid_d0 += 1
                else:
                    covered_d1.append(bool(covered))
                    widths_d1.append(ci_width)
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
    mean_w0 = float(np.mean(widths_d0)) if widths_d0 else float("nan")
    mean_w1 = float(np.mean(widths_d1)) if widths_d1 else float("nan")

    print(
        f"  Theta coverage dose 0: {pct0 if np.isfinite(pct0) else float('nan'):.1f}% "
        f"(n_valid={n_valid_d0})",
        flush=True,
    )
    print(
        f"  Theta coverage dose 1: {pct1 if np.isfinite(pct1) else float('nan'):.1f}% "
        f"(n_valid={n_valid_d1})",
        flush=True,
    )

    summary = {
        "scenario": scenario_name,
        "coverage_theta_d0": cov0,
        "coverage_theta_d0_percent": pct0,
        "coverage_theta_d0_se": se0,
        "n_valid_theta_d0": n_valid_d0,
        "mean_ci_width_theta_d0": mean_w0,
        "coverage_theta_d1": cov1,
        "coverage_theta_d1_percent": pct1,
        "coverage_theta_d1_se": se1,
        "n_valid_theta_d1": n_valid_d1,
        "mean_ci_width_theta_d1": mean_w1,
        "n_failed_simulations": n_failed_simulations,
    }
    return summary, replicate_rows


def _git_commit_short() -> str:
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _warn(msg: str) -> None:
    print(f"  WARNING (QA): {msg}", flush=True)


def _save_progress_csvs(
    *,
    all_kcor: List[Dict[str, Any]],
    all_kcor_reps: List[Dict[str, Any]],
    all_theta: List[Dict[str, Any]],
    all_theta_reps: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Rewrite summary/replicate CSVs with completed scenarios so far (crash-safe)."""
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if all_kcor:
        pd.DataFrame(all_kcor).to_csv(args.output, index=False)
    rep_path = args.output_replicates
    if rep_path and all_kcor_reps:
        rdir = os.path.dirname(rep_path)
        if rdir:
            os.makedirs(rdir, exist_ok=True)
        pd.DataFrame(all_kcor_reps).to_csv(rep_path, index=False)
    if not args.skip_theta and all_theta:
        out_theta_dir = os.path.dirname(args.output_theta)
        if out_theta_dir:
            os.makedirs(out_theta_dir, exist_ok=True)
        pd.DataFrame(all_theta).to_csv(args.output_theta, index=False)
    tr_path = args.output_theta_replicates
    if not args.skip_theta and tr_path and all_theta_reps:
        td = os.path.dirname(tr_path)
        if td:
            os.makedirs(td, exist_ok=True)
        pd.DataFrame(all_theta_reps).to_csv(tr_path, index=False)
    print(
        f"  (checkpoint) wrote partial outputs: {len(all_kcor)} scenario(s) complete",
        flush=True,
    )


def run_qa_checks(
    *,
    truth_seed: int,
    target_week: int,
    n_simulations: int,
    n_bootstrap: int,
    true_kcor_by_scenario: Dict[str, float],
    kcor_summaries: List[Dict[str, Any]],
    kcor_replicates: List[Dict[str, Any]],
    theta_replicates: List[Dict[str, Any]],
    collapsed_ci_warn_frac: float,
    scenario_fns: Dict[str, Callable[[SimConfig], Dict]],
    config_template: SimConfig,
) -> None:
    """Non-fatal QA warnings after a full run."""
    if n_simulations < 100 or n_bootstrap < 100:
        _warn(
            f"n_simulations={n_simulations} or n_bootstrap={n_bootstrap} < 100; "
            "results are not publication-grade."
        )

    for name, tk in true_kcor_by_scenario.items():
        fn = scenario_fns.get(name)
        if fn is None:
            continue
        try:
            recheck = dgp_kcor_at_target_week(fn, config_template, target_week, truth_seed)
        except Exception as e:
            _warn(f"Could not recompute DGP KCOR for '{name}': {e}")
            continue
        if abs(recheck - tk) > 1e-6 * max(1.0, abs(tk)):
            _warn(
                f"true_kcor mismatch for '{name}': manifest/table {tk} vs "
                f"DGP recomputation {recheck} (truth_seed={truth_seed}, week={target_week})."
            )

    for s in kcor_summaries:
        name = s["scenario"]
        cov = s.get("coverage")
        nv = int(s.get("n_valid", 0))
        if nv > n_simulations:
            _warn(f"Scenario '{name}': n_valid={nv} > n_simulations={n_simulations}.")
        if cov is not None and np.isfinite(cov):
            if cov < -COVERAGE_WARN_EPS or cov > 1.0 + COVERAGE_WARN_EPS:
                _warn(f"Scenario '{name}': coverage {cov} outside [0, 1].")

    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in kcor_replicates:
        by_scenario.setdefault(row["scenario_name"], []).append(row)

    for name, rows in by_scenario.items():
        if not rows:
            continue
        neg_w = sum(1 for r in rows if r["ci_width"] < -FLOAT_EQ_TOL)
        if neg_w:
            _warn(f"Scenario '{name}': {neg_w} replicates with negative CI width.")
        collapsed = sum(
            1
            for r in rows
            if np.isclose(r["ci_lower"], r["ci_upper"], atol=FLOAT_EQ_TOL, rtol=0.0)
        )
        frac = collapsed / len(rows)
        if frac > collapsed_ci_warn_frac:
            _warn(
                f"Scenario '{name}': collapsed KCOR CI (lower≈upper) on "
                f"{100.0 * frac:.1f}% of simulations (threshold {100.0 * collapsed_ci_warn_frac:.1f}%)."
            )

    # Theta: collapsed CI by scenario x dose
    theta_groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in theta_replicates:
        theta_groups.setdefault((row["scenario_name"], int(row["dose"])), []).append(row)

    for (name, dose), rows in theta_groups.items():
        if not rows:
            continue
        neg_w = sum(1 for r in rows if r["ci_width"] < -FLOAT_EQ_TOL)
        if neg_w:
            _warn(
                f"Scenario '{name}' dose {dose}: {neg_w} replicates with negative CI width."
            )
        collapsed = sum(
            1
            for r in rows
            if np.isclose(r["ci_lower"], r["ci_upper"], atol=FLOAT_EQ_TOL, rtol=0.0)
        )
        frac = collapsed / len(rows)
        if frac > collapsed_ci_warn_frac:
            _warn(
                f"Scenario '{name}' dose {dose}: collapsed theta CI on "
                f"{100.0 * frac:.1f}% of simulations (threshold {100.0 * collapsed_ci_warn_frac:.1f}%)."
            )

        cov_list = [bool(r["covered"]) for r in rows]
        p = float(np.mean(cov_list))
        if p < -COVERAGE_WARN_EPS or p > 1.0 + COVERAGE_WARN_EPS:
            _warn(f"Scenario '{name}' dose {dose}: coverage {p} outside [0, 1].")


def main() -> pd.DataFrame:
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
        "--truth-seed",
        type=int,
        default=BOOTSTRAP_COVERAGE_TRUTH_SEED,
        help="Seed for the single reference DGP draw defining true KCOR / theta audit",
    )
    parser.add_argument(
        "--collapsed-ci-warn-frac",
        type=float,
        default=0.05,
        help="Warn if fraction of collapsed (degenerate) CIs exceeds this value",
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
        "--output-replicates",
        default="test/sim_grid/out/bootstrap_coverage_replicates.csv",
        help="Per-simulation KCOR replicate output CSV",
    )
    parser.add_argument(
        "--output-theta-replicates",
        default="test/sim_grid/out/bootstrap_coverage_theta_replicates.csv",
        help="Per-simulation theta replicate output CSV",
    )
    parser.add_argument(
        "--manifest",
        default="test/sim_grid/out/bootstrap_coverage_run.json",
        help="Run manifest JSON path",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write bootstrap_coverage_run.json",
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
        help="Base random seed for the Monte Carlo loop (truth uses --truth-seed)",
    )
    parser.add_argument(
        "--only-scenario",
        type=int,
        default=None,
        metavar="INDEX",
        help="If set, run only scenario INDEX (0..4) in the fixed scenario list (debug / partial runs)",
    )

    args = parser.parse_args()

    scenario_defs: List[Tuple[str, Callable[[SimConfig], Dict]]] = [
        ("Gamma-frailty null", run_scenario_1_gamma_null),
        ("Injected effect (harm)", run_scenario_2_hazard_increase),
        ("Injected effect (benefit)", run_scenario_3_hazard_decrease),
        ("Non-gamma frailty", run_scenario_4_nongamma_null),
        ("Sparse events", run_scenario_6_sparse),
    ]
    if args.only_scenario is not None:
        idx = int(args.only_scenario)
        if idx < 0 or idx >= len(scenario_defs):
            raise SystemExit(f"--only-scenario {idx} out of range (0..{len(scenario_defs) - 1})")
        scenario_defs = [scenario_defs[idx]]

    scenario_fns = {name: fn for name, fn in scenario_defs}

    config_probe = SimConfig(seed=args.seed)
    true_kcor_by_scenario: Dict[str, float] = {}
    theta_true_manifest: Dict[str, Dict[str, float]] = {}

    print("=" * 60, flush=True)
    print("KCOR empirical bootstrap coverage", flush=True)
    print("=" * 60, flush=True)
    print(f"Simulation replicates per scenario: {args.n_simulations}", flush=True)
    print(f"Bootstrap replicates per simulation: {args.n_bootstrap}", flush=True)
    print(f"Target week: {args.target_week}", flush=True)
    print(f"Truth seed (reference DGP draw): {args.truth_seed}", flush=True)
    print(flush=True)

    for scenario_name, scenario_fn in scenario_defs:
        tk = dgp_kcor_at_target_week(
            scenario_fn, config_probe, args.target_week, args.truth_seed
        )
        true_kcor_by_scenario[scenario_name] = tk
        tt_map = dgp_theta_true_by_dose(scenario_fn, config_probe, args.truth_seed)
        theta_true_manifest[scenario_name] = {
            f"theta_true_d{k}": v for k, v in sorted(tt_map.items()) if np.isfinite(v)
        }
        print(f"  DGP true_kcor [{scenario_name}]: {tk}", flush=True)

    all_kcor: List[Dict[str, Any]] = []
    all_theta: List[Dict[str, Any]] = []
    all_kcor_reps: List[Dict[str, Any]] = []
    all_theta_reps: List[Dict[str, Any]] = []
    kcor_summaries: List[Dict[str, Any]] = []

    manifest_fields = {
        "truth_seed": args.truth_seed,
        "target_week": args.target_week,
        "ci_method": CI_METHOD_LABEL,
        "n_simulations": args.n_simulations,
        "n_bootstrap": args.n_bootstrap,
        "base_mc_seed": args.seed,
    }

    for scenario_name, scenario_fn in scenario_defs:
        true_kcor = true_kcor_by_scenario[scenario_name]
        config = SimConfig(seed=args.seed)
        result_k, reps_k = compute_coverage_for_scenario_simple(
            scenario_fn,
            scenario_name,
            true_kcor,
            config,
            n_simulations=args.n_simulations,
            n_bootstrap=args.n_bootstrap,
            target_week=args.target_week,
        )
        kcor_summaries.append(result_k)
        all_kcor_reps.extend(reps_k)

        row = {
            "scenario": scenario_name,
            "true_kcor": true_kcor,
            "target_week": args.target_week,
            "truth_seed": args.truth_seed,
            "ci_method": CI_METHOD_LABEL,
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
        row.update(manifest_fields)
        all_kcor.append(row)

        if not args.skip_theta:
            config_t = SimConfig(seed=args.seed)
            result_t, reps_t = compute_coverage_theta_per_arm(
                scenario_fn,
                scenario_name,
                config_t,
                n_simulations=args.n_simulations,
                n_bootstrap=args.n_bootstrap,
                target_week=args.target_week,
            )
            all_theta_reps.extend(reps_t)
            rt = dict(result_t)
            rt.update(manifest_fields)
            all_theta.append(rt)

        _save_progress_csvs(
            all_kcor=all_kcor,
            all_kcor_reps=all_kcor_reps,
            all_theta=all_theta,
            all_theta_reps=all_theta_reps,
            args=args,
        )

    run_qa_checks(
        truth_seed=args.truth_seed,
        target_week=args.target_week,
        n_simulations=args.n_simulations,
        n_bootstrap=args.n_bootstrap,
        true_kcor_by_scenario=true_kcor_by_scenario,
        kcor_summaries=kcor_summaries,
        kcor_replicates=all_kcor_reps,
        theta_replicates=all_theta_reps,
        collapsed_ci_warn_frac=args.collapsed_ci_warn_frac,
        scenario_fns=scenario_fns,
        config_template=SimConfig(seed=args.seed),
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_kcor = pd.DataFrame(all_kcor)
    df_kcor.to_csv(args.output, index=False)
    print(f"\nSaved KCOR coverage results to: {args.output}")

    rep_path = args.output_replicates
    if rep_path:
        rdir = os.path.dirname(rep_path)
        if rdir:
            os.makedirs(rdir, exist_ok=True)
        pd.DataFrame(all_kcor_reps).to_csv(rep_path, index=False)
        print(f"Saved KCOR replicate table to: {rep_path}")

    if not args.skip_theta:
        out_theta_dir = os.path.dirname(args.output_theta)
        if out_theta_dir:
            os.makedirs(out_theta_dir, exist_ok=True)
        df_theta = pd.DataFrame(all_theta)
        df_theta.to_csv(args.output_theta, index=False)
        print(f"Saved theta coverage results to: {args.output_theta}")

        tr_path = args.output_theta_replicates
        if tr_path:
            td = os.path.dirname(tr_path)
            if td:
                os.makedirs(td, exist_ok=True)
            pd.DataFrame(all_theta_reps).to_csv(tr_path, index=False)
            print(f"Saved theta replicate table to: {tr_path}")

    manifest_path = None if args.no_manifest else args.manifest
    if manifest_path:
        mdir = os.path.dirname(manifest_path)
        if mdir:
            os.makedirs(mdir, exist_ok=True)
        manifest: Dict[str, Any] = {
            "script_version": SCRIPT_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit_short(),
            "truth_definition": (
                "dgp_kcor_at_target_week(scenario_fn, SimConfig, target_week, truth_seed); "
                f"BOOTSTRAP_COVERAGE_TRUTH_SEED default={BOOTSTRAP_COVERAGE_TRUTH_SEED}"
            ),
            "truth_seed": args.truth_seed,
            "target_week": args.target_week,
            "n_simulations": args.n_simulations,
            "n_bootstrap": args.n_bootstrap,
            "ci_method": CI_METHOD_LABEL,
            "base_mc_seed": args.seed,
            "true_kcor_by_scenario": true_kcor_by_scenario,
            "theta_true_by_scenario": theta_true_manifest,
            "outputs": {
                "kcor_summary": os.path.normpath(args.output),
                "kcor_replicates": os.path.normpath(rep_path) if rep_path else None,
                "theta_summary": os.path.normpath(args.output_theta)
                if not args.skip_theta
                else None,
                "theta_replicates": os.path.normpath(args.output_theta_replicates)
                if not args.skip_theta
                else None,
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote run manifest: {manifest_path}")

    print("\nDone. QA warnings (if any) are listed above.")
    return df_kcor


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute empirical bootstrap coverage for KCOR simulation scenarios.

For each independent simulated dataset, this script builds a percentile CI from
bootstrap replicates of that dataset, then reports the fraction of datasets whose
CI contains the DGP KCOR at ``target_week`` (Monte Carlo estimate of coverage).
``true_kcor`` is always taken from a **single reference DGP draw** at a fixed
documented ``truth_seed`` (see ``BOOTSTRAP_COVERAGE_TRUTH_SEED``), not from raw
hazard multipliers.

KCOR and θ bootstrap draws share one Poisson resampling world per simulation (one
``compute_kcor_for_scenario`` call per bootstrap draw; Cox fitting is skipped for
coverage runs via ``include_cox=False``).

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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
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

# Picklable scenario index -> runner (for multiprocessing).
SCENARIO_RUNNERS: List[Callable[[SimConfig], Dict]] = [
    run_scenario_1_gamma_null,
    run_scenario_2_hazard_increase,
    run_scenario_3_hazard_decrease,
    run_scenario_4_nongamma_null,
    run_scenario_6_sparse,
]

# Fixed seed for the reference DGP draw used to define true KCOR (and theta truth
# audit). Documented; do not substitute manuscript hazard shortcuts.
BOOTSTRAP_COVERAGE_TRUTH_SEED = 0
SCRIPT_VERSION = "3.0"
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


def compute_truncation_W_for_scenario(
    config: SimConfig,
    target_week: int,
    scenario_data: Dict[str, Any],
) -> int:
    """
    Cached truncation cap W (largest week index kept). Same W for every
    simulation replicate and bootstrap draw within a scenario.
    """
    cohorts = scenario_data.get("cohorts") or []
    if not cohorts:
        return int(target_week)
    last_idx_cap = min(len(c["weeks"]) - 1 for c in cohorts)
    w_need = max(
        int(target_week),
        int(config.quiet_window_end),
        int(config.skip_weeks) + 4,
    )
    return int(min(w_need, last_idx_cap))


def build_bootstrap_scenario_shallow(
    scenario_data: Dict[str, Any],
    *,
    b: int,
    sim_index: int,
    base_seed: int,
) -> Dict[str, Any]:
    """Shallow scenario dict + ``copy.copy`` per cohort; new weeks/alive/dead arrays."""
    cohorts_new: List[Dict[str, Any]] = []
    for idx, cohort in enumerate(scenario_data.get("cohorts", [])):
        weeks = np.asarray(cohort["weeks"])
        alive = np.asarray(cohort["alive"], dtype=float)
        dead = np.asarray(cohort["dead"], dtype=float)
        boot_seed = base_seed + sim_index * 100_000 + b * 1_000 + idx
        w_b, al_b, de_b = bootstrap_resample_cohort_data(
            weeks, alive, dead, seed=boot_seed
        )
        c = copy.copy(cohort)
        c["weeks"] = w_b
        c["alive"] = al_b
        c["dead"] = de_b
        cohorts_new.append(c)
    return {
        "scenario": scenario_data["scenario"],
        "label": scenario_data["label"],
        "cohorts": cohorts_new,
    }


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


def _theta_hat_by_dose_from_result(result: Dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for cr in result.get("cohort_results", []):
        d = int(cr["dose"])
        th = cr.get("theta_hat")
        if th is not None and np.isfinite(th):
            out[d] = float(th)
    return out


def dgp_kcor_at_target_week(
    scenario_fn: Callable[[SimConfig], Dict],
    config: SimConfig,
    target_week: int,
    truth_seed: int,
    *,
    max_week_index: Optional[int] = None,
) -> float:
    """
    KCOR at ``target_week`` from one reference DGP realization (``truth_seed``).

    Uses the same ``scenario_fn`` and ``compute_kcor_for_scenario`` path as the
    Monte Carlo loop (``include_cox=False``), with ``config.seed`` set to
    ``truth_seed`` only for this draw.
    """
    cfg = copy.deepcopy(config)
    cfg.seed = int(truth_seed)
    scenario_data = scenario_fn(cfg)
    result = compute_kcor_for_scenario(
        scenario_data,
        cfg,
        include_cox=False,
        max_week_index=max_week_index,
    )
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


def _env_debug_kcor() -> bool:
    return os.environ.get("KCOR_BOOTSTRAP_DEBUG", "").strip() in ("1", "true", "True")


def _run_one_simulation_combined(
    *,
    scenario_runner_ix: int,
    config: SimConfig,
    sim_index: int,
    base_seed: int,
    n_bootstrap: int,
    target_week: int,
    true_kcor: float,
    scenario_name: str,
    skip_theta: bool,
    max_week_index: Optional[int],
    debug_kcor_call_count: bool,
    min_boot_ok: int,
) -> Dict[str, Any]:
    """
    One simulation replicate: point fit + shared bootstrap loop for KCOR and θ.
    Returns timing fields, optional KCOR replicate row, optional θ replicate rows.
    """
    scenario_fn = SCENARIO_RUNNERS[scenario_runner_ix]
    t_sim0 = time.perf_counter()

    cfg = replace(config)
    cfg.seed = base_seed + sim_index

    t_sg0 = time.perf_counter()
    scenario_data = scenario_fn(cfg)
    t_sg1 = time.perf_counter()
    seconds_scenario_generation = t_sg1 - t_sg0

    seconds_point_compute_kcor = 0.0
    seconds_bootstrap_resample_and_build = 0.0
    seconds_bootstrap_compute_kcor = 0.0
    seconds_bootstrap_total = 0.0

    kcor_boot: List[float] = []
    theta_boot: Dict[int, List[float]] = {0: [], 1: []}

    t_pk0 = time.perf_counter()
    result = compute_kcor_for_scenario(
        scenario_data,
        cfg,
        include_cox=False,
        max_week_index=max_week_index,
    )
    t_pk1 = time.perf_counter()
    seconds_point_compute_kcor = t_pk1 - t_pk0

    point_kcor = _extract_kcor_at_week(result, target_week)
    theta_hat_by_dose = _theta_hat_by_dose_from_result(result) if not skip_theta else {}

    theta_true_by_dose: Dict[int, float] = {}
    for c in scenario_data.get("cohorts", []):
        d = int(c["dose"])
        tt = c.get("theta_true", np.nan)
        theta_true_by_dose[d] = float(tt) if np.isfinite(tt) else float("nan")

    if point_kcor is None:
        t_sim1 = time.perf_counter()
        return {
            "status": "failed",
            "sim_index": sim_index,
            "fail_reason": "point_kcor",
            "kcor_row": None,
            "theta_rows": [],
            "seconds_scenario_generation": seconds_scenario_generation,
            "seconds_point_compute_kcor": seconds_point_compute_kcor,
            "seconds_bootstrap_total": 0.0,
            "seconds_bootstrap_resample_and_build": 0.0,
            "seconds_bootstrap_compute_kcor": 0.0,
            "seconds_sim_total": t_sim1 - t_sim0,
        }

    dbg = debug_kcor_call_count or _env_debug_kcor()
    warned_double = False

    for b in range(n_bootstrap):
        t_iter0 = time.perf_counter()

        t_rb0 = time.perf_counter()
        scenario_boot = build_bootstrap_scenario_shallow(
            scenario_data, b=b, sim_index=sim_index, base_seed=base_seed
        )
        t_rb1 = time.perf_counter()
        seconds_bootstrap_resample_and_build += t_rb1 - t_rb0

        kcor_calls = 0
        t_k0 = time.perf_counter()
        result_b = compute_kcor_for_scenario(
            scenario_boot,
            cfg,
            include_cox=False,
            max_week_index=max_week_index,
        )
        t_k1 = time.perf_counter()
        kcor_calls += 1
        seconds_bootstrap_compute_kcor += t_k1 - t_k0

        if dbg:
            assert kcor_calls == 1, "bootstrap iteration must call compute_kcor exactly once"
        elif kcor_calls != 1 and not warned_double:
            print(
                "  WARNING: bootstrap iteration KCOR call count != 1 (check for regressions).",
                flush=True,
            )
            warned_double = True

        kcor_b = _extract_kcor_at_week(result_b, target_week)
        if kcor_b is not None:
            kcor_boot.append(kcor_b)

        if not skip_theta:
            for cr in result_b.get("cohort_results", []):
                d = int(cr["dose"])
                th = cr.get("theta_hat")
                if th is not None and np.isfinite(th):
                    theta_boot.setdefault(d, []).append(float(th))

        t_iter1 = time.perf_counter()
        seconds_bootstrap_total += t_iter1 - t_iter0

    n_boot_valid = len(kcor_boot)
    n_boot_failed = n_bootstrap - n_boot_valid

    kcor_row: Optional[Dict[str, Any]] = None
    if n_boot_valid < min_boot_ok:
        t_sim1 = time.perf_counter()
        return {
            "status": "failed",
            "sim_index": sim_index,
            "fail_reason": "kcor_boot",
            "kcor_row": None,
            "theta_rows": [],
            "seconds_scenario_generation": seconds_scenario_generation,
            "seconds_point_compute_kcor": seconds_point_compute_kcor,
            "seconds_bootstrap_total": seconds_bootstrap_total,
            "seconds_bootstrap_resample_and_build": seconds_bootstrap_resample_and_build,
            "seconds_bootstrap_compute_kcor": seconds_bootstrap_compute_kcor,
            "seconds_sim_total": t_sim1 - t_sim0,
        }

    kcor_boot_arr = np.asarray(kcor_boot, dtype=float)
    ci_lower = float(np.percentile(kcor_boot_arr, 2.5))
    ci_upper = float(np.percentile(kcor_boot_arr, 97.5))
    ci_width = float(ci_upper - ci_lower)
    covered = (true_kcor >= ci_lower) and (true_kcor <= ci_upper)

    kcor_row = {
        "scenario_name": scenario_name,
        "simulation_id": sim_index + 1,
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

    theta_rows: List[Dict[str, Any]] = []
    if not skip_theta:
        for dose in (0, 1):
            tt = theta_true_by_dose.get(dose, np.nan)
            if not np.isfinite(tt):
                continue
            th_hat = theta_hat_by_dose.get(dose, np.nan)
            if not np.isfinite(th_hat):
                continue
            boots = theta_boot.get(dose, [])
            n_t_ok = len(boots)
            n_t_fail = n_bootstrap - n_t_ok
            if n_t_ok < min_boot_ok:
                continue
            arr = np.asarray(boots, dtype=float)
            ci_lo = float(np.percentile(arr, 2.5))
            ci_hi = float(np.percentile(arr, 97.5))
            w = float(ci_hi - ci_lo)
            cov = (tt >= ci_lo) and (tt <= ci_hi)
            theta_rows.append(
                {
                    "scenario_name": scenario_name,
                    "simulation_id": sim_index + 1,
                    "target_week": target_week,
                    "dose": dose,
                    "theta_true": tt,
                    "theta_hat": th_hat,
                    "point_estimate": th_hat,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "covered": cov,
                    "ci_width": w,
                    "n_boot_valid": n_t_ok,
                    "n_boot_failed": n_t_fail,
                }
            )

    t_sim1 = time.perf_counter()
    return {
        "status": "ok",
        "sim_index": sim_index,
        "fail_reason": None,
        "kcor_row": kcor_row,
        "theta_rows": theta_rows,
        "seconds_scenario_generation": seconds_scenario_generation,
        "seconds_point_compute_kcor": seconds_point_compute_kcor,
        "seconds_bootstrap_total": seconds_bootstrap_total,
        "seconds_bootstrap_resample_and_build": seconds_bootstrap_resample_and_build,
        "seconds_bootstrap_compute_kcor": seconds_bootstrap_compute_kcor,
        "seconds_sim_total": t_sim1 - t_sim0,
    }


def _mp_worker_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level entry for ``ProcessPoolExecutor`` (must be picklable)."""
    cfg = payload["config"]
    assert isinstance(cfg, SimConfig)
    return _run_one_simulation_combined(
        scenario_runner_ix=int(payload["scenario_runner_ix"]),
        config=cfg,
        sim_index=int(payload["sim_index"]),
        base_seed=int(payload["base_seed"]),
        n_bootstrap=int(payload["n_bootstrap"]),
        target_week=int(payload["target_week"]),
        true_kcor=float(payload["true_kcor"]),
        scenario_name=str(payload["scenario_name"]),
        skip_theta=bool(payload["skip_theta"]),
        max_week_index=payload.get("max_week_index"),
        debug_kcor_call_count=bool(payload["debug_kcor_call_count"]),
        min_boot_ok=int(payload["min_boot_ok"]),
    )


def compute_coverage_combined_for_scenario(
    *,
    scenario_runner_ix: int,
    scenario_name: str,
    true_kcor: float,
    config: SimConfig,
    n_simulations: int,
    n_bootstrap: int,
    target_week: int,
    max_week_index: Optional[int],
    skip_theta: bool,
    max_workers: int,
    debug_kcor_call_count: bool,
    checkpoint_every: int,
    abort_slow_seconds_per_sim: Optional[float],
    args_ns: argparse.Namespace,
    all_kcor: List[Dict[str, Any]],
    all_kcor_reps: List[Dict[str, Any]],
    all_theta: List[Dict[str, Any]],
    all_theta_reps: List[Dict[str, Any]],
    scenario_started_callback: Optional[Callable[[], None]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """
    Combined KCOR (+ optional θ) bootstrap coverage for one scenario.

    Returns ``(kcor_summary, theta_summary_or_empty, timing_aggregate)``.
    """
    if scenario_started_callback:
        scenario_started_callback()

    print(f"\nComputing empirical bootstrap coverage for {scenario_name}...", flush=True)
    print(f"  True KCOR (DGP at target week, truth seed): {true_kcor}", flush=True)
    print(f"  Simulation replicates: {n_simulations}", flush=True)
    print(f"  Bootstrap replicates per simulation: {n_bootstrap}", flush=True)
    print(f"  Target week: {target_week}", flush=True)
    print(f"  max_workers: {max_workers}", flush=True)
    if max_week_index is not None:
        print(f"  Truncation max_week_index (cached W): {max_week_index}", flush=True)

    base_seed = int(config.seed)
    min_boot_ok = min(n_bootstrap, max(20, int(0.5 * n_bootstrap)))

    effective_workers = max(1, min(int(max_workers), int(n_simulations)))
    if effective_workers > 1 and checkpoint_every > 0:
        print(
            "  NOTE: --checkpoint-every is only applied between simulations in serial "
            "mode; with --max-workers > 1, mid-scenario checkpoints are skipped (final "
            "save still runs at scenario end).",
            flush=True,
        )

    covered_flags: List[bool] = []
    point_estimates: List[float] = []
    ci_lowers: List[float] = []
    ci_uppers: List[float] = []
    n_valid_simulations = 0
    n_failed_simulations = 0
    kcor_replicate_rows: List[Dict[str, Any]] = []

    covered_d0: List[bool] = []
    covered_d1: List[bool] = []
    widths_d0: List[float] = []
    widths_d1: List[float] = []
    n_valid_d0 = 0
    n_valid_d1 = 0
    theta_replicate_rows: List[Dict[str, Any]] = []

    # Timing aggregates (parent; same fields as workers return).
    sum_seconds_scenario_generation = 0.0
    sum_seconds_point_compute_kcor = 0.0
    sum_seconds_bootstrap_total = 0.0
    sum_seconds_bootstrap_resample_and_build = 0.0
    sum_seconds_bootstrap_compute_kcor = 0.0
    sum_seconds_sim_total = 0.0
    n_timing_samples = 0

    sum_compute_kcor_wall = 0.0  # point + bootstrap compute_kcor only
    total_boot_iters = 0

    completed_for_slow_check: List[float] = []
    slow_warned = False

    def _accumulate_timing(rec: Dict[str, Any]) -> None:
        nonlocal sum_seconds_scenario_generation, sum_seconds_point_compute_kcor
        nonlocal sum_seconds_bootstrap_total, sum_seconds_bootstrap_resample_and_build
        nonlocal sum_seconds_bootstrap_compute_kcor, sum_seconds_sim_total
        nonlocal n_timing_samples, sum_compute_kcor_wall, total_boot_iters

        sum_seconds_scenario_generation += float(rec["seconds_scenario_generation"])
        sum_seconds_point_compute_kcor += float(rec["seconds_point_compute_kcor"])
        sum_seconds_bootstrap_total += float(rec["seconds_bootstrap_total"])
        sum_seconds_bootstrap_resample_and_build += float(
            rec["seconds_bootstrap_resample_and_build"]
        )
        sum_seconds_bootstrap_compute_kcor += float(rec["seconds_bootstrap_compute_kcor"])
        sum_seconds_sim_total += float(rec["seconds_sim_total"])
        n_timing_samples += 1
        sum_compute_kcor_wall += float(rec["seconds_point_compute_kcor"]) + float(
            rec["seconds_bootstrap_compute_kcor"]
        )
        total_boot_iters += n_bootstrap

    def _maybe_slow_and_abort() -> None:
        nonlocal slow_warned
        if len(completed_for_slow_check) < 3:
            return
        avg = float(np.mean(completed_for_slow_check))
        if not slow_warned and avg > 60.0:
            print(
                "  WARNING: average wall time per simulation exceeds 60 s; consider "
                "lowering --n-bootstrap and/or using --max-workers > 1.",
                flush=True,
            )
            slow_warned = True
        if (
            abort_slow_seconds_per_sim is not None
            and avg > float(abort_slow_seconds_per_sim)
        ):
            raise SystemExit(
                f"Abort: average seconds per simulation ({avg:.3f}) exceeds "
                f"--abort-if-slow-seconds-per-sim={abort_slow_seconds_per_sim}. "
                "Partial CSVs may have been written at the last checkpoint."
            )

    def _checkpoint_if_needed() -> None:
        if checkpoint_every <= 0:
            return
        if n_valid_simulations > 0 and n_valid_simulations % checkpoint_every == 0:
            _save_progress_csvs(
                all_kcor=all_kcor,
                all_kcor_reps=all_kcor_reps,
                all_theta=all_theta,
                all_theta_reps=all_theta_reps,
                args=args_ns,
            )

    payloads = []
    for sim in range(n_simulations):
        payloads.append(
            {
                "scenario_runner_ix": scenario_runner_ix,
                "config": replace(config),
                "sim_index": sim,
                "base_seed": base_seed,
                "n_bootstrap": n_bootstrap,
                "target_week": target_week,
                "true_kcor": true_kcor,
                "scenario_name": scenario_name,
                "skip_theta": skip_theta,
                "max_week_index": max_week_index,
                "debug_kcor_call_count": debug_kcor_call_count,
                "min_boot_ok": min_boot_ok,
            }
        )

    ordered_results: List[Optional[Dict[str, Any]]] = [None] * n_simulations

    if effective_workers <= 1:
        for sim in range(n_simulations):
            if (sim + 1) % 25 == 0 or sim == 0:
                print(f"  Simulation {sim + 1}/{n_simulations}", flush=True)
            rec = _run_one_simulation_combined(
                scenario_runner_ix=scenario_runner_ix,
                config=replace(config),
                sim_index=sim,
                base_seed=base_seed,
                n_bootstrap=n_bootstrap,
                target_week=target_week,
                true_kcor=true_kcor,
                scenario_name=scenario_name,
                skip_theta=skip_theta,
                max_week_index=max_week_index,
                debug_kcor_call_count=debug_kcor_call_count,
                min_boot_ok=min_boot_ok,
            )
            ordered_results[sim] = rec
            _accumulate_timing(rec)
            completed_for_slow_check.append(float(rec["seconds_sim_total"]))
            _maybe_slow_and_abort()

            if rec["status"] != "ok":
                n_failed_simulations += 1
                continue

            assert rec["kcor_row"] is not None
            kr = rec["kcor_row"]
            kcor_replicate_rows.append(kr)
            covered_flags.append(bool(kr["covered"]))
            point_estimates.append(float(kr["point_estimate"]))
            ci_lowers.append(float(kr["ci_lower"]))
            ci_uppers.append(float(kr["ci_upper"]))
            n_valid_simulations += 1

            for tr in rec["theta_rows"]:
                theta_replicate_rows.append(tr)
                dose = int(tr["dose"])
                if dose == 0:
                    covered_d0.append(bool(tr["covered"]))
                    widths_d0.append(float(tr["ci_width"]))
                    n_valid_d0 += 1
                else:
                    covered_d1.append(bool(tr["covered"]))
                    widths_d1.append(float(tr["ci_width"]))
                    n_valid_d1 += 1

            _checkpoint_if_needed()
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as ex:
            futures = {
                ex.submit(_mp_worker_entry, payloads[sim]): sim
                for sim in range(n_simulations)
            }
            done = 0
            for fut in as_completed(futures):
                sim = futures[fut]
                rec = fut.result()
                ordered_results[sim] = rec
                done += 1
                if done % 25 == 0 or done == 1:
                    print(f"  Simulation {done}/{n_simulations} (async)", flush=True)
                _accumulate_timing(rec)
                completed_for_slow_check.append(float(rec["seconds_sim_total"]))
                _maybe_slow_and_abort()

        # Deterministic row order by sim_index
        for sim in range(n_simulations):
            rec = ordered_results[sim]
            assert rec is not None
            if rec["status"] != "ok":
                n_failed_simulations += 1
                continue

            assert rec["kcor_row"] is not None
            kr = rec["kcor_row"]
            kcor_replicate_rows.append(kr)
            covered_flags.append(bool(kr["covered"]))
            point_estimates.append(float(kr["point_estimate"]))
            ci_lowers.append(float(kr["ci_lower"]))
            ci_uppers.append(float(kr["ci_upper"]))
            n_valid_simulations += 1

            for tr in rec["theta_rows"]:
                theta_replicate_rows.append(tr)
                dose = int(tr["dose"])
                if dose == 0:
                    covered_d0.append(bool(tr["covered"]))
                    widths_d0.append(float(tr["ci_width"]))
                    n_valid_d0 += 1
                else:
                    covered_d1.append(bool(tr["covered"]))
                    widths_d1.append(float(tr["ci_width"]))
                    n_valid_d1 += 1

        if checkpoint_every > 0 and n_valid_simulations > 0:
            _save_progress_csvs(
                all_kcor=all_kcor,
                all_kcor_reps=all_kcor_reps,
                all_theta=all_theta,
                all_theta_reps=all_theta_reps,
                args=args_ns,
            )

    # Per-scenario timing prints
    if n_timing_samples > 0:
        mean_sim = sum_seconds_sim_total / n_timing_samples
        mean_boot_wall = (
            sum_seconds_bootstrap_total / max(1, n_bootstrap * n_timing_samples)
        )
        pct_kcor = (
            100.0 * sum_compute_kcor_wall / sum_seconds_sim_total
            if sum_seconds_sim_total > 0
            else float("nan")
        )
        print(
            f"  Timing: mean wall seconds / simulation ≈ {mean_sim:.3f}; "
            f"mean wall seconds / bootstrap draw ≈ {mean_boot_wall:.4f} "
            f"(over completed workers); "
            f"compute_kcor % of scenario wall ≈ {pct_kcor:.1f}%",
            flush=True,
        )
        print(
            f"  Total scenario wall time (sum over simulations): {sum_seconds_sim_total:.3f} s",
            flush=True,
        )

    manifest_fields = {
        "truth_seed": args_ns.truth_seed,
        "target_week": target_week,
        "ci_method": CI_METHOD_LABEL,
        "n_simulations": n_simulations,
        "n_bootstrap": n_bootstrap,
        "base_mc_seed": args_ns.seed,
    }

    if n_valid_simulations == 0:
        print("  WARNING: No valid simulations!", flush=True)
        summary_k = {
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
        summary_t = {
            "scenario": scenario_name,
            "coverage_theta_d0": np.nan,
            "coverage_theta_d0_percent": np.nan,
            "coverage_theta_d0_se": np.nan,
            "n_valid_theta_d0": 0,
            "mean_ci_width_theta_d0": np.nan,
            "coverage_theta_d1": np.nan,
            "coverage_theta_d1_percent": np.nan,
            "coverage_theta_d1_se": np.nan,
            "n_valid_theta_d1": 0,
            "mean_ci_width_theta_d1": np.nan,
            "n_failed_simulations": n_failed_simulations,
        }
        timing_agg = _timing_aggregate_dict(
            n_timing_samples,
            sum_seconds_scenario_generation,
            sum_seconds_point_compute_kcor,
            sum_seconds_bootstrap_total,
            sum_seconds_bootstrap_resample_and_build,
            sum_seconds_bootstrap_compute_kcor,
            sum_seconds_sim_total,
            sum_compute_kcor_wall,
            n_simulations,
            n_bootstrap,
        )
        row0 = {
            "scenario": scenario_name,
            "true_kcor": true_kcor,
            "target_week": target_week,
            "truth_seed": args_ns.truth_seed,
            "ci_method": CI_METHOD_LABEL,
            "n_bootstrap": n_bootstrap,
            "n_simulations": n_simulations,
            "coverage_proportion": summary_k["coverage"],
            "coverage_percent": summary_k["coverage_percent"],
            "coverage_se": summary_k["coverage_se"],
            "n_valid": summary_k["n_valid"],
            "n_failed": summary_k["n_failed"],
            "median_point_estimate": summary_k["median_point_estimate"],
            "mean_ci_lower": summary_k["mean_ci_lower"],
            "mean_ci_upper": summary_k["mean_ci_upper"],
            "mean_ci_width": summary_k["mean_ci_width"],
        }
        row0.update(manifest_fields)
        all_kcor.append(row0)
        if not skip_theta:
            rt0 = dict(summary_t)
            rt0.update(manifest_fields)
            all_theta.append(rt0)
        _save_progress_csvs(
            all_kcor=all_kcor,
            all_kcor_reps=all_kcor_reps,
            all_theta=all_theta,
            all_theta_reps=all_theta_reps,
            args=args_ns,
        )
        return summary_k, summary_t, timing_agg

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

    summary_k = {
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

    def _summarize_theta(
        covered: List[bool], n_valid: int
    ) -> Tuple[float, float, float]:
        if n_valid == 0:
            return np.nan, np.nan, np.nan
        p = float(np.mean(covered))
        se = float(np.sqrt(p * (1.0 - p) / n_valid))
        return p, 100.0 * p, se

    cov0, pct0, se0 = _summarize_theta(covered_d0, n_valid_d0)
    cov1, pct1, se1 = _summarize_theta(covered_d1, n_valid_d1)
    mean_w0 = float(np.mean(widths_d0)) if widths_d0 else float("nan")
    mean_w1 = float(np.mean(widths_d1)) if widths_d1 else float("nan")

    if not skip_theta:
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

    summary_t = {
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

    timing_agg = _timing_aggregate_dict(
        n_timing_samples,
        sum_seconds_scenario_generation,
        sum_seconds_point_compute_kcor,
        sum_seconds_bootstrap_total,
        sum_seconds_bootstrap_resample_and_build,
        sum_seconds_bootstrap_compute_kcor,
        sum_seconds_sim_total,
        sum_compute_kcor_wall,
        n_simulations,
        n_bootstrap,
    )

    # Extend caller lists (ordered replicate rows: KCOR by simulation_id, θ by (sim, dose))
    all_kcor_reps.extend(kcor_replicate_rows)
    all_theta_reps.extend(theta_replicate_rows)

    row = {
        "scenario": scenario_name,
        "true_kcor": true_kcor,
        "target_week": target_week,
        "truth_seed": args_ns.truth_seed,
        "ci_method": CI_METHOD_LABEL,
        "n_bootstrap": n_bootstrap,
        "n_simulations": n_simulations,
        "coverage_proportion": summary_k["coverage"],
        "coverage_percent": summary_k["coverage_percent"],
        "coverage_se": summary_k["coverage_se"],
        "n_valid": summary_k["n_valid"],
        "n_failed": summary_k["n_failed"],
        "median_point_estimate": summary_k["median_point_estimate"],
        "mean_ci_lower": summary_k["mean_ci_lower"],
        "mean_ci_upper": summary_k["mean_ci_upper"],
        "mean_ci_width": summary_k["mean_ci_width"],
    }
    row.update(manifest_fields)
    all_kcor.append(row)

    if not skip_theta:
        rt = dict(summary_t)
        rt.update(manifest_fields)
        all_theta.append(rt)

    _save_progress_csvs(
        all_kcor=all_kcor,
        all_kcor_reps=all_kcor_reps,
        all_theta=all_theta,
        all_theta_reps=all_theta_reps,
        args=args_ns,
    )

    return summary_k, summary_t, timing_agg


def _timing_aggregate_dict(
    n_timing_samples: int,
    sum_sg: float,
    sum_pk: float,
    sum_bt: float,
    sum_rb: float,
    sum_bk: float,
    sum_sim: float,
    sum_kcor_wall: float,
    n_simulations: int,
    n_bootstrap: int,
) -> Dict[str, float]:
    ns = max(1, n_timing_samples)
    n_boot_draws = max(1, n_timing_samples * n_bootstrap)
    return {
        "n_timing_samples": float(n_timing_samples),
        "mean_seconds_scenario_generation": sum_sg / ns,
        "mean_seconds_point_compute_kcor": sum_pk / ns,
        "mean_seconds_bootstrap_total_per_sim": sum_bt / ns,
        "mean_seconds_bootstrap_resample_and_build_per_sim": sum_rb / ns,
        "mean_seconds_bootstrap_compute_kcor_per_sim": sum_bk / ns,
        "mean_seconds_sim_total": sum_sim / ns,
        "mean_seconds_per_bootstrap_draw_wall": sum_bt / n_boot_draws,
        "mean_compute_kcor_fraction_of_sim_wall": sum_kcor_wall / max(sum_sim, 1e-12),
        "total_seconds_sim_wall_sum": sum_sim,
    }


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
    if getattr(args, "benchmark_mode", False) and not getattr(
        args, "benchmark_write", False
    ):
        return
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
    truncate: bool = False,
    skip_theta: bool = False,
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
            W_recheck: Optional[int] = None
            if truncate:
                cfg_t = copy.deepcopy(config_template)
                cfg_t.seed = int(truth_seed)
                probe = fn(cfg_t)
                W_recheck = compute_truncation_W_for_scenario(
                    config_template, target_week, probe
                )
            recheck = dgp_kcor_at_target_week(
                fn,
                config_template,
                target_week,
                truth_seed,
                max_week_index=W_recheck,
            )
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

    if skip_theta:
        return

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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Process pool size for outer simulation loop (default 1 = serial)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        metavar="N",
        help="Write partial CSVs every N completed valid simulations per scenario (0=off)",
    )
    parser.add_argument(
        "--abort-if-slow-seconds-per-sim",
        type=float,
        default=None,
        metavar="T",
        help="After 3+ simulations, exit non-zero if mean wall seconds per sim exceeds T",
    )
    parser.add_argument(
        "--debug-kcor-call-count",
        action="store_true",
        help="Assert exactly one compute_kcor call per bootstrap iteration",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate cohort arrays to cached W per scenario (see plan; regression-tested)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark/timing mode: default scenario 0 only; skip theta unless --benchmark-with-theta; "
        "no CSV/manifest unless --benchmark-write",
    )
    parser.add_argument(
        "--benchmark-write",
        action="store_true",
        help="With --benchmark, still write CSV/manifest outputs",
    )
    parser.add_argument(
        "--benchmark-all-scenarios",
        action="store_true",
        help="With --benchmark, run all scenarios instead of scenario index 0 only",
    )
    parser.add_argument(
        "--benchmark-with-theta",
        action="store_true",
        help="With --benchmark, include theta extraction/bootstrap in timing",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny preset (5 sims, 20 boots) with a runtime warning",
    )

    args = parser.parse_args()

    if args.smoke:
        args.n_simulations = 5
        args.n_bootstrap = 20
        print(
            "  WARNING: --smoke enables a tiny preset for quick checks; "
            "results are not statistically meaningful.",
            flush=True,
        )

    args.benchmark_mode = bool(args.benchmark)
    if args.benchmark_mode:
        if not args.benchmark_with_theta:
            args.skip_theta = True
        if not args.benchmark_write:
            args.no_manifest = True

    all_scenario_triples: List[Tuple[str, Callable[[SimConfig], Dict], int]] = [
        ("Gamma-frailty null", run_scenario_1_gamma_null, 0),
        ("Injected effect (harm)", run_scenario_2_hazard_increase, 1),
        ("Injected effect (benefit)", run_scenario_3_hazard_decrease, 2),
        ("Non-gamma frailty", run_scenario_4_nongamma_null, 3),
        ("Sparse events", run_scenario_6_sparse, 4),
    ]
    scenario_worklist = list(all_scenario_triples)
    if args.benchmark_mode and not args.benchmark_all_scenarios:
        scenario_worklist = [all_scenario_triples[0]]
    if args.only_scenario is not None:
        idx = int(args.only_scenario)
        if idx < 0 or idx >= len(all_scenario_triples):
            raise SystemExit(
                f"--only-scenario {idx} out of range (0..{len(all_scenario_triples) - 1})"
            )
        scenario_worklist = [all_scenario_triples[idx]]

    scenario_fns = {name: fn for name, fn, _ in scenario_worklist}

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

    t_truth0 = time.perf_counter()
    max_W_by_scenario: Dict[str, Optional[int]] = {}

    for scenario_name, scenario_fn, _runner_ix in scenario_worklist:
        # Probe once to cache truncation W (same for all sims/boots in this scenario).
        probe_cfg = copy.deepcopy(config_probe)
        probe_cfg.seed = int(args.truth_seed)
        probe_scenario = scenario_fn(probe_cfg)
        W: Optional[int] = None
        if args.truncate:
            W = compute_truncation_W_for_scenario(
                config_probe, args.target_week, probe_scenario
            )
        max_W_by_scenario[scenario_name] = W

        tk = dgp_kcor_at_target_week(
            scenario_fn,
            config_probe,
            args.target_week,
            args.truth_seed,
            max_week_index=W,
        )
        true_kcor_by_scenario[scenario_name] = tk
        if not args.skip_theta:
            tt_map = dgp_theta_true_by_dose(scenario_fn, config_probe, args.truth_seed)
            theta_true_manifest[scenario_name] = {
                f"theta_true_d{k}": v for k, v in sorted(tt_map.items()) if np.isfinite(v)
            }
        print(f"  DGP true_kcor [{scenario_name}]: {tk}", flush=True)

    seconds_truth_parent = time.perf_counter() - t_truth0

    all_kcor: List[Dict[str, Any]] = []
    all_theta: List[Dict[str, Any]] = []
    all_kcor_reps: List[Dict[str, Any]] = []
    all_theta_reps: List[Dict[str, Any]] = []
    kcor_summaries: List[Dict[str, Any]] = []
    timing_by_scenario: Dict[str, Dict[str, float]] = {}

    for scenario_name, scenario_fn, runner_ix in scenario_worklist:
        true_kcor = true_kcor_by_scenario[scenario_name]
        config = SimConfig(seed=args.seed)
        W = max_W_by_scenario[scenario_name]

        _, _, timing_agg = compute_coverage_combined_for_scenario(
            scenario_runner_ix=runner_ix,
            scenario_name=scenario_name,
            true_kcor=true_kcor,
            config=config,
            n_simulations=args.n_simulations,
            n_bootstrap=args.n_bootstrap,
            target_week=args.target_week,
            max_week_index=W,
            skip_theta=bool(args.skip_theta),
            max_workers=int(args.max_workers),
            debug_kcor_call_count=bool(args.debug_kcor_call_count),
            checkpoint_every=int(args.checkpoint_every),
            abort_slow_seconds_per_sim=args.abort_if_slow_seconds_per_sim,
            args_ns=args,
            all_kcor=all_kcor,
            all_kcor_reps=all_kcor_reps,
            all_theta=all_theta,
            all_theta_reps=all_theta_reps,
        )
        timing_by_scenario[scenario_name] = timing_agg
        # summaries already appended inside compute_coverage_combined_for_scenario
        kcor_summaries.append(all_kcor[-1])

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
        truncate=bool(args.truncate),
        skip_theta=bool(args.skip_theta),
    )


    write_outputs = (not args.benchmark_mode) or args.benchmark_write

    if write_outputs:
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
    if write_outputs and manifest_path:
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
            "truncate": bool(args.truncate),
            "max_workers": int(args.max_workers),
            "true_kcor_by_scenario": true_kcor_by_scenario,
            "seconds_truth_parent": seconds_truth_parent,
            "timing_seconds": {
                "by_scenario": timing_by_scenario,
            },
            "outputs": {
                "kcor_summary": os.path.normpath(args.output),
                "kcor_replicates": os.path.normpath(args.output_replicates)
                if args.output_replicates
                else None,
                "theta_summary": os.path.normpath(args.output_theta)
                if not args.skip_theta
                else None,
                "theta_replicates": os.path.normpath(args.output_theta_replicates)
                if not args.skip_theta
                else None,
            },
        }
        if not args.skip_theta:
            manifest["theta_true_by_scenario"] = theta_true_manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote run manifest: {manifest_path}")

    print("\nDone. QA warnings (if any) are listed above.")
    return pd.DataFrame(all_kcor)


if __name__ == "__main__":
    main()

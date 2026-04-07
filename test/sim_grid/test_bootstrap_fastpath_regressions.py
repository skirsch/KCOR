#!/usr/bin/env python3
"""
Regression checks for bootstrap fast path: truncation vs full horizon and Cox on/off.

Documented in the faster-bootstrap plan: KCOR(target_week) and theta_hat by dose must
match (tight tolerance) for a fixed DGP draw and for at least one Poisson-resampled
bootstrap scenario (fixed seeds).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_CODE = os.path.join(os.path.dirname(__file__), "code")
sys.path.insert(0, _CODE)

from compute_bootstrap_coverage import (  # noqa: E402
    bootstrap_resample_cohort_data,
    build_bootstrap_scenario_shallow,
    compute_truncation_W_for_scenario,
)
from generate_sim_grid import (  # noqa: E402
    SimConfig,
    compute_kcor_for_scenario,
    run_scenario_1_gamma_null,
    run_scenario_2_hazard_increase,
)


def _kcor_at(result: dict, week: int) -> float:
    t = result["kcor_trajectory"]
    i = min(week, len(t) - 1)
    return float(t[i])


def _theta_by_dose(result: dict) -> dict:
    out = {}
    for cr in result.get("cohort_results", []):
        out[int(cr["dose"])] = float(cr["theta_hat"])
    return out


def _assert_kcor_theta_close(
    a: dict, b: dict, *, target_week: int, label: str, tol: float = 1e-9
) -> None:
    ka = _kcor_at(a, target_week)
    kb = _kcor_at(b, target_week)
    assert np.isfinite(ka) and np.isfinite(kb), label
    assert abs(ka - kb) <= tol * max(1.0, abs(ka)), f"{label} KCOR week={target_week}"

    ta = _theta_by_dose(a)
    tb = _theta_by_dose(b)
    assert set(ta.keys()) == set(tb.keys()), label
    for d in ta:
        assert abs(ta[d] - tb[d]) <= tol * max(1.0, abs(ta[d])), f"{label} theta dose={d}"


def test_truncation_matches_full_on_dgp_draw() -> None:
    cfg = SimConfig(seed=11)
    scenario = run_scenario_1_gamma_null(cfg)
    target_week = 80
    W = compute_truncation_W_for_scenario(cfg, target_week, scenario)
    r_full = compute_kcor_for_scenario(scenario, cfg, include_cox=False)
    r_trunc = compute_kcor_for_scenario(
        scenario, cfg, include_cox=False, max_week_index=W
    )
    _assert_kcor_theta_close(
        r_full, r_trunc, target_week=target_week, label="dgp_trunc"
    )


def test_truncation_matches_full_on_bootstrap_resample() -> None:
    cfg = SimConfig(seed=13)
    scenario = run_scenario_1_gamma_null(cfg)
    target_week = 80
    base_seed = 900_000
    sim_index = 0
    b = 0
    boot = build_bootstrap_scenario_shallow(
        scenario, b=b, sim_index=sim_index, base_seed=base_seed
    )
    W = compute_truncation_W_for_scenario(cfg, target_week, boot)
    r_full = compute_kcor_for_scenario(boot, cfg, include_cox=False)
    r_trunc = compute_kcor_for_scenario(
        boot, cfg, include_cox=False, max_week_index=W
    )
    _assert_kcor_theta_close(
        r_full, r_trunc, target_week=target_week, label="boot_trunc"
    )


def test_include_cox_does_not_change_kcor_theta_path() -> None:
    """Cox is side output; kcor_trajectory and theta_hat should match with Cox on/off."""
    for seed, runner in [(21, run_scenario_1_gamma_null), (22, run_scenario_2_hazard_increase)]:
        cfg = SimConfig(seed=seed)
        scenario = runner(cfg)
        target_week = 80
        r_on = compute_kcor_for_scenario(scenario, cfg, include_cox=True)
        r_off = compute_kcor_for_scenario(scenario, cfg, include_cox=False)
        _assert_kcor_theta_close(
            r_on, r_off, target_week=target_week, label=f"cox_invariance_seed{seed}"
        )


def test_bootstrap_resample_formula_unchanged_single_cohort() -> None:
    """Spot-check Poisson bootstrap draws match a fixed reference (same seed)."""
    weeks = np.arange(5, dtype=int)
    alive = np.array([1000.0, 950, 900, 850, 800])
    dead = np.array([10.0, 12, 11, 9, 8])
    w2, a2, d2 = bootstrap_resample_cohort_data(weeks, alive, dead, seed=12345)
    assert np.allclose(w2, weeks)
    rng = np.random.default_rng(12345)
    manual_dead0 = float(rng.poisson(dead[0]))
    assert int(d2[0]) == int(min(manual_dead0, int(alive[0])))


def main() -> None:
    test_truncation_matches_full_on_dgp_draw()
    test_truncation_matches_full_on_bootstrap_resample()
    test_include_cox_does_not_change_kcor_theta_path()
    test_bootstrap_resample_formula_unchanged_single_cohort()
    print("bootstrap_fastpath_regressions: OK")


if __name__ == "__main__":
    main()

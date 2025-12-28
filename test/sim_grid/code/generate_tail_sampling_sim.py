#!/usr/bin/env python3
"""
KCOR Tail-Sampling / Bimodal Selection Simulation

Implements the adversarial selection geometry scenario described in paper Appendix B.5.
This tests KCOR's operating characteristics when cohorts are drawn from different
parts of the same underlying frailty distribution:

- Mid-sampled cohort: frailty restricted to central quantiles (25th-75th percentile)
- Tail-sampled cohort: mixture of low + high tails (0-15th and 85th-100th percentiles)

Both cohorts share the same baseline hazard with no treatment effect (negative control),
or a known hazard multiplier is applied (positive control).

Usage:
    python generate_tail_sampling_sim.py --output-dir test/sim_grid/out/tail_sampling/
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import lognorm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TailSamplingConfig:
    """Configuration for tail-sampling simulation."""
    # Time parameters (weeks since cohort entry)
    n_weeks: int = 120
    quiet_window_start: int = 20
    quiet_window_end: int = 80
    effect_window_start: int = 20
    effect_window_end: int = 80
    skip_weeks: int = 2
    
    # Cohort parameters
    n_initial: int = 100_000
    baseline_hazard: float = 0.002  # Weekly baseline hazard rate
    
    # Base frailty distribution: Log-normal with mean=1, variance=0.5
    # For LogNormal(mu, sigma): mean = exp(mu + sigma^2/2), var = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    # To get mean=1, var=0.5: solve numerically or use approximation
    # Approximation: sigma^2 ≈ log(1 + var/mean^2) = log(1.5) ≈ 0.405, so sigma ≈ 0.637
    # mu = -sigma^2/2 for mean=1
    lognormal_sigma: float = 0.637
    
    # Quantile thresholds for selection
    mid_quantile_low: float = 0.25   # 25th percentile
    mid_quantile_high: float = 0.75  # 75th percentile
    tail_low_upper: float = 0.15     # 0-15th percentile (low tail)
    tail_high_lower: float = 0.85    # 85th-100th percentile (high tail)
    
    # Effect multipliers for positive controls
    hazard_increase_r: float = 1.2
    hazard_decrease_r: float = 0.8
    
    # Random seed
    seed: int = 42
    
    # Enrollment date for KCOR_CMR format
    enrollment_date: str = "2021-06-14"


# ============================================================================
# Frailty Distribution Functions
# ============================================================================

def get_lognormal_params(sigma: float) -> Tuple[float, float]:
    """
    Get lognormal parameters for mean=1.
    
    For LogNormal(mu, sigma), mean = exp(mu + sigma^2/2).
    Setting mean=1: mu = -sigma^2/2
    """
    mu = -sigma**2 / 2
    return mu, sigma


def sample_mid_quantile_frailty(
    n: int, 
    sigma: float, 
    q_low: float, 
    q_high: float, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample frailty values from central quantiles of a lognormal distribution.
    
    Samples are restricted to [q_low, q_high] quantiles and renormalized to mean=1.
    """
    mu, sig = get_lognormal_params(sigma)
    
    # Get quantile bounds in frailty space
    z_low = lognorm.ppf(q_low, s=sig, scale=np.exp(mu))
    z_high = lognorm.ppf(q_high, s=sig, scale=np.exp(mu))
    
    # Sample uniformly in probability space within bounds, then transform
    u = rng.uniform(q_low, q_high, size=n)
    frailties = lognorm.ppf(u, s=sig, scale=np.exp(mu))
    
    # Renormalize to mean=1
    frailties = frailties / np.mean(frailties)
    
    return frailties


def sample_tail_mixture_frailty(
    n: int,
    sigma: float,
    low_upper: float,
    high_lower: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample frailty values from mixture of low and high tails.
    
    Equal weights to low tail [0, low_upper] and high tail [high_lower, 1].
    Renormalized to mean=1.
    """
    mu, sig = get_lognormal_params(sigma)
    
    # Determine how many samples from each tail (equal weights)
    n_low = n // 2
    n_high = n - n_low
    
    # Sample from low tail
    u_low = rng.uniform(0, low_upper, size=n_low)
    frailties_low = lognorm.ppf(u_low, s=sig, scale=np.exp(mu))
    
    # Sample from high tail
    u_high = rng.uniform(high_lower, 1.0, size=n_high)
    frailties_high = lognorm.ppf(u_high, s=sig, scale=np.exp(mu))
    
    # Combine
    frailties = np.concatenate([frailties_low, frailties_high])
    rng.shuffle(frailties)
    
    # Renormalize to mean=1
    frailties = frailties / np.mean(frailties)
    
    return frailties


# ============================================================================
# Gamma-Frailty Model Functions (for KCOR fitting)
# ============================================================================

def H_model(t: np.ndarray, k: float, theta: float) -> np.ndarray:
    """Model-predicted cumulative hazard under gamma frailty."""
    t_arr = np.asarray(t, dtype=float)
    if theta <= 1e-10:
        return k * t_arr
    return (1.0 / theta) * np.log1p(theta * k * t_arr)


def invert_gamma_frailty(H_obs: np.ndarray, theta: float) -> np.ndarray:
    """Invert gamma-frailty cumulative hazard to baseline cumulative hazard."""
    H_arr = np.asarray(H_obs, dtype=float)
    if theta <= 1e-10:
        return H_arr.copy()
    return np.expm1(theta * H_arr) / theta


def fit_k_theta_cumhaz(
    t: np.ndarray,
    H_obs: np.ndarray,
    k0: Optional[float] = None,
    theta0: float = 0.1,
) -> Tuple[Tuple[float, float], Dict]:
    """Fit gamma-frailty parameters (k, theta) from observed cumulative hazard."""
    EPS = 1e-12
    
    t_arr = np.asarray(t, dtype=float)
    H_arr = np.asarray(H_obs, dtype=float)
    
    mask = np.isfinite(t_arr) & np.isfinite(H_arr)
    t_arr = t_arr[mask]
    H_arr = H_arr[mask]
    n_obs = len(t_arr)
    
    if n_obs < 2:
        return (np.nan, np.nan), {
            "success": False, "n_obs": n_obs, "rmse_Hobs": np.nan,
            "status": None, "message": "insufficient_points",
        }
    
    if k0 is None:
        try:
            k0_est = float(np.polyfit(t_arr, H_arr, 1)[0])
        except Exception:
            k0_est = 1e-8
        k0 = max(k0_est if np.isfinite(k0_est) else 1e-8, 1e-12)
    else:
        k0 = max(float(k0), 1e-12)
    
    theta0 = max(float(theta0), 0.0)
    
    def _residuals(params):
        k, theta = params
        if (k <= 0.0) or (theta < 0.0):
            return np.ones_like(H_arr) * 1e6
        return H_model(t_arr, k, theta) - H_arr
    
    try:
        res = least_squares(
            _residuals,
            x0=[k0, theta0],
            bounds=([1e-15, 0.0], [1.0, 50.0]),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            max_nfev=2000,
        )
        k_hat, theta_hat = res.x
        
        H_pred = H_model(t_arr, k_hat, theta_hat)
        rmse = np.sqrt(np.mean((H_arr - H_pred) ** 2))
        
        return (k_hat, theta_hat), {
            "success": res.success, "n_obs": n_obs, "rmse_Hobs": rmse,
            "status": res.status, "message": res.message,
        }
    except Exception as e:
        return (np.nan, np.nan), {
            "success": False, "n_obs": n_obs, "rmse_Hobs": np.nan,
            "status": None, "message": str(e),
        }


# ============================================================================
# Simulation Functions
# ============================================================================

def simulate_cohort_with_frailty(
    config: TailSamplingConfig,
    frailties: np.ndarray,
    hazard_multiplier: float = 1.0,
    effect_start: Optional[int] = None,
    effect_end: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate cohort mortality under given frailty distribution.
    
    Individual hazard: h_i(t) = z_i * k * [optional multiplier]
    
    Returns:
        weeks, alive, dead arrays
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)
    
    n_initial = len(frailties)
    weeks = np.arange(config.n_weeks)
    alive = np.zeros(config.n_weeks, dtype=float)
    dead = np.zeros(config.n_weeks, dtype=float)
    
    is_alive = np.ones(n_initial, dtype=bool)
    
    for t in range(config.n_weeks):
        n_alive = np.sum(is_alive)
        alive[t] = n_alive
        
        if n_alive == 0:
            dead[t] = 0
            continue
        
        # Base hazard
        k = config.baseline_hazard
        
        # Apply effect multiplier if in window
        if effect_start is not None and effect_end is not None:
            if effect_start <= t < effect_end:
                k *= hazard_multiplier
        
        # Individual hazards: h_i(t) = z_i * k
        individual_hazards = frailties[is_alive] * k
        p_death = 1 - np.exp(-individual_hazards)
        
        # Stochastic deaths
        deaths_mask = rng.random(n_alive) < p_death
        n_deaths = np.sum(deaths_mask)
        
        dead[t] = n_deaths
        
        alive_indices = np.where(is_alive)[0]
        is_alive[alive_indices[deaths_mask]] = False
    
    return weeks, alive, dead


def run_tail_sampling_scenario(
    config: TailSamplingConfig,
    scenario_type: str = "null",  # "null", "harm", "benefit"
) -> Dict:
    """
    Run tail-sampling simulation scenario.
    
    Args:
        config: Simulation configuration
        scenario_type: "null" (no effect), "harm" (r=1.2), "benefit" (r=0.8)
    
    Returns:
        Dictionary with scenario data
    """
    rng = np.random.default_rng(config.seed)
    
    # Generate frailty distributions
    mid_frailties = sample_mid_quantile_frailty(
        config.n_initial,
        config.lognormal_sigma,
        config.mid_quantile_low,
        config.mid_quantile_high,
        rng,
    )
    
    tail_frailties = sample_tail_mixture_frailty(
        config.n_initial,
        config.lognormal_sigma,
        config.tail_low_upper,
        config.tail_high_lower,
        rng,
    )
    
    # Determine effect parameters
    if scenario_type == "null":
        multiplier = 1.0
        effect_start = None
        effect_end = None
        label = "Tail-Sampling Null"
    elif scenario_type == "harm":
        multiplier = config.hazard_increase_r
        effect_start = config.effect_window_start
        effect_end = config.effect_window_end
        label = "Tail-Sampling + Harm"
    elif scenario_type == "benefit":
        multiplier = config.hazard_decrease_r
        effect_start = config.effect_window_start
        effect_end = config.effect_window_end
        label = "Tail-Sampling + Benefit"
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    # Simulate mid-quantile cohort (Dose 0 = reference)
    weeks_mid, alive_mid, dead_mid = simulate_cohort_with_frailty(
        config, mid_frailties, rng=np.random.default_rng(config.seed)
    )
    
    # Simulate tail-mixture cohort (Dose 1 = treatment)
    weeks_tail, alive_tail, dead_tail = simulate_cohort_with_frailty(
        config, tail_frailties,
        hazard_multiplier=multiplier,
        effect_start=effect_start,
        effect_end=effect_end,
        rng=np.random.default_rng(config.seed + 100),
    )
    
    return {
        "scenario": f"tail_sampling_{scenario_type}",
        "label": label,
        "cohorts": [
            {
                "dose": 0,
                "weeks": weeks_mid,
                "alive": alive_mid,
                "dead": dead_mid,
                "selection": "mid-quantile",
                "frailty_mean": np.mean(mid_frailties),
                "frailty_var": np.var(mid_frailties),
            },
            {
                "dose": 1,
                "weeks": weeks_tail,
                "alive": alive_tail,
                "dead": dead_tail,
                "selection": "tail-mixture",
                "frailty_mean": np.mean(tail_frailties),
                "frailty_var": np.var(tail_frailties),
            },
        ],
        "effect_multiplier": multiplier if scenario_type != "null" else None,
        "effect_window": (effect_start, effect_end) if effect_start else None,
    }


# ============================================================================
# KCOR Processing
# ============================================================================

def compute_kcor_for_scenario(scenario_data: Dict, config: TailSamplingConfig) -> Dict:
    """Compute KCOR and diagnostics for a tail-sampling scenario."""
    results = {
        "scenario": scenario_data["scenario"],
        "label": scenario_data["label"],
        "cohort_results": [],
        "kcor_trajectory": None,
    }
    
    H0_trajectories = {}
    weeks = np.arange(config.n_weeks)
    
    for cohort in scenario_data["cohorts"]:
        dose = cohort["dose"]
        alive = cohort["alive"]
        dead = cohort["dead"]
        
        # Compute hazard
        MR = np.where(alive > 0, dead / alive, 0)
        MR = np.clip(MR, 0, 0.99)
        hazard = -np.log(1 - MR)
        hazard = np.nan_to_num(hazard, nan=0, posinf=0, neginf=0)
        
        # Apply skip-week rule and compute cumulative hazard
        h_eff = np.where(weeks >= config.skip_weeks, hazard, 0)
        H_obs = np.cumsum(h_eff)
        
        # Fit gamma-frailty model over quiet window
        quiet_mask = (weeks >= config.quiet_window_start) & (weeks <= config.quiet_window_end)
        t_quiet = weeks[quiet_mask].astype(float)
        H_quiet = H_obs[quiet_mask]
        
        (k_hat, theta_hat), fit_diag = fit_k_theta_cumhaz(t_quiet, H_quiet)
        
        # Invert to depletion-neutralized
        if np.isfinite(theta_hat) and theta_hat >= 0:
            H0 = invert_gamma_frailty(H_obs, theta_hat)
        else:
            H0 = H_obs.copy()
            theta_hat = 0.0
        
        # Compute R² for linearity diagnostic
        valid_mask = (weeks >= config.skip_weeks) & (weeks <= config.quiet_window_end)
        t_valid = weeks[valid_mask].astype(float)
        H0_valid = H0[valid_mask]
        
        if len(t_valid) > 2:
            coeffs = np.polyfit(t_valid, H0_valid, 1)
            H0_pred = np.polyval(coeffs, t_valid)
            ss_res = np.sum((H0_valid - H0_pred) ** 2)
            ss_tot = np.sum((H0_valid - np.mean(H0_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r_squared = np.nan
        
        cohort_result = {
            "dose": dose,
            "selection": cohort["selection"],
            "k_hat": k_hat,
            "theta_hat": theta_hat,
            "frailty_var_true": cohort["frailty_var"],
            "rmse": fit_diag.get("rmse_Hobs", np.nan),
            "r_squared": r_squared,
            "n_obs": fit_diag.get("n_obs", 0),
            "fit_success": fit_diag.get("success", False),
            "H_obs": H_obs,
            "H0": H0,
        }
        results["cohort_results"].append(cohort_result)
        H0_trajectories[dose] = H0
    
    # Compute KCOR trajectory
    if 0 in H0_trajectories and 1 in H0_trajectories:
        H0_num = H0_trajectories[1]
        H0_den = H0_trajectories[0]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            kcor_raw = np.where(H0_den > 1e-10, H0_num / H0_den, np.nan)
        
        norm_week = config.skip_weeks + 4
        if norm_week < len(kcor_raw) and np.isfinite(kcor_raw[norm_week]) and kcor_raw[norm_week] > 0:
            kcor = kcor_raw / kcor_raw[norm_week]
        else:
            kcor = kcor_raw
        
        results["kcor_trajectory"] = kcor
        
        # Summary statistics over diagnostic window
        diagnostic_mask = (weeks >= 20) & (weeks <= 100)
        kcor_diagnostic = kcor[diagnostic_mask]
        kcor_diagnostic = kcor_diagnostic[np.isfinite(kcor_diagnostic)]
        
        if len(kcor_diagnostic) > 0:
            results["kcor_median"] = float(np.median(kcor_diagnostic))
            results["kcor_mean"] = float(np.mean(kcor_diagnostic))
            results["kcor_std"] = float(np.std(kcor_diagnostic))
            results["kcor_drift"] = float(kcor_diagnostic[-1] - kcor_diagnostic[0])
    
    return results


# ============================================================================
# Output Generation
# ============================================================================

def generate_kcor_cmr_format(scenario_data: Dict, config: TailSamplingConfig) -> pd.DataFrame:
    """Convert scenario data to KCOR_CMR format."""
    rows = []
    start_date = datetime.strptime(config.enrollment_date, "%Y-%m-%d")
    
    for cohort in scenario_data["cohorts"]:
        dose = cohort["dose"]
        weeks = cohort["weeks"]
        alive = cohort["alive"]
        dead = cohort["dead"]
        
        for t in range(len(weeks)):
            date = start_date + timedelta(weeks=int(t))
            iso_year, iso_week, _ = date.isocalendar()
            
            rows.append({
                "ISOweekDied": f"{iso_year}-{iso_week:02d}",
                "DateDied": date.strftime("%Y-%m-%d"),
                "YearOfBirth": 1950,
                "Sex": "F",
                "Dose": dose,
                "Alive": int(alive[t]),
                "Dead": int(dead[t]),
            })
    
    return pd.DataFrame(rows)


def save_results(
    all_results: List[Dict],
    scenario_data_list: List[Dict],
    config: TailSamplingConfig,
    output_dir: str,
):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save KCOR_CMR format data
    data_path = os.path.join(output_dir, "tail_sampling_data.xlsx")
    with pd.ExcelWriter(data_path, engine='xlsxwriter') as writer:
        for scenario_data, result in zip(scenario_data_list, all_results):
            df = generate_kcor_cmr_format(scenario_data, config)
            sheet_name = result["scenario"][:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Save KCOR results
    results_path = os.path.join(output_dir, "tail_sampling_results.xlsx")
    with pd.ExcelWriter(results_path, engine='xlsxwriter') as writer:
        weeks = np.arange(config.n_weeks)
        
        for result in all_results:
            scenario = result["scenario"]
            kcor = result.get("kcor_trajectory", np.full(config.n_weeks, np.nan))
            
            df_kcor = pd.DataFrame({"Week": weeks, "KCOR": kcor})
            df_kcor.to_excel(writer, sheet_name=f"{scenario[:25]}_kcor", index=False)
            
            cohort_rows = []
            for cr in result.get("cohort_results", []):
                cohort_rows.append({
                    "Dose": cr["dose"],
                    "Selection": cr["selection"],
                    "k_hat": cr["k_hat"],
                    "theta_hat": cr["theta_hat"],
                    "frailty_var_true": cr["frailty_var_true"],
                    "RMSE": cr["rmse"],
                    "R_squared": cr["r_squared"],
                    "n_obs": cr["n_obs"],
                    "fit_success": cr["fit_success"],
                })
            df_cohort = pd.DataFrame(cohort_rows)
            df_cohort.to_excel(writer, sheet_name=f"{scenario[:25]}_fit", index=False)
    
    # Save diagnostics CSV
    diag_path = os.path.join(output_dir, "tail_sampling_diagnostics.csv")
    diag_rows = []
    for result in all_results:
        for cr in result.get("cohort_results", []):
            diag_rows.append({
                "scenario": result["scenario"],
                "label": result.get("label", ""),
                "dose": cr["dose"],
                "selection": cr["selection"],
                "k_hat": cr["k_hat"],
                "theta_hat": cr["theta_hat"],
                "frailty_var_true": cr["frailty_var_true"],
                "rmse": cr["rmse"],
                "r_squared": cr["r_squared"],
                "n_obs": cr["n_obs"],
                "fit_success": cr["fit_success"],
                "kcor_median": result.get("kcor_median", np.nan),
                "kcor_drift": result.get("kcor_drift", np.nan),
            })
    
    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(diag_path, index=False)
    
    print(f"Saved data to: {data_path}")
    print(f"Saved results to: {results_path}")
    print(f"Saved diagnostics to: {diag_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate tail-sampling / bimodal selection simulation for KCOR"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="test/sim_grid/out/tail_sampling/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    config = TailSamplingConfig(seed=args.seed)
    
    print("=" * 60)
    print("KCOR Tail-Sampling Simulation")
    print("=" * 60)
    print(f"Mid-quantile cohort: {config.mid_quantile_low*100:.0f}th-{config.mid_quantile_high*100:.0f}th percentile")
    print(f"Tail-mixture cohort: 0-{config.tail_low_upper*100:.0f}th + {config.tail_high_lower*100:.0f}th-100th percentile")
    print(f"Baseline hazard: {config.baseline_hazard}")
    print(f"Effect multipliers: harm={config.hazard_increase_r}, benefit={config.hazard_decrease_r}")
    print(f"Random seed: {config.seed}")
    print()
    
    # Run all scenarios
    scenarios = [
        ("Null (no effect)", "null"),
        ("Harm (r=1.2)", "harm"),
        ("Benefit (r=0.8)", "benefit"),
    ]
    
    all_results = []
    scenario_data_list = []
    
    for name, scenario_type in scenarios:
        print(f"Running {name}...")
        scenario_data = run_tail_sampling_scenario(config, scenario_type)
        scenario_data_list.append(scenario_data)
        
        result = compute_kcor_for_scenario(scenario_data, config)
        
        print(f"  KCOR median (weeks 20-100): {result.get('kcor_median', np.nan):.4f}")
        print(f"  KCOR drift: {result.get('kcor_drift', np.nan):+.4f}")
        for cr in result.get("cohort_results", []):
            print(f"  {cr['selection']}: θ̂={cr['theta_hat']:.4f}, RMSE={cr['rmse']:.6f}, R²={cr['r_squared']:.4f}")
        print()
        
        all_results.append(result)
    
    # Save results
    save_results(all_results, scenario_data_list, config, args.output_dir)
    
    # Print summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    for result in all_results:
        median = result.get("kcor_median", np.nan)
        drift = result.get("kcor_drift", np.nan)
        
        # Check diagnostics
        rmse_ok = all(cr["rmse"] < 0.05 for cr in result["cohort_results"] if np.isfinite(cr["rmse"]))
        r2_ok = all(cr["r_squared"] > 0.95 for cr in result["cohort_results"] if np.isfinite(cr["r_squared"]))
        
        if result["scenario"] == "tail_sampling_null":
            expected = "≈ 1.0"
            in_range = 0.90 <= median <= 1.10 if np.isfinite(median) else False
        elif result["scenario"] == "tail_sampling_harm":
            expected = "> 1.05"
            in_range = median > 1.05 if np.isfinite(median) else False
        elif result["scenario"] == "tail_sampling_benefit":
            expected = "< 0.95"
            in_range = median < 0.95 if np.isfinite(median) else False
        else:
            expected = "?"
            in_range = False
        
        status = "✓" if in_range else "⚠"
        diag_status = "good" if (rmse_ok and r2_ok) else "degraded"
        
        print(f"{result['label']}:")
        print(f"  KCOR median = {median:.4f} (expected {expected}) {status}")
        print(f"  Drift = {drift:+.4f}")
        print(f"  Diagnostics: {diag_status}")
        print()
    
    print("Tail-sampling simulation complete!")
    print()
    print("Key findings for paper:")
    print("- If null scenario KCOR ≈ 1.0 with good diagnostics: gamma model adequate")
    print("- If null scenario KCOR drifts with degraded diagnostics: model misspecification flagged")
    print("- Harm/benefit scenarios should show expected directional effects")


if __name__ == "__main__":
    main()


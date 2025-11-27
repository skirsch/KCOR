#!/usr/bin/env python3
"""
KCOR Mortality Statistical Inference

This module provides statistical methods for uncertainty quantification in KCOR analysis:
- Confidence intervals using Nelson-Aalen variance estimator
- Hypothesis testing (KCOR ≠ 1)
- P-value computation
- Bootstrap methods (optional)

USAGE:
    python kcor_mortality_stats.py <hazard_csv> <output_csv> [--alpha ALPHA] [--bootstrap]

DEPENDENCIES:
    pip install pandas numpy scipy
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import stats
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

EPS = 1e-12


def nelson_aalen_variance(
    deaths: np.ndarray,
    at_risk: np.ndarray,
    scale_factors: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute Nelson-Aalen variance estimator for cumulative hazard.
    
    The variance of cumulative hazard H(t) is:
        Var[H(t)] = Σ (deaths_i / at_risk_i^2)
    
    With slope normalization, we multiply by scale factor squared:
        Var[H_adj(t)] = Σ (deaths_i / at_risk_i^2) * s_i^2
    
    Args:
        deaths: Array of death counts per time period
        at_risk: Array of persons at risk per time period
        scale_factors: Optional array of scale factors for slope normalization
    
    Returns:
        Array of variance estimates per time point
    """
    # Avoid division by zero
    at_risk_safe = np.maximum(at_risk, EPS)
    
    # Variance increments: deaths / at_risk^2
    var_increments = deaths / (at_risk_safe ** 2)
    
    # Apply scale factors if provided
    if scale_factors is not None:
        var_increments = var_increments * (scale_factors ** 2)
    
    # Cumulative variance
    var_cumulative = np.cumsum(var_increments)
    
    return var_cumulative


def compute_kcor_confidence_intervals(
    haz: pd.DataFrame,
    ref_cohort: str = "dose0_unvaccinated",
    vax_cohorts: Optional[list] = None,
    alpha: float = 0.05,
    normalization_t: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute confidence intervals for KCOR ratios using Nelson-Aalen variance.
    
    This follows the approach in KCOR.py:
    - Compute post-anchor cumulative hazard increments
    - Use Nelson-Aalen variance with slope normalization
    - Compute CI on log scale, then exponentiate
    
    Args:
        haz: Hazard DataFrame with columns: cohort, t, deaths, at_risk, hazard_adj, slope
        ref_cohort: Reference cohort name
        vax_cohorts: List of vaccinated cohort names (if None, auto-detect)
        alpha: Significance level (default: 0.05 for 95% CI)
        normalization_t: Time point for normalization (if None, uses first available)
    
    Returns:
        KCOR DataFrame with CI_lower and CI_upper columns added
    """
    print(f"Computing confidence intervals (alpha={alpha})...")
    
    # Auto-detect vaccinated cohorts
    if vax_cohorts is None:
        all_cohorts = haz["cohort"].unique()
        vax_cohorts = [c for c in all_cohorts if c != ref_cohort]
    
    # Get reference cohort data
    ref_haz = haz[haz["cohort"] == ref_cohort].sort_values("t").reset_index(drop=True)
    
    if len(ref_haz) == 0:
        raise ValueError(f"Reference cohort '{ref_cohort}' not found")
    
    # Determine normalization time point
    if normalization_t is None:
        normalization_t = min(4, len(ref_haz) - 1)  # Default to t=4 or first available
    
    # Compute scale factors for reference (slope normalization)
    ref_scale_factors = np.exp(-ref_haz["slope"].fillna(0) * ref_haz["t"])
    
    # Compute reference cumulative hazard and variance
    ref_ch = ref_haz["hazard_adj"].cumsum().values
    ref_var = nelson_aalen_variance(
        ref_haz["deaths"].values,
        ref_haz["at_risk"].values,
        ref_scale_factors.values
    )
    
    # Get reference values at normalization point
    ref_ch_t0 = ref_ch[normalization_t] if normalization_t < len(ref_ch) else ref_ch[0]
    ref_var_t0 = ref_var[normalization_t] if normalization_t < len(ref_var) else ref_var[0]
    
    # Process each vaccinated cohort
    kcor_rows = []
    
    for vax_cohort in vax_cohorts:
        vax_haz = haz[haz["cohort"] == vax_cohort].sort_values("t").reset_index(drop=True)
        
        if len(vax_haz) == 0:
            print(f"  Warning: Cohort {vax_cohort} not found, skipping")
            continue
        
        # Compute scale factors for vaccinated cohort
        vax_scale_factors = np.exp(-vax_haz["slope"].fillna(0) * vax_haz["t"])
        
        # Compute vaccinated cumulative hazard and variance
        vax_ch = vax_haz["hazard_adj"].cumsum().values
        vax_var = nelson_aalen_variance(
            vax_haz["deaths"].values,
            vax_haz["at_risk"].values,
            vax_scale_factors.values
        )
        
        # Get vaccinated values at normalization point
        vax_ch_t0 = vax_ch[normalization_t] if normalization_t < len(vax_ch) else vax_ch[0]
        vax_var_t0 = vax_var[normalization_t] if normalization_t < len(vax_var) else vax_var[0]
        
        # Compute post-anchor increments
        ref_delta_ch = ref_ch - ref_ch_t0
        vax_delta_ch = vax_ch - vax_ch_t0
        
        # Post-anchor variances (approximate: variance of increment ≈ variance at t - variance at t0)
        ref_delta_var = np.maximum(ref_var - ref_var_t0, EPS)
        vax_delta_var = np.maximum(vax_var - vax_var_t0, EPS)
        
        # Compute KCOR ratio (post-anchor)
        kcor_ratio = np.where(
            ref_delta_ch > EPS,
            vax_delta_ch / ref_delta_ch,
            np.nan
        )
        
        # Compute log-scale standard error
        # SE[log(KCOR)] ≈ sqrt(Var[ΔH_vax] / (ΔH_vax)^2 + Var[ΔH_ref] / (ΔH_ref)^2)
        log_se = np.sqrt(
            vax_delta_var / (vax_delta_ch ** 2 + EPS) +
            ref_delta_var / (ref_delta_ch ** 2 + EPS)
        )
        
        # Compute confidence intervals on log scale
        z_score = stats.norm.ppf(1 - alpha / 2)
        log_kcor = np.log(np.maximum(kcor_ratio, EPS))
        
        ci_lower = np.exp(log_kcor - z_score * log_se)
        ci_upper = np.exp(log_kcor + z_score * log_se)
        
        # Create DataFrame for this cohort
        for i, t in enumerate(vax_haz["t"].values):
            if not np.isnan(kcor_ratio[i]):
                kcor_rows.append({
                    "cohort": vax_cohort,
                    "t": t,
                    "kcor_ratio": kcor_ratio[i],
                    "ci_lower": ci_lower[i],
                    "ci_upper": ci_upper[i],
                    "log_se": log_se[i],
                    "ref_ch": ref_ch[i] if i < len(ref_ch) else np.nan,
                    "vax_ch": vax_ch[i] if i < len(vax_ch) else np.nan,
                    "ref_var": ref_var[i] if i < len(ref_var) else np.nan,
                    "vax_var": vax_var[i] if i < len(vax_var) else np.nan
                })
    
    kcor_df = pd.DataFrame(kcor_rows)
    
    if len(kcor_df) == 0:
        raise ValueError("No KCOR ratios could be computed")
    
    return kcor_df


def test_kcor_hypothesis(
    kcor_df: pd.DataFrame,
    null_value: float = 1.0,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test hypothesis that KCOR ≠ null_value (two-sided test).
    
    Args:
        kcor_df: KCOR DataFrame with kcor_ratio and log_se columns
        null_value: Null hypothesis value (default: 1.0)
        alpha: Significance level
    
    Returns:
        DataFrame with p_value column added
    """
    print(f"Testing hypothesis: KCOR ≠ {null_value} (alpha={alpha})...")
    
    # Compute test statistic on log scale
    log_kcor = np.log(kcor_df["kcor_ratio"] / null_value)
    z_scores = log_kcor / kcor_df["log_se"]
    
    # Two-sided p-value
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # Add to DataFrame
    kcor_df = kcor_df.copy()
    kcor_df["z_score"] = z_scores
    kcor_df["p_value"] = p_values
    kcor_df["significant"] = p_values < alpha
    
    return kcor_df


def bootstrap_kcor_ci(
    pm: pd.DataFrame,
    haz: pd.DataFrame,
    ref_cohort: str = "dose0_unvaccinated",
    vax_cohort: str = "dose1",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute bootstrap confidence intervals for KCOR.
    
    This resamples persons (with replacement) and recomputes KCOR for each bootstrap sample.
    
    Args:
        pm: Person-month DataFrame
        haz: Hazard DataFrame (for reference)
        ref_cohort: Reference cohort name
        vax_cohort: Vaccinated cohort name
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with bootstrap statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print(f"Computing bootstrap CI (n={n_bootstrap})...")
    
    # Get unique persons per cohort
    ref_persons = pm[pm["cohort"] == ref_cohort]["id_zeny"].unique()
    vax_persons = pm[pm["cohort"] == vax_cohort]["id_zeny"].unique()
    
    if len(ref_persons) == 0 or len(vax_persons) == 0:
        raise ValueError("Insufficient persons for bootstrap")
    
    # Get maximum time
    max_t = pm["t"].max()
    
    bootstrap_kcors = []
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")
        
        # Resample persons with replacement
        ref_sample = np.random.choice(ref_persons, size=len(ref_persons), replace=True)
        vax_sample = np.random.choice(vax_persons, size=len(vax_persons), replace=True)
        
        # Create bootstrap person-month table
        ref_pm_boot = pm[pm["id_zeny"].isin(ref_sample) & (pm["cohort"] == ref_cohort)]
        vax_pm_boot = pm[pm["id_zeny"].isin(vax_sample) & (pm["cohort"] == vax_cohort)]
        
        # Compute hazards
        ref_events = ref_pm_boot.groupby("t")["event"].sum()
        ref_risk = ref_pm_boot.groupby("t")["id_zeny"].nunique()
        ref_hazards = ref_events / (ref_risk + EPS)
        
        vax_events = vax_pm_boot.groupby("t")["event"].sum()
        vax_risk = vax_pm_boot.groupby("t")["id_zeny"].nunique()
        vax_hazards = vax_events / (vax_risk + EPS)
        
        # Compute cumulative hazards
        ref_ch = ref_hazards.cumsum()
        vax_ch = vax_hazards.cumsum()
        
        # Compute KCOR (simple ratio, no normalization for bootstrap)
        kcor_boot = (vax_ch / (ref_ch + EPS)).values
        
        # Store final KCOR
        if len(kcor_boot) > 0:
            bootstrap_kcors.append(kcor_boot[-1])
    
    bootstrap_kcors = np.array(bootstrap_kcors)
    
    # Compute percentiles
    ci_lower = np.percentile(bootstrap_kcors, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_kcors, (1 - alpha / 2) * 100)
    median = np.median(bootstrap_kcors)
    
    return {
        "bootstrap_kcors": bootstrap_kcors,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median": median,
        "mean": np.mean(bootstrap_kcors),
        "std": np.std(bootstrap_kcors)
    }


def add_statistical_inference_to_hazards(
    haz_path: str,
    output_path: str,
    ref_cohort: str = "dose0_unvaccinated",
    alpha: float = 0.05,
    normalization_t: Optional[int] = None
) -> pd.DataFrame:
    """
    Add statistical inference to existing hazard CSV file.
    
    Args:
        haz_path: Path to kcor_hazard_adjusted.csv
        output_path: Path to save output CSV with CIs
        ref_cohort: Reference cohort name
        alpha: Significance level
        normalization_t: Normalization time point
    
    Returns:
        KCOR DataFrame with statistical inference
    """
    print(f"Loading hazards from {haz_path}...")
    haz = pd.read_csv(haz_path)
    
    # Compute confidence intervals
    kcor_df = compute_kcor_confidence_intervals(
        haz,
        ref_cohort=ref_cohort,
        alpha=alpha,
        normalization_t=normalization_t
    )
    
    # Test hypotheses
    kcor_df = test_kcor_hypothesis(kcor_df, null_value=1.0, alpha=alpha)
    
    # Save
    kcor_df.to_csv(output_path, index=False)
    print(f"Saved results with statistical inference to {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Statistical Inference Summary")
    print("=" * 80)
    
    for cohort in kcor_df["cohort"].unique():
        cohort_data = kcor_df[kcor_df["cohort"] == cohort]
        final_row = cohort_data.iloc[-1]
        
        print(f"\n{cohort}:")
        print(f"  Final KCOR: {final_row['kcor_ratio']:.4f}")
        print(f"  95% CI: [{final_row['ci_lower']:.4f}, {final_row['ci_upper']:.4f}]")
        print(f"  P-value: {final_row['p_value']:.4f}")
        print(f"  Significant (p<{alpha}): {final_row['significant']}")
    
    return kcor_df


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="KCOR Mortality Statistical Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("hazard_csv", help="Path to kcor_hazard_adjusted.csv")
    parser.add_argument("output_csv", help="Path to save output CSV with statistical inference")
    parser.add_argument("--ref-cohort", default="dose0_unvaccinated", help="Reference cohort (default: dose0_unvaccinated)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    parser.add_argument("--normalization-t", type=int, help="Normalization time point (default: auto)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hazard_csv):
        print(f"Error: Input file not found: {args.hazard_csv}")
        return 1
    
    try:
        add_statistical_inference_to_hazards(
            haz_path=args.hazard_csv,
            output_path=args.output_csv,
            ref_cohort=args.ref_cohort,
            alpha=args.alpha,
            normalization_t=args.normalization_t
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import os
    import sys
    exit(main())


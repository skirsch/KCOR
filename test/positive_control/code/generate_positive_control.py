"""
Generate positive control test data for KCOR validation.

This script generates synthetic cohort data in KCOR_CMR format where:
1. Two cohorts (Dose 0 and Dose 1) share the same baseline hazard
2. A known hazard multiplier is injected into Dose 1 over a specified time window
3. After KCOR normalization, the ratio should deviate from 1.0 in the expected direction

Usage:
    python generate_positive_control.py [--harm] [--benefit] [--both] [--output OUTPUT_PATH]
    
Output: Excel file in KCOR_CMR format ready for KCOR.py processing
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PositiveControlConfig:
    """Configuration for positive control data generation."""
    # Cohort parameters
    n_initial: int = 100_000  # Initial cohort size
    n_weeks: int = 150  # Total weeks of follow-up
    
    # Baseline hazard (weekly probability of death)
    baseline_hazard: float = 0.002  # ~10% cumulative mortality over 50 weeks
    
    # Gamma frailty parameters for selection-induced curvature
    theta_dose0: float = 0.5  # Frailty variance for dose 0 (control)
    theta_dose1: float = 1.0  # Frailty variance for dose 1 (treatment)
    
    # Injection window (weeks)
    injection_start_week: int = 20
    injection_end_week: int = 80
    
    # Hazard multiplier for injection (r > 1 = harm, r < 1 = benefit)
    hazard_multiplier: float = 1.2  # Default: 20% increased hazard (harm)
    
    # Enrollment date
    enrollment_iso_week: str = "2021_24"
    enrollment_date: str = "2021-06-14"
    
    # Year of birth for synthetic cohort
    year_of_birth: int = 1950
    
    # Random seed for reproducibility
    seed: int = 42


def gamma_frailty_survival(t: np.ndarray, k: float, theta: float) -> np.ndarray:
    """
    Compute survival probability under gamma-frailty model.
    
    S(t) = (1 + theta * k * t)^(-1/theta)
    
    where k is baseline hazard rate and theta is frailty variance.
    """
    if theta <= 0:
        return np.exp(-k * t)
    return np.power(1 + theta * k * t, -1.0 / theta)


def gamma_frailty_hazard(t: np.ndarray, k: float, theta: float) -> np.ndarray:
    """
    Compute instantaneous hazard under gamma-frailty model.
    
    h(t) = k / (1 + theta * k * t)
    
    This is the cohort-level hazard after frailty mixing.
    """
    if theta <= 0:
        return k * np.ones_like(t, dtype=float)
    return k / (1 + theta * k * t)


def simulate_cohort_with_injection(
    config: PositiveControlConfig,
    theta: float,
    apply_injection: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate cohort mortality with optional hazard injection.
    
    Returns:
        weeks: Array of week indices
        alive: Array of alive counts at start of each week
        dead: Array of deaths during each week
    """
    np.random.seed(config.seed if not apply_injection else config.seed + 1)
    
    weeks = np.arange(config.n_weeks)
    alive = np.zeros(config.n_weeks, dtype=float)
    dead = np.zeros(config.n_weeks, dtype=float)
    
    current_alive = float(config.n_initial)
    
    for t in range(config.n_weeks):
        alive[t] = current_alive
        
        if current_alive <= 0:
            dead[t] = 0
            continue
        
        # Base hazard from gamma-frailty model
        base_h = gamma_frailty_hazard(np.array([t]), config.baseline_hazard, theta)[0]
        
        # Apply injection if in window and this is the treatment cohort
        h = base_h
        if apply_injection:
            if config.injection_start_week <= t < config.injection_end_week:
                h = base_h * config.hazard_multiplier
        
        # Convert hazard to probability (discrete time approximation)
        p_death = 1 - np.exp(-h)
        
        # Compute deaths (use expected value for deterministic simulation)
        deaths = current_alive * p_death
        
        dead[t] = deaths
        current_alive -= deaths
    
    return weeks, alive, dead


def generate_iso_weeks(start_date: str, n_weeks: int) -> List[str]:
    """Generate ISO week strings starting from enrollment date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    iso_weeks = []
    for w in range(n_weeks):
        date = start + timedelta(weeks=w)
        iso_year, iso_week, _ = date.isocalendar()
        iso_weeks.append(f"{iso_year}-{iso_week:02d}")
    return iso_weeks


def generate_date_strings(start_date: str, n_weeks: int) -> List[str]:
    """Generate date strings starting from enrollment date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = []
    for w in range(n_weeks):
        date = start + timedelta(weeks=w)
        dates.append(date.strftime("%Y-%m-%d"))
    return dates


def build_kcor_cmr_dataframe(
    config: PositiveControlConfig,
    scenario: str = "harm",
) -> pd.DataFrame:
    """
    Build a KCOR_CMR format DataFrame for positive control.
    
    Args:
        config: Configuration parameters
        scenario: "harm" (r>1), "benefit" (r<1), or "null" (r=1)
    
    Returns:
        DataFrame in KCOR_CMR format with columns:
        ISOweekDied, DateDied, YearOfBirth, Sex, Dose, Alive, Dead
    """
    # Set multiplier based on scenario
    if scenario == "harm":
        config.hazard_multiplier = 1.2
    elif scenario == "benefit":
        config.hazard_multiplier = 0.8
    else:  # null
        config.hazard_multiplier = 1.0
    
    # Generate ISO weeks and dates
    iso_weeks = generate_iso_weeks(config.enrollment_date, config.n_weeks)
    date_strings = generate_date_strings(config.enrollment_date, config.n_weeks)
    
    rows = []
    
    # Dose 0: Control cohort (no injection)
    weeks_d0, alive_d0, dead_d0 = simulate_cohort_with_injection(
        config, theta=config.theta_dose0, apply_injection=False
    )
    
    for sex in ['F', 'M']:
        for t in range(config.n_weeks):
            rows.append({
                'ISOweekDied': iso_weeks[t],
                'DateDied': date_strings[t],
                'YearOfBirth': config.year_of_birth,
                'Sex': sex,
                'Dose': 0,
                'Alive': int(alive_d0[t] / 2),  # Split by sex
                'Dead': int(dead_d0[t] / 2),
            })
    
    # Dose 1: Treatment cohort (with injection)
    weeks_d1, alive_d1, dead_d1 = simulate_cohort_with_injection(
        config, theta=config.theta_dose1, apply_injection=True
    )
    
    for sex in ['F', 'M']:
        for t in range(config.n_weeks):
            rows.append({
                'ISOweekDied': iso_weeks[t],
                'DateDied': date_strings[t],
                'YearOfBirth': config.year_of_birth,
                'Sex': sex,
                'Dose': 1,
                'Alive': int(alive_d1[t] / 2),  # Split by sex
                'Dead': int(dead_d1[t] / 2),
            })
    
    df = pd.DataFrame(rows)
    return df[['ISOweekDied', 'DateDied', 'YearOfBirth', 'Sex', 'Dose', 'Alive', 'Dead']]


def generate_positive_control(
    output_path: str,
    scenarios: List[str] = ["harm", "benefit"],
    config: PositiveControlConfig = None,
) -> dict:
    """
    Generate positive control test data file.
    
    Args:
        output_path: Path to output Excel file
        scenarios: List of scenarios to generate ("harm", "benefit")
        config: Optional configuration override
    
    Returns:
        Dictionary with scenario parameters for documentation
    """
    if config is None:
        config = PositiveControlConfig()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {}
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for scenario in scenarios:
            # Create scenario-specific config
            sc_config = PositiveControlConfig(
                n_initial=config.n_initial,
                n_weeks=config.n_weeks,
                baseline_hazard=config.baseline_hazard,
                theta_dose0=config.theta_dose0,
                theta_dose1=config.theta_dose1,
                injection_start_week=config.injection_start_week,
                injection_end_week=config.injection_end_week,
                hazard_multiplier=1.2 if scenario == "harm" else 0.8,
                enrollment_iso_week=config.enrollment_iso_week,
                enrollment_date=config.enrollment_date,
                year_of_birth=config.year_of_birth,
                seed=config.seed,
            )
            
            df = build_kcor_cmr_dataframe(sc_config, scenario)
            
            # Sheet name based on scenario
            sheet_name = f"{config.enrollment_iso_week}_{scenario}"
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            
            results[scenario] = {
                'sheet_name': sheet_name,
                'hazard_multiplier': sc_config.hazard_multiplier,
                'injection_window': f"week {sc_config.injection_start_week} to {sc_config.injection_end_week}",
                'expected_kcor_direction': '> 1' if scenario == 'harm' else '< 1',
                'n_weeks': sc_config.n_weeks,
                'theta_dose0': sc_config.theta_dose0,
                'theta_dose1': sc_config.theta_dose1,
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate positive control test data for KCOR validation"
    )
    parser.add_argument(
        "--output", "-o",
        default="test/positive_control/data/KCOR_positive_control.xlsx",
        help="Output Excel file path"
    )
    parser.add_argument(
        "--harm", action="store_true",
        help="Generate harm scenario (r=1.2)"
    )
    parser.add_argument(
        "--benefit", action="store_true",
        help="Generate benefit scenario (r=0.8)"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Generate both harm and benefit scenarios (default)"
    )
    parser.add_argument(
        "--multiplier", "-r", type=float, default=None,
        help="Custom hazard multiplier (overrides --harm/--benefit)"
    )
    parser.add_argument(
        "--injection-start", type=int, default=20,
        help="Start week of injection window (default: 20)"
    )
    parser.add_argument(
        "--injection-end", type=int, default=80,
        help="End week of injection window (default: 80)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Determine scenarios
    scenarios = []
    if args.both or (not args.harm and not args.benefit):
        scenarios = ["harm", "benefit"]
    else:
        if args.harm:
            scenarios.append("harm")
        if args.benefit:
            scenarios.append("benefit")
    
    # Create config
    config = PositiveControlConfig(
        injection_start_week=args.injection_start,
        injection_end_week=args.injection_end,
        seed=args.seed,
    )
    
    if args.multiplier is not None:
        config.hazard_multiplier = args.multiplier
    
    # Generate data
    results = generate_positive_control(
        output_path=args.output,
        scenarios=scenarios,
        config=config,
    )
    
    print(f"Generated positive control data: {args.output}")
    print("\nScenarios:")
    for scenario, info in results.items():
        print(f"\n  {scenario.upper()}:")
        print(f"    Sheet: {info['sheet_name']}")
        print(f"    Hazard multiplier: {info['hazard_multiplier']}")
        print(f"    Injection window: {info['injection_window']}")
        print(f"    Expected KCOR direction: {info['expected_kcor_direction']}")


if __name__ == "__main__":
    main()


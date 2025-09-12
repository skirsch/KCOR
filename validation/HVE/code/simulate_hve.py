#!/usr/bin/env python3
"""
simulate_hve.py — Stylized dynamic HVE simulator

Models dynamic Healthy Vaccinee Effect (HVE) with an exponential decay
and generates:
  1) A plot of deaths/week by cohort (Dose 0, Dose 1, Dose 2)
  2) A CSV (and optional XLSX) with week-by-week deaths/week per cohort

Assumptions (default values can be overridden by CLI args):
- No true biological effect; this isolates selection (HVE) only.
- Vaccinated cohorts (Dose 1 and Dose 2) have a hazard multiplier:
      m_v(t) = 1 - max_effect * exp(-ln(2) * t / half_life_weeks)
- Dose 0 hazard multiplier is determined by mass-balance so that the
  population-average hazard multiplier stays at 1.0:
      f0*m0 + f1*m1 + f2*m2 = 1  =>  m0 = (1 - f1*m1 - f2*m2) / f0
- Fixed cohorts are defined at time T = D2 + t0_d2 weeks (default 5w).
  We simulate W additional weeks after T. For vaccinated cohorts, the
  time since last dose is (t0_* + w).

Example:
    python simulate_hve.py --N 1000000 --uptake 0.8 --frac_d2_within_vax 0.9 \
      --baseline_rate 0.0005 --max_effect 0.75 --half_life 10 \
      --t0_d2 5 --t0_d1 5 --weeks 30 \
      --out_csv hve.csv --out_png hve.png

Author: ChatGPT (GPT-5 Thinking)
"""
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Params:
    N: int = 1_000_000
    uptake: float = 0.80                # fraction vaccinated at T (Dose1 + Dose2)
    frac_d2_within_vax: float = 0.90    # among vaccinated at T, fraction that are Dose 2
    baseline_rate: float = 0.0005       # deaths/person/week
    max_effect: float = 0.75            # 0..1, 0.75 = 75% max suppression
    half_life_weeks: float = 10.0       # HVE half-life (weeks)
    t0_d2: float = 5.0                  # weeks since D2 at T
    t0_d1: float = 5.0                  # weeks since D1 at T
    weeks: int = 30                     # simulate weeks after T
    mass_balance: bool = True           # enforce population average hazard = 1.0
    out_csv: str = "hve_output.csv"
    out_xlsx: str = ""                  # optional
    out_png: str = "hve_plot.png"
    show: bool = False                  # show interactive plot


def fractions_at_T(uptake: float, frac_d2_within_vax: float) -> Tuple[float, float, float]:
    uptake = float(uptake)
    fd2 = uptake * float(frac_d2_within_vax)
    fd1 = uptake * (1.0 - float(frac_d2_within_vax))
    f0  = 1.0 - uptake
    # Guard against numerical drift
    s = fd2 + fd1 + f0
    if not math.isclose(s, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        fd2, fd1, f0 = fd2 / s, fd1 / s, f0 / s
    return fd2, fd1, f0


def vaccinated_multiplier(t_since: np.ndarray, max_effect: float, half_life_weeks: float) -> np.ndarray:
    lam = math.log(2.0) / float(half_life_weeks)
    return 1.0 - float(max_effect) * np.exp(-lam * t_since)


def simulate(p: Params) -> pd.DataFrame:
    f2, f1, f0 = fractions_at_T(p.uptake, p.frac_d2_within_vax)
    weeks = np.arange(0, int(p.weeks) + 1)
    t2 = p.t0_d2 + weeks
    t1 = p.t0_d1 + weeks

    m2 = vaccinated_multiplier(t2, p.max_effect, p.half_life_weeks)
    m1 = vaccinated_multiplier(t1, p.max_effect, p.half_life_weeks)

    if p.mass_balance:
        # Solve for m0 by mass balance so total hazard = baseline
        m0 = (1.0 - f1 * m1 - f2 * m2) / f0 if f0 > 0 else np.zeros_like(m1)
        # Clip to avoid negative hazard (if parameters are extreme)
        m0 = np.maximum(m0, 0.0)
    else:
        m0 = np.ones_like(m1)

    # Deaths per week (counts)
    D2 = p.N * f2 * p.baseline_rate * m2
    D1 = p.N * f1 * p.baseline_rate * m1
    D0 = p.N * f0 * p.baseline_rate * m0

    df = pd.DataFrame({
        "week_since_T": weeks,
        "dose2_deaths_per_week": D2,
        "dose1_deaths_per_week": D1,
        "dose0_deaths_per_week": D0,
        "total_deaths_per_week": D2 + D1 + D0,
        "m2": m2,
        "m1": m1,
        "m0": m0,
        "f2": f2,
        "f1": f1,
        "f0": f0,
    })
    return df


def make_plot(df: pd.DataFrame, p: Params):
    plt.figure(figsize=(8, 5))
    plt.plot(df["week_since_T"], df["dose2_deaths_per_week"], label="Dose 2")
    plt.plot(df["week_since_T"], df["dose1_deaths_per_week"], label="Dose 1")
    plt.plot(df["week_since_T"], df["dose0_deaths_per_week"], label="Dose 0")
    plt.xlabel("Weeks since T (T = D2 + t0_d2, fixed cohorts at T)")
    plt.ylabel("Deaths per week (counts)")
    plt.title(
        f"Dynamic HVE (half-life={p.half_life_weeks}w, max={p.max_effect:.0%})\n"
        f"N={p.N:,}; uptake={p.uptake:.0%} (D2={p.frac_d2_within_vax:.0%} of vax); "
        f"baseline={p.baseline_rate}/wk"
    )
    plt.legend()
    plt.tight_layout()
    if p.out_png:
        plt.savefig(p.out_png, dpi=150, bbox_inches="tight")
    if p.show:
        plt.show()
    plt.close()


def pct_change(x0: float, x1: float) -> float:
    return 100.0 * (x1 - x0) / x0 if x0 != 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stylized dynamic HVE simulator")
    ap.add_argument("--N", type=int, default=1_000_000, help="Total population (default: 1,000,000)")
    ap.add_argument("--uptake", type=float, default=0.80, help="Fraction vaccinated at T (Dose1 + Dose2)")
    ap.add_argument("--frac_d2_within_vax", type=float, default=0.90, help="Among vaccinated, fraction in Dose 2 at T")
    ap.add_argument("--baseline_rate", type=float, default=0.0005, help="Deaths per person per week")
    ap.add_argument("--max_effect", type=float, default=0.75, help="HVE max suppression (0..1)")
    ap.add_argument("--half_life", dest="half_life_weeks", type=float, default=10.0, help="HVE half-life in weeks")
    ap.add_argument("--t0_d2", type=float, default=5.0, help="Weeks since D2 at T")
    ap.add_argument("--t0_d1", type=float, default=5.0, help="Weeks since D1 at T")
    ap.add_argument("--weeks", type=int, default=30, help="Weeks to simulate after T (inclusive)")
    ap.add_argument("--no_mass_balance", action="store_true", help="Disable mass-balance (let totals vary)")
    ap.add_argument("--out_csv", type=str, default="hve_output.csv", help="Path to CSV output")
    ap.add_argument("--out_xlsx", type=str, default="", help="Optional XLSX output path")
    ap.add_argument("--out_png", type=str, default="hve_plot.png", help="Path to PNG plot")
    ap.add_argument("--show", action="store_true", help="Show the plot interactively")

    args = ap.parse_args(argv)
    p = Params(
        N=args.N,
        uptake=args.uptake,
        frac_d2_within_vax=args.frac_d2_within_vax,
        baseline_rate=args.baseline_rate,
        max_effect=args.max_effect,
        half_life_weeks=args.half_life_weeks,
        t0_d2=args.t0_d2,
        t0_d1=args.t0_d1,
        weeks=args.weeks,
        mass_balance=not args.no_mass_balance,
        out_csv=args.out_csv,
        out_xlsx=args.out_xlsx,
        out_png=args.out_png,
        show=args.show,
    )

    df = simulate(p)

    # Save outputs
    if p.out_csv:
        df.to_csv(p.out_csv, index=False)
    if p.out_xlsx:
        try:
            df.to_excel(p.out_xlsx, index=False)
        except Exception as e:
            print(f"Warning: failed to write XLSX ({e}). CSV was written to {p.out_csv}.", file=sys.stderr)

    # Plot
    make_plot(df, p)

    # Quick console summary
    w0, wK = 0, min(20, p.weeks)
    d2 = df["dose2_deaths_per_week"].to_numpy()
    d1 = df["dose1_deaths_per_week"].to_numpy()
    d0 = df["dose0_deaths_per_week"].to_numpy()
    tot = df["total_deaths_per_week"].to_numpy()
    print(f"Week {w0}: D2={d2[w0]:.1f}, D1={d1[w0]:.1f}, D0={d0[w0]:.1f}, Total={tot[w0]:.1f}")
    print(f"Week {wK}: D2={d2[wK]:.1f}, D1={d1[wK]:.1f}, D0={d0[wK]:.1f}, Total={tot[wK]:.1f}")
    print(f"% change ({w0}→{wK}w): Dose2={pct_change(d2[w0], d2[wK]):+.1f}%, "
          f"Dose1={pct_change(d1[w0], d1[wK]):+.1f}%, Dose0={pct_change(d0[w0], d0[wK]):+.1f}%")


if __name__ == "__main__":
    main()

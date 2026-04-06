#!/usr/bin/env python3
"""
Plot bootstrap coverage summaries from ``compute_bootstrap_coverage.py`` outputs.

Reads:
  - ``bootstrap_coverage.csv`` (KCOR summaries)
  - ``bootstrap_coverage_theta.csv`` (theta summaries)

Writes (under output directory by default):
  - ``fig_bootstrap_coverage_kcor.png`` [+ optional ``.pdf``]
  - ``fig_bootstrap_coverage_kcor_ciwidth.png`` [+ ``.pdf``]
  - ``fig_bootstrap_coverage_theta.png`` [+ ``.pdf``]
  - ``fig_bootstrap_coverage_theta_ciwidth.png`` [+ ``.pdf``]

Scenario order is fixed for comparability across runs.

Usage::

    python plot_bootstrap_coverage.py --out-dir test/sim_grid/out --pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent ordering (must match compute_bootstrap_coverage.py scenario names)
SCENARIO_ORDER = [
    "Gamma-frailty null",
    "Injected effect (harm)",
    "Injected effect (benefit)",
    "Non-gamma frailty",
    "Sparse events",
]


def _scenario_sort_key(series: pd.Series) -> pd.Series:
    order_map = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    return series.map(lambda x: order_map.get(str(x), 999))


def _load_kcor(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "scenario" not in df.columns:
        raise ValueError(f"{path}: expected column 'scenario'")
    df = df.copy()
    df["_k"] = _scenario_sort_key(df["scenario"])
    return df.sort_values("_k").drop(columns=["_k"])


def _load_theta(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "scenario" not in df.columns:
        raise ValueError(f"{path}: expected column 'scenario'")
    df = df.copy()
    df["_k"] = _scenario_sort_key(df["scenario"])
    return df.sort_values("_k").drop(columns=["_k"])


def _coverage_fraction(row: pd.Series, col_prop: str, col_pct: str) -> float:
    if col_prop in row.index and np.isfinite(row[col_prop]):
        v = float(row[col_prop])
        if v > 1.0 + 1e-6:
            return v / 100.0
        return v
    if col_pct in row.index and np.isfinite(row[col_pct]):
        return float(row[col_pct]) / 100.0
    return float("nan")


def _save(fig: plt.Figure, base: str, pdf: bool) -> None:
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    if pdf:
        fig.savefig(base + ".pdf", bbox_inches="tight")


def plot_kcor_coverage(df: pd.DataFrame, out_base: str, pdf: bool) -> None:
    scenarios = df["scenario"].tolist()
    y = [_coverage_fraction(row, "coverage_proportion", "coverage_percent") for _, row in df.iterrows()]
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, y, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axhline(0.95, color="crimson", linestyle="--", linewidth=1.5, label="Nominal 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("KCOR bootstrap coverage (per simulated dataset)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_base, pdf)
    plt.close(fig)


def plot_kcor_ciwidth(df: pd.DataFrame, out_base: str, pdf: bool) -> None:
    scenarios = df["scenario"].tolist()
    if "mean_ci_width" not in df.columns:
        raise ValueError("KCOR summary CSV missing mean_ci_width")
    y = df["mean_ci_width"].astype(float).tolist()
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, y, color="seagreen", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("Mean CI width (KCOR)")
    ax.set_title("Mean bootstrap CI width for KCOR at evaluation week")
    fig.tight_layout()
    _save(fig, out_base, pdf)
    plt.close(fig)


def plot_theta_coverage(df: pd.DataFrame, out_base: str, pdf: bool) -> None:
    scenarios = df["scenario"].tolist()
    n = len(scenarios)
    x = np.arange(n)
    w = 0.36
    y0 = [
        _coverage_fraction(row, "coverage_theta_d0", "coverage_theta_d0_percent")
        for _, row in df.iterrows()
    ]
    y1 = [
        _coverage_fraction(row, "coverage_theta_d1", "coverage_theta_d1_percent")
        for _, row in df.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, y0, width=w, label="dose 0", color="tab:orange", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, y1, width=w, label="dose 1", color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.axhline(0.95, color="crimson", linestyle="--", linewidth=1.5, label="Nominal 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(r"$\theta$ bootstrap coverage by dose")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_base, pdf)
    plt.close(fig)


def plot_theta_ciwidth(df: pd.DataFrame, out_base: str, pdf: bool) -> None:
    scenarios = df["scenario"].tolist()
    n = len(scenarios)
    x = np.arange(n)
    w = 0.36
    c0, c1 = "mean_ci_width_theta_d0", "mean_ci_width_theta_d1"
    if c0 not in df.columns or c1 not in df.columns:
        raise ValueError(f"Theta summary CSV missing {c0} / {c1} (re-run compute_bootstrap_coverage.py)")
    y0 = df[c0].astype(float).tolist()
    y1 = df[c1].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, y0, width=w, label="dose 0", color="tab:orange", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, y1, width=w, label="dose 1", color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel(r"Mean CI width ($\theta$)")
    ax.set_title(r"Mean bootstrap CI width for $\hat\theta$ by dose")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_base, pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bootstrap coverage summary figures")
    parser.add_argument(
        "--out-dir",
        default="test/sim_grid/out",
        help="Directory for PNG/PDF outputs",
    )
    parser.add_argument(
        "--kcor-csv",
        default=None,
        help="KCOR summary CSV (default: <out-dir>/bootstrap_coverage.csv)",
    )
    parser.add_argument(
        "--theta-csv",
        default=None,
        help="Theta summary CSV (default: <out-dir>/bootstrap_coverage_theta.csv)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write vector PDFs alongside PNGs",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    kcor_path = args.kcor_csv or os.path.join(out_dir, "bootstrap_coverage.csv")
    theta_path = args.theta_csv or os.path.join(out_dir, "bootstrap_coverage_theta.csv")

    if not os.path.isfile(kcor_path):
        print(f"Missing KCOR summary: {kcor_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    df_k = _load_kcor(kcor_path)
    plot_kcor_coverage(df_k, os.path.join(out_dir, "fig_bootstrap_coverage_kcor"), args.pdf)
    plot_kcor_ciwidth(df_k, os.path.join(out_dir, "fig_bootstrap_coverage_kcor_ciwidth"), args.pdf)
    print(f"Wrote KCOR figures under {out_dir}")

    if os.path.isfile(theta_path):
        df_t = _load_theta(theta_path)
        plot_theta_coverage(df_t, os.path.join(out_dir, "fig_bootstrap_coverage_theta"), args.pdf)
        c0, c1 = "mean_ci_width_theta_d0", "mean_ci_width_theta_d1"
        if c0 in df_t.columns and c1 in df_t.columns:
            plot_theta_ciwidth(
                df_t, os.path.join(out_dir, "fig_bootstrap_coverage_theta_ciwidth"), args.pdf
            )
        else:
            print(
                f"Skipping theta CI-width figure (need {c0} and {c1}; re-run compute_bootstrap_coverage.py)"
            )
        print(f"Wrote theta figures under {out_dir}")
    else:
        print(f"Skipping theta plots (missing {theta_path})")


if __name__ == "__main__":
    main()

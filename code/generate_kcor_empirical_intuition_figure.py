"""
Generate an empirical KCOR intuition figure for the preprint.

The selected comparison is deterministic:
- grouped cohorts only
- 24-month follow-up
- non-extreme, near-null KCOR trajectory
- visible reduction in between-cohort curvature after normalization
- largest available cohort size among valid candidates
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REFERENCE_COHORT = "dose0_unvaccinated"


@dataclass(frozen=True)
class Candidate:
    config_id: str
    cohort: str
    sample_size: float
    final_kcor: float
    curvature_drop: float
    raw_gap: float
    adj_gap: float


def friendly_label(cohort: str) -> str:
    mapping = {
        "dose0_unvaccinated": "Dose 0 reference",
        "dose1": "Dose 1",
        "dose2": "Dose 2",
        "dose3": "Dose 3",
        "dose4": "Dose 4",
        "dose3plus": "Dose 3+",
    }
    return mapping.get(cohort, cohort.replace("_", " "))


def load_candidate_series(base_dir: Path, config_id: str, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(base_dir / config_id / "raw" / "kcor_hazard_raw.csv")
    adj = pd.read_csv(base_dir / config_id / "raw" / "kcor_hazard_adjusted.csv")
    ratios = pd.read_csv(base_dir / config_id / "results" / "kcor_ratios.csv")

    keep = [REFERENCE_COHORT, cohort]
    raw_pair = raw[raw["cohort"].isin(keep)].copy()
    adj_pair = adj[adj["cohort"].isin(keep)].copy()
    ratio_pair = ratios[ratios["cohort"] == cohort].copy()
    if raw_pair.empty or adj_pair.empty or ratio_pair.empty:
        raise ValueError(f"Missing series for config={config_id}, cohort={cohort}")

    raw_pair = raw_pair.sort_values(["cohort", "t"]).reset_index(drop=True)
    adj_pair = adj_pair.sort_values(["cohort", "t"]).reset_index(drop=True)
    ratio_pair = ratio_pair.sort_values("t").reset_index(drop=True)
    raw_pair["cum_hazard"] = raw_pair.groupby("cohort")["hazard"].cumsum()
    return raw_pair, adj_pair, ratio_pair


def pair_curvature_gap(df: pd.DataFrame, value_col: str, cohort: str) -> float:
    ref_vals = (
        df[df["cohort"] == REFERENCE_COHORT]
        .sort_values("t")[value_col]
        .to_numpy(dtype=float)
    )
    cohort_vals = df[df["cohort"] == cohort].sort_values("t")[value_col].to_numpy(dtype=float)
    if ref_vals.size < 3 or cohort_vals.size < 3:
        return float("inf")
    return float(np.mean(np.abs(np.diff(cohort_vals, n=2) - np.diff(ref_vals, n=2))))


def assert_panel_consistency(raw_pair: pd.DataFrame, adj_pair: pd.DataFrame, ratio_pair: pd.DataFrame, cohort: str) -> tuple[list[str], list[float]]:
    expected_cohorts = [REFERENCE_COHORT, cohort]
    raw_cohorts = sorted(raw_pair["cohort"].unique().tolist())
    adj_cohorts = sorted(adj_pair["cohort"].unique().tolist())
    if raw_cohorts != sorted(expected_cohorts):
        raise ValueError(f"Raw panel cohort mismatch: expected {expected_cohorts}, got {raw_cohorts}")
    if adj_cohorts != sorted(expected_cohorts):
        raise ValueError(f"Normalized panel cohort mismatch: expected {expected_cohorts}, got {adj_cohorts}")

    raw_times = sorted(raw_pair["t"].unique().tolist())
    adj_times = sorted(adj_pair["t"].unique().tolist())
    ratio_times = sorted(ratio_pair["t"].unique().tolist())
    if raw_times != adj_times or raw_times != ratio_times:
        raise ValueError("Panel time ranges do not match across raw, normalized, and KCOR series")
    return expected_cohorts, raw_times


def select_candidate(summary_path: Path, base_dir: Path) -> Candidate:
    summary = pd.read_csv(summary_path)
    filtered = summary[
        (~summary["separate_doses"].astype(bool))
        & (summary["max_fu_months"] == 24)
        & (summary["cohort"].isin(["dose1", "dose2"]))
        & (summary["final_kcor"].between(0.85, 1.15))
        & (summary["max_kcor"] <= 1.5)
        & (summary["min_kcor"] > 0.0)
    ].copy()
    if filtered.empty:
        raise ValueError("No valid candidate rows found for empirical intuition figure")

    candidates: list[Candidate] = []
    fallback: list[Candidate] = []
    for row in filtered.itertuples(index=False):
        raw_pair, adj_pair, _ = load_candidate_series(base_dir, row.config_id, row.cohort)
        cohort_at_risk = float(raw_pair[raw_pair["cohort"] == row.cohort]["at_risk"].iloc[0])
        ref_at_risk = float(raw_pair[raw_pair["cohort"] == REFERENCE_COHORT]["at_risk"].iloc[0])
        raw_gap = pair_curvature_gap(raw_pair, "cum_hazard", row.cohort)
        adj_gap = pair_curvature_gap(adj_pair, "cum_hazard_adj", row.cohort)
        candidate = Candidate(
            config_id=str(row.config_id),
            cohort=str(row.cohort),
            sample_size=min(cohort_at_risk, ref_at_risk),
            final_kcor=float(row.final_kcor),
            curvature_drop=raw_gap - adj_gap,
            raw_gap=raw_gap,
            adj_gap=adj_gap,
        )
        fallback.append(candidate)
        if candidate.curvature_drop > 0.0:
            candidates.append(candidate)

    ranked = candidates if candidates else fallback
    ranked.sort(
        key=lambda item: (
            -item.sample_size,
            -item.curvature_drop,
            abs(item.final_kcor - 1.0),
            item.config_id,
            item.cohort,
        )
    )
    chosen = ranked[0]
    print(
        "[KCOR] Selected empirical intuition pair:",
        f"config={chosen.config_id}",
        f"cohort={chosen.cohort}",
        f"reference={REFERENCE_COHORT}",
        f"sample_size={chosen.sample_size:.0f}",
        f"final_kcor={chosen.final_kcor:.3f}",
        f"curvature_drop={chosen.curvature_drop:.6f}",
        flush=True,
    )
    return chosen


def write_audit_log(
    output_path: Path,
    chosen: Candidate,
    cohorts: list[str],
    time_values: list[float],
    *,
    panel_c_mode: str,
    anchor_t0: float | None,
) -> Path:
    audit_path = output_path.with_suffix(".json")
    payload = {
        "figure": output_path.name,
        "selection": {
            "config_id": chosen.config_id,
            "cohort": chosen.cohort,
            "reference_cohort": REFERENCE_COHORT,
            "cohorts": cohorts,
            "sample_size_min": chosen.sample_size,
            "final_kcor": chosen.final_kcor,
            "curvature_drop": chosen.curvature_drop,
            "raw_gap": chosen.raw_gap,
            "adj_gap": chosen.adj_gap,
        },
        "strata": {
            "grouping": "grouped cohorts",
            "time_window_label": chosen.config_id,
            "event_time_start": min(time_values),
            "event_time_end": max(time_values),
            "n_timepoints": len(time_values),
        },
        "panel_c": {
            "mode": panel_c_mode,
            "anchor_t0": anchor_t0,
            "reference_level": 1.0,
        },
    }
    audit_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return audit_path


def add_direct_label(ax, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=9,
        va="center",
        ha="left",
    )


def plot(output_path: Path, base_dir: Path, summary_path: Path, *, anchor_t0: float | None = None) -> None:
    chosen = select_candidate(summary_path, base_dir)
    raw_pair, adj_pair, ratio_pair = load_candidate_series(base_dir, chosen.config_id, chosen.cohort)
    cohorts, time_values = assert_panel_consistency(raw_pair, adj_pair, ratio_pair, chosen.cohort)
    panel_c_mode = "anchored_kcor" if anchor_t0 is not None else "kcor"

    ratio_y = ratio_pair["kcor_ratio"].astype(float).copy()
    if anchor_t0 is not None:
        anchor_row = ratio_pair.loc[np.isclose(ratio_pair["t"].astype(float), anchor_t0)]
        if anchor_row.empty:
            raise ValueError(f"Requested anchored KCOR at t0={anchor_t0}, but no such time point exists")
        anchor_value = float(anchor_row.iloc[0]["kcor_ratio"])
        if not np.isfinite(anchor_value) or anchor_value == 0.0:
            raise ValueError(f"Invalid anchor KCOR value at t0={anchor_t0}: {anchor_value}")
        ratio_y = ratio_y / anchor_value
    else:
        anchor_value = None

    colors = {
        REFERENCE_COHORT: "0.40",
        chosen.cohort: "tab:blue",
    }
    labels = {
        REFERENCE_COHORT: friendly_label(REFERENCE_COHORT),
        chosen.cohort: friendly_label(chosen.cohort),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True)
    x_min = float(min(time_values))
    x_max = float(max(time_values))

    ax = axes[0]
    for cohort in [REFERENCE_COHORT, chosen.cohort]:
        part = raw_pair[raw_pair["cohort"] == cohort].sort_values("t")
        ax.plot(part["t"], part["cum_hazard"], color=colors[cohort], lw=2.2)
        add_direct_label(ax, float(part["t"].iloc[-1]), float(part["cum_hazard"].iloc[-1]), labels[cohort], colors[cohort])
    ax.set_title("A. Raw cohort trajectories", fontsize=12)
    ax.set_xlabel("Follow-up time")
    ax.set_ylabel("Observed cumulative hazard")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    for cohort in [REFERENCE_COHORT, chosen.cohort]:
        part = adj_pair[adj_pair["cohort"] == cohort].sort_values("t")
        ax.plot(part["t"], part["cum_hazard_adj"], color=colors[cohort], lw=2.2)
        add_direct_label(ax, float(part["t"].iloc[-1]), float(part["cum_hazard_adj"].iloc[-1]), labels[cohort], colors[cohort])
    ax.set_title("B. After KCOR normalization", fontsize=12)
    ax.set_xlabel("Follow-up time")
    ax.set_ylabel("Normalized cumulative hazard")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[2]
    ax.plot(ratio_pair["t"], ratio_y, color=colors[chosen.cohort], lw=2.2)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.5)
    add_direct_label(
        ax,
        float(ratio_pair["t"].iloc[-1]),
        float(ratio_y.iloc[-1]),
        f"Anchored KCOR(t; {anchor_t0:g})" if anchor_t0 is not None else "KCOR(t)",
        colors[chosen.cohort],
    )
    ax.set_title(f"C. Anchored KCOR(t; {anchor_t0:g})" if anchor_t0 is not None else "C. KCOR(t)", fontsize=12)
    ax.set_xlabel("Follow-up time")
    ax.set_ylabel(f"Anchored KCOR(t; {anchor_t0:g})" if anchor_t0 is not None else "KCOR(t)")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, ls=":", alpha=0.4)

    fig.text(
        0.5,
        0.01,
        (
            "Deterministically selected grouped Czech comparison: "
            f"{chosen.config_id}, {friendly_label(chosen.cohort)} vs {friendly_label(REFERENCE_COHORT)}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    audit_path = write_audit_log(
        output_path,
        chosen,
        cohorts,
        time_values,
        panel_c_mode=panel_c_mode,
        anchor_t0=anchor_t0,
    )
    print(f"[KCOR] Wrote audit log to {audit_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="data/Czech2/kcor_mortality_output/sensitivity",
        help="Sensitivity output directory containing summary_all_configs.csv and per-config outputs.",
    )
    parser.add_argument(
        "--summary",
        default="data/Czech2/kcor_mortality_output/sensitivity/summary_all_configs.csv",
        help="Summary CSV used for deterministic cohort selection.",
    )
    parser.add_argument(
        "--output",
        default="documentation/preprint/figures/fig_kcor_empirical_intuition.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--anchor-t0",
        type=float,
        default=None,
        help="Optional anchor time for anchored KCOR in panel C. If omitted, plot raw KCOR(t).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(
        output_path=Path(args.output),
        base_dir=Path(args.base_dir),
        summary_path=Path(args.summary),
        anchor_t0=args.anchor_t0,
    )


if __name__ == "__main__":
    main()

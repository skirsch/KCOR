"""Reproduce the Cox experiment's KCOR points and their person-bootstrap intervals."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[2] / "test/sim_grid"
FIGURE = (HERE.parent.parent / "iterative review/revised/documentation/preprint/figures"
          / "fig_cox_bias_kcor_vs_theta.png")
sys.path.insert(0, str(STUDY / "code"))
import generate_cox_bias_sim as sim


def bootstrap_hazard(group, rng, horizon, draws=1999):
    n = len(group)
    events = np.sort(group.loc[group.event == 1, "time"].to_numpy())
    assert np.all(np.diff(events) > 0) and events[-1] < horizon
    assert (group.loc[group.event == 0, "time"] == horizon).all()
    # One category per event time, one shared category for administrative censoring.
    # Multinomial weights resample people with replacement within their cohort.
    probabilities = np.r_[np.full(len(events), 1/n), 1-len(events)/n]
    result = np.empty(draws)
    for start in range(0, draws, 25):
        stop = min(start+25, draws)
        dead = rng.multinomial(n, probabilities, size=stop-start)[:, :-1]
        at_risk = n-np.cumsum(dead, axis=1)+dead
        result[start:stop] = np.sum(dead/at_risk, axis=1)
    point = np.sum(1/(n-np.arange(len(events))))
    return point, result


def intervals():
    cfg = sim.CoxBiasConfig()
    df = pd.read_csv(STUDY / "out/cox_bias_results.csv")
    rows = []
    for index, row in enumerate(df.itertuples()):
        people = sim.simulate_two_cohorts(row.theta_B, cfg)
        rng = np.random.default_rng([20261122, index])
        a, b = [bootstrap_hazard(group, rng, cfg.censor_horizon)
                for _, group in people.groupby("cohort", sort=True)]
        point = (sim.invert_gamma_frailty(np.asarray(b[0]), row.theta_B)+1e-12)/(a[0]+1e-12)
        np.testing.assert_allclose(point, row.kcor_asymptote, rtol=1e-11)
        ratios = (sim.invert_gamma_frailty(b[1], row.theta_B)+1e-12)/(a[1]+1e-12)
        lo, hi = np.quantile(ratios, [.025, .975])
        rows.append((row.theta_B, row.kcor_asymptote, lo, hi))
    return pd.DataFrame(rows, columns=["theta_B", "kcor_asymptote", "lower95", "upper95"])


def plot(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    y = df.kcor_asymptote
    bars = ax.errorbar(df.theta_B, y, yerr=[y-df.lower95, df.upper95-y],
                       fmt="o-", color="#1f77b4", linewidth=2, markersize=6,
                       elinewidth=1.3, capsize=2.5, capthick=1.3)
    null = ax.axhline(1., color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Frailty variance θ (cohort B)", fontsize=12)
    ax.set_ylabel("KCOR asymptote (end of follow-up)", fontsize=12)
    ax.set_title("KCOR under synthetic null: remains centered near 1 across θ", fontsize=13)
    ax.grid(True, alpha=.3)
    ax.set_axisbelow(True)
    ax.legend([bars, null], ["KCOR asymptote (95% CI)", "True KCOR = 1"],
              loc="upper left", fontsize=10)
    fig.tight_layout()
    try:
        with TemporaryDirectory(dir=FIGURE.parent) as staging:
            rendered = Path(staging) / FIGURE.name
            fig.savefig(rendered, dpi=300, bbox_inches="tight")
            rendered.replace(FIGURE)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Check recomputed points and endpoints against the saved CSV before plotting.")
    args = parser.parse_args()
    archive = HERE / "figure_intervals.csv"
    data = intervals()
    if args.verify:
        saved = pd.read_csv(archive, float_precision="round_trip")
        pd.testing.assert_frame_equal(data, saved, check_exact=False, rtol=1e-12, atol=0)
        print(f"Verified {len(data)} estimates and {2 * len(data)} interval endpoints.")
    else:
        data.to_csv(archive, index=False)
    plot(data)

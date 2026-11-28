"""Build the planned fifth-week figures and optional fourth-week sensitivity replay."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2.7"))
from comorbidity_sweep import N, STUDY, study, summarize


def plot(grid, h_ref):
    figures = HERE.parent.parent / "iterative review/revised/documentation/preprint/figures"
    grid = grid.assign(KCOR_asymptote_margin95=grid.upper95-grid.KCOR_asymptote_mean)
    panels = [grid[grid.contamination_type == mode+"_agg"].to_dict("records")
              for mode in ("dynamic", "comorbidity", "combined")]
    study.plot_figures(figures, *panels, h_ref, mean_intervals=True)


def add_intervals(grid):
    margin = t.ppf(.975, grid.n_success-1)*grid.KCOR_asymptote_std/np.sqrt(grid.n_success)
    return grid.assign(lower95=grid.KCOR_asymptote_mean-margin,
                       upper95=grid.KCOR_asymptote_mean+margin)


def build_primary():
    archive = STUDY / "out"
    grid = add_intervals(pd.read_csv(archive / "quiet_contamination_agg.csv"))
    meta = pd.read_csv(archive / "quiet_contamination_meta.csv", index_col=0)["value"]
    grid.to_csv(HERE / "planned-anchor-grid.csv", index=False)
    plot(grid, float(meta.k_simulator))


def main():
    output = HERE / "workbook-anchor"
    output.mkdir(exist_ok=True)
    archive = STUDY / "out"
    meta = pd.read_csv(archive / "quiet_contamination_meta.csv", index_col=0)["value"]
    k, gamma = map(float, (meta.k_simulator, meta.gamma_per_week))
    skip = study.K.DYNAMIC_HVE_SKIP_WEEKS
    assert (skip, study.K.KCOR_NORMALIZATION_WEEKS, study.QUIET_T_REBASED_MIN) == (2, 4, 4)
    times = study.build_time_arrays(study.N_WEEKS, skip)
    fit_weeks = study.K.get_gompertz_theta_fit_config()["k_anchor_weeks"]

    def sweep(mode, dynamic, comorbidity, seed, replicates):
        study.N_REPS = study.MIN_SUCCESS_REPS = replicates
        return study.sweep_mode(mode, dynamic, comorbidity, k, gamma, k, *times,
                                fit_weeks, gamma, seed, workbook_anchor=True)

    dynamic = sweep("dynamic", study.EPSILON_DYNAMIC, [0.], 42, 20)
    combined = sweep("combined", study.EPSILON_DYNAMIC_POS, study.EPSILON_COMORB_POS, 20042, 20)
    comorbidity = sweep("comorbidity", [0.], study.EPSILON_COMORB, 10042, 20)
    extended = pd.DataFrame(sweep("comorbidity", [0.], study.EPSILON_COMORB, 10042, N))
    extended = extended[extended.rep >= 0]
    keys = ["epsilon_comorb", "rep"]
    expected = pd.read_csv(HERE / "anchor_week_pairs.csv").set_index(keys).sort_index()
    observed = extended.set_index(keys).sort_index()
    pd.testing.assert_index_equal(observed.index, expected.index)
    np.testing.assert_allclose(observed.KCOR_asymptote, expected.fourth, rtol=0, atol=1e-8)

    results = pd.DataFrame(dynamic + comorbidity + combined)
    runs, grid = results[results.rep >= 0], results[results.rep < 0].copy()
    identifiers = ["contamination_type", "epsilon_dyn", "epsilon_comorb", "rep"]
    historical = pd.read_csv(archive / "quiet_contamination_long.csv").set_index(identifiers).sort_index()
    revised = runs.set_index(identifiers).sort_index()
    pd.testing.assert_index_equal(revised.index, historical.index)
    for column in ("theta0_hat_A", "theta0_hat_B"):
        print(f"Largest difference from archived {column}: {abs(revised[column]-historical[column]).max():.3g}")
        np.testing.assert_allclose(revised[column], historical[column], rtol=0, atol=5e-6)
    grid = add_intervals(grid)
    runs.to_csv(output / "runs.csv", index=False)
    grid.to_csv(output / "grid.csv", index=False)
    summarize(extended).to_csv(output / "comorbidity_200_summary.csv", index=False)
    print(f"Verified {len(runs)} grid runs and {len(extended)} extended comorbidity runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-sensitivity", action="store_true")
    if parser.parse_args().rerun_sensitivity:
        main()
    build_primary()

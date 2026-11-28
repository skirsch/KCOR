"""Extend the published comorbidity sweep to 200 runs per level."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[2] / "test/quiet_window_contamination"
sys.path.insert(0, str(STUDY))
import run_contamination_test as study

N = 200


def simulate(level):
    # Use the calibration archived with the original 20-replicate results.
    meta = pd.read_csv(STUDY / "out/quiet_contamination_meta.csv", index_col=0)["value"]
    h = float(meta["k_simulator"])
    gamma = float(meta["gamma_per_week"])
    anchor = study.K.get_gompertz_theta_fit_config()["k_anchor_weeks"]
    times = study.build_time_arrays(study.N_WEEKS, study.K.DYNAMIC_HVE_SKIP_WEEKS)
    study.N_REPS = N
    # Require complete seed coverage before using sweep_mode's replicate IDs.
    study.MIN_SUCCESS_REPS = N
    rows = study.sweep_mode("comorbidity", [0.0], [level], h, gamma, h, *times,
                            anchor, gamma, base_seed=10042)
    return pd.DataFrame(row for row in rows if row["rep"] >= 0)


def summarize(runs):
    base = runs.loc[runs.epsilon_comorb == 0, "KCOR_asymptote"]
    rows = []
    for level, group in runs.groupby("epsilon_comorb"):
        values = group.KCOR_asymptote
        mean = values.mean()
        margin = t.ppf(0.975, len(values) - 1) * values.sem()
        row = dict(added_weekly_hazard=level, runs=len(values), mean=mean,
                   lower95=mean-margin, upper95=mean+margin, sd=values.std(),
                   above_one=int((values > 1).sum()), difference=mean-base.mean())
        if level > 0:
            # The original seed depends on amplitude. Samples at different
            # amplitudes are independent, including those with the same rep ID.
            ci = ttest_ind(values, base, equal_var=False).confidence_interval()
            row.update(difference_lower95=ci.low, difference_upper95=ci.high)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as pool:
        runs = pd.concat(pool.map(simulate, study.EPSILON_COMORB), ignore_index=True)
    runs = runs[["epsilon_comorb", "rep", "KCOR_asymptote", "fit_status_A", "fit_status_B"]]
    runs.to_csv(HERE / "comorbidity_200_runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(HERE / "comorbidity_200_summary.csv", index=False)
    print(summary.to_string(index=False))
    old = pd.read_csv(STUDY / "out/quiet_contamination_long.csv")
    old = old[old.contamination_type == "comorbidity"]
    keys = ["epsilon_comorb", "rep"]
    original = old.set_index(keys).KCOR_asymptote.sort_index()
    replay = runs[runs.rep < 20].set_index(keys).KCOR_asymptote.sort_index()
    pd.testing.assert_index_equal(original.index, replay.index)
    print("Maximum absolute difference across the first 100 archived estimates:",
          np.max(np.abs(original - replay)))

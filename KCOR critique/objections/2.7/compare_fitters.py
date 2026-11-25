"""Refit the same simulated counts in each arm of the comorbidity sweep."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

from comorbidity_sweep import HERE, N, STUDY, study, summarize

sys.path.insert(0, str(HERE.parent / "1.3/paired-seed-fit"))
import compare as edition


def run_level(level):
    original_fit = study.K.fit_theta0_gompertz
    fits = {"original": original_fit}
    if level == 0:
        edition.initialize("revised")
        fits.update(checkpoint_fixed=edition.production_fit, checkpoint_joint=edition.joint_fit)
    edition.initialize("wave-entry")
    fits.update(revised_fixed=edition.production_fit, revised_joint=edition.joint_fit)
    meta = pd.read_csv(STUDY / "out/quiet_contamination_meta.csv", index_col=0)["value"]
    k, gamma = map(float, (meta.k_simulator, meta.gamma_per_week))
    skip = study.K.DYNAMIC_HVE_SKIP_WEEKS
    weeks, rebased, quiet = study.build_time_arrays(study.N_WEEKS, skip)
    anchor = study.K.get_gompertz_theta_fit_config()["k_anchor_weeks"]
    generate = partial(study.simulate_one_cohort, n0=study.N_INITIAL, n_weeks=study.N_WEEKS,
                       k=k, gamma_wk=gamma, skip=skip)

    def contamination(week):
        return level if study.QUIET_T_REBASED_MIN <= week <= study.QUIET_T_REBASED_MAX else 0.

    rows = []
    for rep in range(N):
        # Same level-specific streams as sweep_mode(..., base_seed=10042).
        code = round(float(level) * 1e12) % (2**31)
        rng = np.random.default_rng(np.random.SeedSequence([10042, 2, 0, code, rep]))
        a = generate(rng, theta0=study.THETA0_A, contamination_fn=contamination)
        b = generate(rng, theta0=study.THETA0_B, contamination_fn=None)
        for name, fitter in fits.items():
            with patch.object(study.K, "fit_theta0_gompertz", fitter):
                fitted = [study.cohort_pipeline(dead, alive, weeks, rebased, quiet, anchor, gamma)
                          for alive, dead in (a, b)]
            (_, _, da, _, ha), (_, _, db, _, hb) = fitted
            endpoint = np.nan
            if da["success"] and db["success"]:
                endpoint = study.kcor_asymptote_value(study.anchored_kcor(ha, hb, skip), hb)
            rows.append(dict(epsilon_comorb=level, rep=rep, fitter=name, KCOR_asymptote=endpoint,
                             status_A=da["status"], status_B=db["status"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as pool:
        runs = pd.concat(pool.map(run_level, study.EPSILON_COMORB), ignore_index=True)
    runs.to_csv(HERE / "fitter_comparison_runs.csv", index=False)
    keys = ["epsilon_comorb", "rep"]
    archived = pd.read_csv(HERE / "comorbidity_200_runs.csv").set_index(keys).sort_index()
    endpoints = runs.pivot(index=keys, columns="fitter", values="KCOR_asymptote").sort_index()
    pd.testing.assert_index_equal(endpoints.index, archived.index)
    np.testing.assert_allclose(endpoints.original, archived.KCOR_asymptote, atol=1e-8, rtol=0)
    stats = []
    contrasts = []
    for name, group in runs.groupby("fitter"):
        successful = group.dropna(subset=["KCOR_asymptote"])
        stats.append(summarize(successful).assign(fitter=name))
    for level, group in endpoints.groupby(level=0):
        pairs = [("original", "revised_joint"), ("revised_fixed", "revised_joint")]
        if level == 0:
            pairs += [("original", "checkpoint_fixed"), ("checkpoint_fixed", "checkpoint_joint"),
                      ("checkpoint_joint", "revised_joint")]
        for previous, current in pairs:
            delta = (group[current] - group[previous]).dropna()
            mean = delta.mean()
            margin = t.ppf(.975, len(delta)-1) * delta.sem()
            contrasts.append(dict(level=level, previous=previous, current=current, pairs=len(delta),
                                  mean_difference=mean, lower95=mean-margin, upper95=mean+margin))
    pd.concat(stats).to_csv(HERE / "fitter_comparison_means.csv", index=False)
    pd.DataFrame(contrasts).to_csv(HERE / "fitter_comparison_paired.csv", index=False)

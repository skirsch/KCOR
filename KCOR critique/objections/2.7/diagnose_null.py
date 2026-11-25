"""Audit the zero-contamination endpoint using the original generator and saved seeds."""
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t

from compare_fitters import HERE, N, STUDY, edition, study


class RecordProbabilities:
    """The generator's null weekly probability is independent of realized counts."""

    def __init__(self):
        self.probabilities = []

    def binomial(self, alive, probability):
        self.probabilities.append(probability)
        return 0


def audit(job):
    size, start, stop = job
    edition.initialize("wave-entry")
    fits = {"joint": edition.joint_fit}
    if size == study.N_INITIAL:
        fits["original"] = study.K.fit_theta0_gompertz
    meta = pd.read_csv(STUDY / "out/quiet_contamination_meta.csv", index_col=0)["value"]
    k, gamma = float(meta.k_simulator), float(meta.gamma_per_week)
    skip = study.K.DYNAMIC_HVE_SKIP_WEEKS
    weeks, rebased, quiet = study.build_time_arrays(study.N_WEEKS, skip)
    anchor = study.K.get_gompertz_theta_fit_config()["k_anchor_weeks"]
    truth = (study.THETA0_A, study.THETA0_B)
    results = []
    for rep in range(start, stop):
        rng = np.random.default_rng(np.random.SeedSequence([10042, 2, 0, 0, max(rep, 0)]))
        cohorts = []
        for theta in truth:
            stream = RecordProbabilities() if rep == -1 else rng
            alive, deaths = study.simulate_one_cohort(stream, size, study.N_WEEKS, k, gamma,
                                                      theta, skip, None)
            if rep == -1:
                p = np.array(stream.probabilities)
                alive = size * np.r_[1., np.cumprod(1.-p[:-1])]
                deaths = alive * p
            cohorts.append((deaths, alive))
        for name, fit in fits.items():
            with patch.object(study.K, "fit_theta0_gompertz", fit):
                fitted = [study.cohort_pipeline(d, a, weeks, rebased, quiet, anchor, gamma)
                          for d, a in cohorts]
            record = dict(size=size, rep=rep, fitter=name)
            curves = []
            for arm, true_theta, (rate, theta, diag, observed, adjusted) in zip("ab", truth, fitted):
                assert diag["success"] and np.isfinite(theta)
                record.update({f"theta_{arm}": theta, f"k_{arm}": rate, f"status_{arm}": diag["status"]})
                curves.append((study.K.invert_gamma_frailty(observed, true_theta), adjusted))
            for label, i, j in [("oracle", 0, 0), ("fit_a", 1, 0),
                                ("fit_b", 0, 1), ("fitted", 1, 1)]:
                a, b = curves[0][i], curves[1][j]
                record[label+"_raw"] = a[-1]/b[-1]
                record[label+"_anchored"] = study.anchored_kcor(a, b, skip)[-1]
            results.append(record)
    return results


def anchor_check():
    """Exact multinomial counts at the three boundaries needed with known theta."""
    size = study.N_INITIAL
    skip = study.K.DYNAMIC_HVE_SKIP_WEEKS
    meta = pd.read_csv(STUDY / "out/quiet_contamination_meta.csv", index_col=0)["value"]
    rng = np.random.default_rng(20261122)
    growth, reference, anchor_counts = [], [], []
    for theta in (study.THETA0_A, study.THETA0_B):
        stream = RecordProbabilities()
        study.simulate_one_cohort(stream, size, study.N_WEEKS, float(meta.k_simulator),
                                  float(meta.gamma_per_week), theta, skip, None)
        survival = np.cumprod(1.-np.array(stream.probabilities))
        boundaries = survival[[skip-1, skip+4, -1]]
        prob = -np.diff(np.r_[1., boundaries, 0.])
        counts = rng.multinomial(size, prob, size=100_000)
        survivors = size-np.cumsum(counts[:, :3], axis=1)
        # Weekly integrated hazards telescope to -log(survival relative to entry).
        observed = -np.log(survivors[:, 1:]/survivors[:, :1])
        adjusted = study.K.invert_gamma_frailty(observed, theta)
        growth.append(adjusted)
        reference.append(study.K.invert_gamma_frailty(-np.log(boundaries[1:]/boundaries[0]), theta))
        anchor_counts.append(size*prob[1])
    a, b = growth
    raw = a[:, 1]/b[:, 1]
    sampled = raw/(a[:, 0]/b[:, 0])
    exact = raw/(reference[0][0]/reference[1][0])
    print("Expected deaths through the five-week anchor (A, B):", anchor_counts)
    for label, v in [("Sampled anchor",sampled), ("Population anchor",exact),
                     ("Paired anchor difference",sampled-exact)]:
        margin = t.ppf(.975,len(v)-1)*np.std(v,ddof=1)/np.sqrt(len(v))
        print(label, v.mean(), v.mean()-margin, v.mean()+margin)


if __name__ == "__main__":
    anchor_check()
    jobs = [(size, start, start+25) for size in (50_000, 500_000, 5_000_000)
            for start in range(0, N, 25)]
    jobs += [(size, -1, 0) for size in (50_000, 500_000, 5_000_000)]
    with ProcessPoolExecutor(max_workers=5) as pool:
        rows = pd.DataFrame(row for group in pool.map(audit, jobs) for row in group)
    saved = pd.read_csv(HERE / "fitter_comparison_runs.csv")
    saved = saved[(saved.epsilon_comorb == 0) & saved.fitter.isin(["revised_joint","original"])].copy()
    saved["fitter"] = saved.fitter.replace({"revised_joint": "joint"})
    keys = ["rep","fitter"]
    a = rows[(rows["size"] == 50_000) & (rows.rep >= 0)].set_index(keys).sort_index()
    b = saved.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_allclose(a.fitted_anchored, b.KCOR_asymptote, rtol=0, atol=1e-8)
    rows.to_csv(HERE / "null_diagnostic_runs.csv", index=False)

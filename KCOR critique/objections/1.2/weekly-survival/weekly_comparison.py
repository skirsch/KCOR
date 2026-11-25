"""Compare implied survivor frailty and cumulative exposures during the wave."""
import io
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "1.3"))
import table8_bootstrap as table
from trial import gamma_quadrature, make_correction

GROUPS = ("two-dose", "unvaccinated", "all-persons")
METRICS = ("starting_theta", "old_gamma_proxy_theta", "old_divisor",
           "trial_survivor_mean", "trial_survivor_variance", "trial_relative_variance",
           "trial_survivor_moment_ratio", "old_wave_exposure", "trial_wave_exposure",
           "old_corrected_hazard", "trial_corrected_hazard")


def survivor_moments(theta, alpha, baseline, wave):
    if theta < table.K.KCOR6_THETA_EPS:
        return np.ones_like(baseline), np.zeros_like(baseline), np.ones_like(baseline)
    z, weights = gamma_quadrature(theta)
    scaled = z[None, :] / (1 + theta * baseline[:, None])
    log_weights = np.log(weights)[None, :] - wave[:, None] * scaled**alpha
    prob = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
    mean = (prob * scaled).sum(axis=1)
    variance = (prob * (scaled - mean[:, None])**2).sum(axis=1)
    moment = (prob * scaled**alpha).sum(axis=1)
    return mean, variance, moment / mean


def initialize(populations):
    global original, correction
    table.initialize(populations, "entry-moment")
    original = table.K.apply_nph_correction_post_inversion
    correction = make_correction(table.K)


def estimate(task):
    enrollment, draw = task
    records = []

    def compare(H0, time, dates, theta, alpha, start, end, k, gamma, **kwargs):
        before, _ = original(H0, time, dates, theta, alpha, start, end, k, gamma, **kwargs)
        after, diag = correction(H0, time, dates, theta, alpha, start, end, k, gamma, **kwargs)
        rebased = time - table.K.DYNAMIC_HVE_SKIP_WEEKS
        indices = np.flatnonzero((dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
                                 & (rebased >= 0))
        first = indices[0]
        opening = H0[first-1] if first else 0.0
        proxy = table.K.propagate_theta(theta, H0[indices])
        divisor = table.K.gamma_moment_alpha(proxy, alpha)
        mean, moment = table.K._wave_entry_moments(theta, opening, alpha)
        divisor[0] = moment / mean
        increments = k * table.K.safe_exp(gamma * rebased[indices]) * table.K._gompertz_week_factor(gamma)
        reference = opening + np.cumsum(increments)
        excess = H0[indices] - reference
        old_wave = np.where(excess > table.K.EPS, excess / divisor, 0.0)
        if theta < table.K.KCOR6_THETA_EPS:
            baseline = np.where(excess > table.K.EPS, reference, H0[indices])
            wave = old_wave
        else:
            assert diag["correction_mode"] == "weekly_survival_trial"
            baseline = np.array(diag["baseline_exposure_path"])
            wave = np.array(diag["wave_exposure_path"])
        mean, variance, moment_ratio = survivor_moments(theta, alpha, baseline, wave)
        values = np.column_stack((np.full(len(indices), theta), proxy, divisor, mean, variance,
                                  variance/mean**2, moment_ratio, old_wave, wave,
                                  before[indices], after[indices]))
        records.append((dates[indices], values))
        return after, diag

    table.K.apply_nph_correction_post_inversion = compare
    points, _, _ = table.estimate(enrollment, draw)
    assert len(records) == 9
    return enrollment, draw, points, records


def main():
    workbook = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB], cwd=HERE)
    populations = [table.read_population(io.BytesIO(workbook), sheet) for sheet in table.SHEETS]
    with np.load(HERE / "table8-draws.npz") as archive:
        expected = archive["points"][0]
    dates_by_enrollment = {}
    values_by_enrollment = {}
    tasks = [(i, draw) for draw in range(1000) for i in range(2)]
    with ProcessPoolExecutor(6, initializer=initialize, initargs=(populations,)) as pool:
        for n, (i, draw, points, records) in enumerate(pool.map(estimate, tasks, chunksize=4), 1):
            np.testing.assert_allclose(points, expected[i, draw], rtol=1e-10, atol=1e-12)
            dates = records[0][0]
            for dates_group, _ in records:
                np.testing.assert_array_equal(dates_group, dates)
            if i not in values_by_enrollment:
                dates_by_enrollment[i] = dates
                values_by_enrollment[i] = np.empty((1000, 9, len(dates), len(METRICS)))
            np.testing.assert_array_equal(dates, dates_by_enrollment[i])
            values_by_enrollment[i][draw] = [v for _, v in records]
            if n % 400 == 0:
                print(f"{n}/2000 verified enrollment runs", flush=True)
    results = []
    for i, sheet in enumerate(table.SHEETS):
        values = values_by_enrollment[i]
        assert np.isfinite(values).all()
        limits = np.quantile(values[1:], [.025, .975], axis=0)
        for group, (subgroup, dose) in enumerate((s, d) for s in GROUPS for d in range(3)):
            for week, date in enumerate(dates_by_enrollment[i]):
                row = dict(enrollment=sheet, subgroup=subgroup, pseudo_dose=dose,
                           wave_week=week+1, date=str(date)[:10])
                for col, name in enumerate(METRICS):
                    row[name] = values[0, group, week, col]
                    row[name+"_lower95"], row[name+"_upper95"] = limits[:, group, week, col]
                results.append(row)
    pd.DataFrame(results).to_csv(HERE / "weekly-comparison.csv", index=False)
    print(f"{len(results)} cohort-weeks with pointwise intervals")


if __name__ == "__main__":
    main()

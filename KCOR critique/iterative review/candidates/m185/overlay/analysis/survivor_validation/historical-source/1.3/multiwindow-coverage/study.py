"""Paired coverage study of the current two-stratum pooled Monte Carlo estimator."""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.special import gamma as gamma_function
from scipy.stats import binomtest, t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paired-seed-fit"))
import compare

START = pd.Timestamp("2021-06-14")
DATES = pd.date_range(START, periods=53, freq="7D")
TIME = np.arange(len(DATES), dtype=float)
WINDOWS = ((2, 26), (35, 52))
WAVE = (27, 34)
STRATA = (1940, 1950)
RATES = np.array([.002, .001])
THETA = np.array([[4., .5], [4., .5]])
POPULATION = 1_000_000
SEED = 20261124
SCENARIOS = ("no_wave", "nonlinear_wave")
REPETITIONS = 100
DRAWS = 199


def initialize():
    global model, config, guards, skip, anchor, quiet
    compare.initialize("entry-moment")
    model = compare.adapter.K
    model.MONTE_CARLO_MODE = True
    model.MC_INCLUDE_AGGREGATES = True
    model.MC_YOB_START, model.MC_YOB_END = min(STRATA), max(STRATA)
    model.MC_ENROLLMENT_DATE = "2021_24"
    model._load_dataset_dose_pairs_config()
    config = model.get_gompertz_theta_fit_config()
    guards = model.get_theta_degenerate_fit_config()
    skip = model.DYNAMIC_HVE_SKIP_WEEKS
    anchor = model.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    quiet = np.logical_or.reduce([(TIME >= a) & (TIME <= b) for a, b in WINDOWS])
    assert skip == 2 and anchor == 5


@lru_cache(maxsize=None)
def population(scenario):
    """True survival under the stipulated gamma law at modeled entry."""
    slope = config["gamma_per_week"]
    rebased = TIME - skip
    baseline = RATES[:, None] * np.expm1(slope * np.maximum(rebased + 1, 0)) / slope
    alpha = 1.18 if scenario == "nonlinear_wave" else 1.
    wave = np.zeros_like(baseline)
    if alpha != 1:
        wave = 3 * RATES[:, None] * np.cumsum((TIME >= WAVE[0]) & (TIME <= WAVE[1]))
    probabilities = np.empty((2, 2, len(DATES)+1))
    for age, dose in np.ndindex(THETA.shape):
        theta = THETA[age, dose]
        b, w = baseline[age], wave[age]
        if alpha == 1:
            survival = np.exp(-np.log1p(theta * b) / theta)
        else:
            # Independent quadrature using the gamma-variable substitution from
            # objection 3.2's known-truth tests.
            def integrand(u):
                x = u**theta
                z = theta*x
                return np.exp(-x - z*b - z**alpha*w)
            integral, error = quad_vec(integrand, 0., np.inf, epsabs=1e-12, epsrel=1e-12)
            assert error < 1e-9
            survival = integral * theta / gamma_function(1/theta)
        # Early deaths are independent of frailty. The post-skip population
        # therefore has the stated mean-one frailty distribution.
        survival[:skip] = np.exp(-RATES[age] * (TIME[:skip]+1))
        survival[skip:] *= survival[skip-1]
        probabilities[age, dose] = np.r_[-np.diff(np.r_[1., survival]), survival[-1]]
    assert probabilities.min() >= 0
    np.testing.assert_allclose(probabilities.sum(axis=2), 1., atol=1e-12)
    # Equal baseline and wave exposures within each age give an exact null
    # anchored ratio in every age, independently of the pooling weights.
    return probabilities, 1.


def samples(scenario, rep, draws):
    probs, truth = population(scenario)
    rng = np.random.default_rng([SEED, SCENARIOS.index(scenario), rep])
    counts = rng.multinomial(POPULATION, (probs/4).ravel()).reshape(probs.shape)
    bootstrap = rng.multinomial(POPULATION, counts.ravel()/POPULATION, size=draws)
    return np.concatenate((counts[None], bootstrap.reshape(draws, *counts.shape))), truth


def arrays(counts):
    return counts[..., :-1], np.cumsum(counts[..., ::-1], axis=-1)[..., ::-1][..., :-1]


def pooling_weights(batch):
    """The workbook uses a common weight computed across the entire MC batch."""
    dead, alive = arrays(batch)
    early_mr = dead[..., :4] / (alive[..., :4] + model.EPS)
    mean_mr = np.nanmean(np.where(early_mr > model.EPS, early_mr, np.nan), axis=(0, 2, 3))
    weights = mean_mr * alive[..., :4].sum(axis=(0, 2, 3))
    assert np.isfinite(weights).all() and (weights > 0).all()
    return weights / weights.sum()


def estimate(counts, weights, scenario):
    dead, alive = arrays(counts)
    hazard = model.hazard_from_mr(np.clip(dead/(alive+model.EPS), 0, .999))
    hazard[..., :skip] = 0.
    observed = np.cumsum(hazard, axis=-1)
    outcomes = []
    active = scenario == "nonlinear_wave"
    keep = (TIME <= WAVE[1]) | (TIME == TIME[-1]) if active else np.ones(len(TIME), bool)
    for fit in (compare.production_fit, compare.joint_fit):
        final = np.empty((2, 2))
        flags, failures, gap_clamped = 0, 0, 0
        frailties = []
        for age, dose in np.ndindex(THETA.shape):
            (k, theta), diag = fit(hazard[age, dose], TIME-skip, quiet,
                                   config["k_anchor_weeks"], config["gamma_per_week"],
                                   deaths_arr=dead[age, dose])
            failures += int(not diag["success"])
            flags += int(bool(diag.get("insufficient_signal")))
            gap_clamped += int(bool(diag.get("delta_negative_clamped")))
            theta = theta if diag["success"] and np.isfinite(theta) and theta >= 0 else 0.
            theta, _, _ = model._apply_theta_degenerate_guard(theta, guards)
            frailties.append(theta)
            H = observed[age, dose, keep]
            curve = model.invert_gamma_frailty(H, theta)
            if active:
                curve, _ = model.apply_nph_correction_post_inversion(
                    curve, TIME[keep], DATES.to_numpy()[keep], theta, 1.18,
                    DATES[WAVE[0]], DATES[WAVE[1]], k, config["gamma_per_week"],
                    observed_cumhaz=H)
            final[age, dose] = curve[-1] / curve[anchor]
        ratios = final[:, 1] / final[:, 0]
        point = float(model.safe_exp(np.average(model.safe_log(ratios), weights=weights)))
        outcomes.append([point, *ratios, flags, failures, gap_clamped, *frailties])
    return outcomes


def run_dataset(task):
    scenario, rep, draws = task
    batch, truth = samples(scenario, rep, draws)
    weights = pooling_weights(batch)
    estimates = np.asarray([estimate(c, weights, scenario) for c in batch])
    return scenario, rep, truth, weights, estimates


def mean_interval(values):
    values = np.asarray(values)
    mean = values.mean()
    margin = t.ppf(.975, len(values)-1) * values.std(ddof=1)/np.sqrt(len(values))
    return [float(mean), float(mean-margin), float(mean+margin)]


def summarize(values, truth):
    assert np.isfinite(values).all()
    samples = values[:, 1:, :, 0]
    low, high = np.quantile(samples, [.025, .975], axis=1)
    covered = (low <= truth) & (high >= truth)
    records = {}
    for j, label in enumerate(("fixed", "joint")):
        n = len(values)
        successes = int(covered[:, j].sum())
        ci = binomtest(successes, n).proportion_ci()
        prefix_low, prefix_high = np.quantile(samples[:, :99, j], [.025, .975], axis=1)
        records[label] = dict(
            covered=successes, datasets=n, coverage=successes/n, coverage_lower=ci.low,
            coverage_upper=ci.high,
            mean_width=mean_interval(high[:, j]-low[:, j]),
            mean_point=mean_interval(values[:, 0, j, 0]),
            rmse=float(np.sqrt(np.mean((values[:, 0, j, 0]-truth)**2))),
            prefix_coverage=float(np.mean((prefix_low <= truth) & (prefix_high >= truth))),
            originals_with_insufficient_signal=int((values[:, 0, j, 3] > 0).sum()),
            draws_with_insufficient_signal=int((values[:, 1:, j, 3] > 0).sum()),
            failed_fit_calls=int(values[:, :, j, 4].sum()),
            originals_with_clamped_gap=int((values[:, 0, j, 5] > 0).sum()))
    gain = int((~covered[:, 0] & covered[:, 1]).sum())
    loss = int((covered[:, 0] & ~covered[:, 1]).sum())
    return dict(truth=truth, **records, gained_coverage=gain, lost_coverage=loss,
                paired_p=binomtest(gain, gain+loss).pvalue if gain+loss else 1.,
                width_difference=mean_interval((high[:, 1]-low[:, 1])-(high[:, 0]-low[:, 0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    tasks = [(scenario, i, args.draws) for i in range(args.repetitions) for scenario in SCENARIOS]
    archive = {scenario: [] for scenario in SCENARIOS}
    weights = {scenario: [] for scenario in SCENARIOS}
    with ProcessPoolExecutor(args.workers, initializer=initialize) as pool:
        for n, (scenario, rep, truth, w, estimates) in enumerate(pool.map(run_dataset, tasks, chunksize=1), 1):
            # Preserve completed batches so an interrupted run is inspectable.
            assert rep == len(archive[scenario])
            archive[scenario].append(estimates)
            weights[scenario].append(w)
            if n % 10 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)} simulated datasets completed", flush=True)
                np.savez_compressed(HERE / "coverage-draws.npz", **{s: np.array(v) for s, v in archive.items()},
                                    **{s+"_weights": np.array(v) for s, v in weights.items()})
    results = {s: summarize(np.asarray(v), 1.) for s, v in archive.items()}
    (HERE / "results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

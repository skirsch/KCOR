"""Verify recovery against an independent integration of the generating law."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from check_post_wave import survival_loss
from trial import make_correction
import KCOR


def compare_case(theta, alpha, baseline, wave, gamma=.001):
    count = len(baseline)
    weeks = np.arange(count, dtype=float)
    dates = np.datetime64("2022-01-03") + np.arange(count) * np.timedelta64(7, "D")
    start, end = 2, 5
    if theta == 0:
        observed = baseline + wave
    else:
        observed = -np.log1p(-np.array([survival_loss(b, w, theta, alpha)
                                       for b, w in zip(baseline, wave)]))
    ordinary = KCOR.invert_gamma_frailty(observed, theta)
    k = (baseline[start] - baseline[start-1]) / (
        np.exp(gamma * start) * KCOR._gompertz_week_factor(gamma))
    args = (ordinary, weeks, dates, theta, alpha, dates[start], dates[end], k, gamma)
    current, _ = KCOR.apply_nph_correction_post_inversion(*args, t_rebased=weeks, observed_cumhaz=observed)
    revised, diag = make_correction(KCOR)(*args, t_rebased=weeks, observed_cumhaz=observed)
    np.testing.assert_allclose(revised, baseline + wave, rtol=3e-8, atol=2e-9)
    if theta and alpha != 1:
        np.testing.assert_allclose(diag["wave_exposure_path"], wave[start:end+1], rtol=3e-8, atol=2e-9)
        np.testing.assert_allclose(diag["baseline_exposure_path"], baseline[start:end+1], atol=2e-9)
        assert diag["max_survival_residual"] < 1e-12
    else:
        np.testing.assert_array_equal(revised, ordinary)
    return current, revised, diag


def main():
    gamma = .001
    wave = np.array([0, 0, .05, .08, .08, .12, .12, .12])
    baseline = .2 + .02 * np.expm1(gamma * (np.arange(8) + 1)) / gamma
    print("theta,alpha,old_max_error,trial_max_error")
    for theta in (0, .01, .5, 1, 4, 10, 100):
        for alpha in (.7, 1, 1.18, 2):
            previous, trial, _ = compare_case(theta, alpha, baseline, wave)
            true = baseline + wave
            print(f"{theta},{alpha},{np.max(abs(previous-true)):.10g},{np.max(abs(trial-true)):.10g}")

    # During one week, deaths are below the forecast from the fitted baseline.
    # Previous wave exposure stays fixed and the baseline increases by less.
    lower_baseline = baseline.copy()
    lower_baseline[4:] -= .01
    _, _, diag = compare_case(1., 1.18, lower_baseline, wave)
    assert diag["deficit_weeks"] == 1
    # The same logic applies to a week with no deaths.
    no_deaths = baseline.copy()
    no_deaths[4:] -= baseline[4] - baseline[3]
    _, _, diag = compare_case(1., 1.18, no_deaths, wave)
    assert diag["deficit_weeks"] == 1
    # The modeled follow-up may begin in the middle of a calendar wave.
    dates = np.datetime64("2022-01-03") + np.arange(5) * np.timedelta64(7, "D")
    time = np.arange(5, dtype=float)
    baseline = np.array([0., 0., .02, .04, .06])
    wave = np.array([0., 0., .05, .08, .08])
    observed = -np.log1p(-np.array([survival_loss(b, w, 1., 1.18)
                                   for b, w in zip(baseline, wave)]))
    ordinary = KCOR.invert_gamma_frailty(observed, 1.)
    corrected, diag = make_correction(KCOR)(
        ordinary, time, dates, 1., 1.18, dates[0], dates[3], .02, 0.,
        t_rebased=time - 2, observed_cumhaz=observed)
    np.testing.assert_allclose(corrected, baseline + wave, atol=2e-9)
    assert diag["wave_points"] == 2 and diag["deficit_weeks"] == 0
    assert np.all(np.diff(corrected) >= 0)
    print("Known-exposure, alpha=1, zero-frailty and deficit checks passed.")


if __name__ == "__main__":
    main()

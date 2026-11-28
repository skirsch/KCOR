"""Experimental weekly inversion for the optional NPH module.

Use the existing KCOR reference rate, wave dates and post-wave continuation.
If a week cannot support a positive wave increment, retain prior wave exposure
and recover that week's baseline increment from conditional survival.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaincinv, logsumexp, roots_legendre


@lru_cache(maxsize=256)
def gamma_quadrature(theta):
    # Same probability-space quadrature as the existing post-wave solver.
    nodes, weights = roots_legendre(32)
    edges = np.array([0.0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, .001, .01, .1, .5])
    half_width = np.diff(edges)[:, None] / 2
    probs = (edges[:-1, None] + half_width * (nodes + 1)).ravel()
    weights = (half_width * weights).ravel()
    z = theta * gammaincinv(1 / theta, np.r_[probs, 1 - probs])
    return z, np.tile(weights, 2)


def cumulative_hazard(baseline, wave, theta, alpha):
    """Negative log survival under the starting gamma law."""
    gamma_hazard = np.log1p(theta * baseline) / theta
    if wave == 0:
        return gamma_hazard
    z, weights = gamma_quadrature(theta)
    powers = wave * np.power(z / (1 + theta * baseline), alpha)
    lost = np.dot(weights, -np.expm1(-powers))
    log_wave = np.log1p(-lost) if lost < .5 else logsumexp(-powers, b=weights)
    return gamma_hazard - log_wave


def solve_increase(hazard, target, initial_upper):
    lower = hazard(0.0)
    if target <= lower:
        return 0.0
    upper = max(initial_upper, 1e-10)
    while hazard(upper) < target:
        upper *= 2
        if not np.isfinite(upper):
            raise ArithmeticError("Could not bracket the weekly exposure")
    return brentq(lambda x: hazard(x) - target, 0, upper, xtol=1e-14)


def make_correction(model):
    original = model.apply_nph_correction_post_inversion

    def correct(H0, t_vals, dates, theta_applied, alpha, start_dt, end_dt,
                k_hat, gamma_per_week, t_rebased=None, observed_cumhaz=None):
        theta = float(theta_applied)
        H0 = np.asarray(H0, dtype=float)
        t_r = np.asarray(t_vals, dtype=float) - model.DYNAMIC_HVE_SKIP_WEEKS if t_rebased is None else np.asarray(t_rebased)
        dates = np.asarray(dates, dtype="datetime64[ns]")
        mask = (dates >= np.datetime64(start_dt)) & (dates <= np.datetime64(end_dt)) & (t_r >= 0)
        indices = np.flatnonzero(mask)
        if (abs(alpha - 1) <= model.NPH_IDENTITY_TOL or theta < model.KCOR6_THETA_EPS
                or not np.isfinite(theta) or not len(indices)
                or not (np.isfinite(k_hat) and k_hat > 0 and np.isfinite(gamma_per_week))):
            return original(H0, t_vals, dates, theta, alpha, start_dt, end_dt, k_hat,
                            gamma_per_week, t_rebased=t_r, observed_cumhaz=observed_cumhaz)

        observed = (np.log1p(theta * H0) / theta if observed_cumhaz is None
                    else np.asarray(observed_cumhaz, dtype=float))
        first, last = indices[0], indices[-1]
        baseline = float(H0[first - 1]) if first else 0.0
        wave = 0.0
        entry_loss = cumulative_hazard(baseline, wave, theta, alpha)
        entry_observed = observed[first - 1] if first else 0.0
        baseline_path = k_hat * model.safe_exp(gamma_per_week * t_r) * model._gompertz_week_factor(gamma_per_week)
        output = H0.copy()
        baseline_values, wave_values = [], []
        deficits = 0
        max_residual = 0.0
        for i in indices:
            target = entry_loss + observed[i] - entry_observed
            planned = baseline + baseline_path[i]
            planned_loss = cumulative_hazard(planned, wave, theta, alpha)
            if target > planned_loss:
                baseline = planned
                wave += solve_increase(
                    lambda x: cumulative_hazard(baseline, wave + x, theta, alpha),
                    target, target - planned_loss)
            else:
                deficits += int(target < planned_loss - 1e-12)
                baseline += solve_increase(
                    lambda x: cumulative_hazard(baseline + x, wave, theta, alpha),
                    target, baseline_path[i])
            output[i] = baseline + wave
            baseline_values.append(baseline)
            wave_values.append(wave)
            max_residual = max(max_residual, abs(cumulative_hazard(baseline, wave, theta, alpha) - target))

        if last + 1 < len(output):
            tail = slice(last + 1, None)
            increases = model.post_wave_baseline_increases(
                observed[tail] - observed[last], theta, alpha, baseline, wave)
            output[tail] = output[last] + increases

        new_wave = np.diff(wave_values, prepend=0)
        diag = dict(correction_mode="weekly_survival_trial", wave_points=len(indices),
                    corrected_points=int(np.count_nonzero(new_wave > 0)),
                    skipped_nonpositive_excess=int(np.count_nonzero(new_wave == 0)),
                    deficit_weeks=deficits, max_survival_residual=max_residual,
                    wave_exposure_path=wave_values, baseline_exposure_path=baseline_values,
                    post_wave_exposure=wave, post_wave_baseline=baseline,
                    max_abs_change=float(np.max(np.abs(output - H0))),
                    post_wave_level_shift=float(output[last] - H0[last]),
                    any_nonmonotone_fixed=False, f_min=float("nan"), f_max=float("nan"))
        return output, diag

    return correct

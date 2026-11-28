"""Check first-wave entry against survival integrated over enrollment frailty."""
from math import exp, inf
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "1.3/paired-seed-fit"))
import compare


def survival(baseline, wave, alpha):
    # Unit exponential enrollment frailty. Integrate without using KCOR's inversion.
    return quad(lambda z: exp(-(1 + baseline)*z - wave*z**alpha), 0, inf,
                epsabs=1e-14, epsrel=2e-13)[0]


def run_week(model, baseline, k, wave, alpha):
    observed = -np.log([survival(baseline, 0, alpha), survival(baseline+k, wave, alpha)])
    dates = np.array(["2022-01-03", "2022-01-10"], dtype="datetime64[D]")
    time = np.arange(2, dtype=float)
    inverted = np.expm1(observed)
    values, diag = model.apply_nph_correction_post_inversion(
        inverted, time, dates, 1., alpha, dates[1], dates[1], k, 0.,
        t_rebased=time, observed_cumhaz=observed)
    return inverted, values, diag


def main():
    compare.initialize("wave-week")
    previous = compare.adapter.K
    compare.initialize("entry-moment")
    model = compare.adapter.K
    for theta in (0., 1e-9, .5, 1., 4.):
        for baseline in (0., 1., 2.):
            mean, moment = model._wave_entry_moments(theta, baseline, 2.)
            expected_mean = 1/(1+theta*baseline)
            np.testing.assert_allclose([mean, moment], [expected_mean, (1+theta)*expected_mean**2],
                                       rtol=1e-11)
    for alpha in (.7, 1.18, 2.):
        density = lambda z: 2*exp(-2*z)
        exact = quad(lambda z: density(z)*z**alpha, 0, inf)[0]
        _, _, diag = run_week(model, 1., .02, .05, alpha)
        np.testing.assert_allclose([diag["entry_survivor_mean"], diag["entry_survivor_moment"],
                                    diag["f_min"]], [.5, exact, exact/.5], rtol=1e-10)
    _, _, no_excess = run_week(model, 1., .02, 0., 2.)
    assert no_excess["corrected_points"] == 0
    ordinary, corrected, diag = run_week(model, 1., .02, .05, 1.)
    np.testing.assert_array_equal(ordinary, corrected)
    assert diag["correction_mode"] == "identity"
    # Enrollment during a calendar wave. The first modeled week is after the
    # skip; with no modeled depletion yet it uses the enrollment distribution.
    dates = np.datetime64("2022-01-03") + np.arange(4)*np.timedelta64(7, "D")
    time = np.arange(4, dtype=float)
    for first, rebased in ((0, time), (2, time-2)):
        cumulative = np.zeros(4)
        cumulative[first:] = [.08 + .12*i for i in range(4-first)]
        args = (cumulative, time, dates, 1., 2., dates[0], dates[-1], .02, 0.)
        now, diag = model.apply_nph_correction_post_inversion(*args, t_rebased=rebased)
        old, old_diag = previous.apply_nph_correction_post_inversion(*args, t_rebased=rebased)
        np.testing.assert_allclose([diag["entry_survivor_mean"], diag["entry_survivor_moment"]], [1., 2.])
        np.testing.assert_allclose(now[first], .02+.06/2, atol=1e-12)
        np.testing.assert_array_equal(now[first+1:], old[first+1:])
        assert diag["post_wave_exposure"] == old_diag["post_wave_exposure"]
    # The opening derivative applies in the limit of a short interval. At a
    # finite duration the wave itself selects survivors during that interval.
    print("Duration, true exposure, previous estimate, revised estimate")
    errors = []
    for duration in (1., .1, .01, .001):
        wave = .05*duration
        values = [run_week(m, 1., .02*duration, wave, 2.)[2]["post_wave_exposure"]
                  for m in (previous, model)]
        print(f"{duration:g}, {wave:.9f}, {values[0]:.9f}, {values[1]:.9f}")
        errors.append(abs(values[1]/wave - 1))
    assert np.all(np.diff(errors) < 0) and errors[-1] < 1e-4
    np.testing.assert_allclose(values[0]/wave, 2/3, atol=1e-4)
    print("Passed moment, hazard-scale, proportional identity and finite-week checks.")


if __name__ == "__main__":
    main()

"""Check post-wave increments against independent integration of the starting gamma law."""
from pathlib import Path
import math
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "iterative review" / "revised" / "code"))
from KCOR import post_wave_baseline_increases


def survival_loss(baseline, wave, theta, alpha):
    shape = 1 / theta

    def integrand(log_z):
        z = math.exp(log_z)
        log_density = shape * (log_z - math.log(theta)) - z / theta - gammaln(shape)
        return math.exp(log_density) * -math.expm1(-baseline * z - wave * z**alpha)

    return sum(quad(integrand, a, b, epsabs=1e-13)[0]
               for a, b in [(-120, 0), (0, 15)])


def test_post_wave_inversion():
    expected = np.array([0, 0.001, 0.001, 0.1, 0.3])
    baseline = 0.20
    for theta in (0.01, 1, 10, 100):
        for alpha in (0.5, 1, 1.18, 2):
            for wave in (0, 0.05):
                log_survival = np.array([
                    math.log1p(-survival_loss(baseline + u, wave, theta, alpha))
                    for u in expected
                ])
                observed = log_survival[0] - log_survival
                actual = post_wave_baseline_increases(observed, theta, alpha, baseline, wave)
                np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)
    np.testing.assert_array_equal(
        post_wave_baseline_increases(expected, 0, 1.18, baseline, 0.05), expected
    )


if __name__ == "__main__":
    test_post_wave_inversion()
    print("Post-wave conditional survival checks passed.")

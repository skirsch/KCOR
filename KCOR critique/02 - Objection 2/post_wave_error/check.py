"""Verify the post-wave gamma-inversion formulas against direct integration."""

import math
from itertools import product

from scipy.integrate import quad
from scipy.special import erfcx, pbdv


def integrate_survival_and_mean(theta, baseline, wave, alpha):
    shape = 1 / theta
    rate = shape + baseline

    def integral(power):
        # Integrate in u=(rate*z)**shape to remove a possible singularity at z=0.
        def integrand(u):
            scaled_frailty = u**theta
            z = scaled_frailty / rate
            return z**power * math.exp(-scaled_frailty - wave * z**alpha)

        return quad(integrand, 0, math.inf, epsabs=1e-12, epsrel=1e-12)[0]

    mass = integral(0)
    prior_constant = math.exp(shape * math.log(shape / rate) - math.lgamma(shape + 1))
    return prior_constant * mass, integral(1) / mass


def quadratic_survival_and_mean(theta, baseline, wave):
    shape = 1 / theta
    scale = math.sqrt(2 * wave)
    x = (shape + baseline) / scale
    d_shape = pbdv(-shape, x)[0]
    d_next = pbdv(-shape - 1, x)[0]
    survival = math.exp(shape * math.log(shape / scale) + x*x/4 + math.log(d_shape))
    return survival, (shape / scale) * (d_next / d_shape)


def relative_hazard_error(theta, survival, survivor_mean):
    return math.expm1(math.log(survivor_mean) - theta * math.log(survival))


def verify_formulas():
    for theta, baseline, wave in product((.25, .5, 1., 2., 4.), (0., .2, .6), (.01, .05, .2)):
        exact = quadratic_survival_and_mean(theta, baseline, wave)
        integrated = integrate_survival_and_mean(theta, baseline, wave, 2)
        for actual, expected in zip(exact, integrated, strict=True):
            assert math.isclose(actual, expected, rel_tol=2e-8), (theta, baseline, wave)
        if theta == 1:
            x = (1 + baseline) / (2 * math.sqrt(wave))
            survival = math.sqrt(math.pi) * erfcx(x) / (2 * math.sqrt(wave))
            mean = (1/survival - 1 - baseline) / (2 * wave)
            assert math.isclose(survival, integrated[0], rel_tol=1e-10)
            assert math.isclose(mean, integrated[1], rel_tol=1e-10)

    for theta, alpha in product((.25, 1., 4.), (0., .5, 1., 1.18, 2.)):
        baseline = .2
        survival, mean = integrate_survival_and_mean(theta, baseline, .05, alpha)
        error = relative_hazard_error(theta, survival, mean)
        if alpha == 1:
            assert abs(error) < 1e-10
        else:
            assert error * (1 - alpha) > 0
        weak_wave = 1e-6
        survival, mean = integrate_survival_and_mean(theta, baseline, weak_wave, alpha)
        shape = 1 / theta
        moment = math.exp(math.lgamma(shape + alpha) - math.lgamma(shape)) / (shape + baseline)**alpha
        approximate = (1 - alpha) * theta * weak_wave * moment
        assert math.isclose(relative_hazard_error(theta, survival, mean), approximate,
                            rel_tol=1e-3, abs_tol=1e-10), (theta, alpha)


def main():
    verify_formulas()
    print("Closed form, error direction, and weak-wave approximation verified.")
    theta, baseline, wave, increment = 1., .20, .05, .001
    for alpha in (1., 1.18, 2.):
        start, mean = integrate_survival_and_mean(theta, baseline, wave, alpha)
        end, _ = integrate_survival_and_mean(theta, baseline + increment, wave, alpha)
        inferred = (end**-theta - start**-theta) / theta
        instantaneous_error = relative_hazard_error(theta, start, mean)
        print(f"alpha={alpha:.2f}: true quiet-week increment {increment:.9f}, "
              f"inferred {inferred:.9f}, error {inferred/increment-1:.4%} "
              f"(instantaneous {instantaneous_error:.4%})")


if __name__ == "__main__":
    main()

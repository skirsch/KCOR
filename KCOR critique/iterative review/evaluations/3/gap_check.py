"""Run with py -3.11 -B gap_check.py <frozen m83 root>. Writes no results."""
import os
from pathlib import Path
import sys

import numpy as np
import scipy


def history(model, weeks, k, theta, regroup=False):
    gamma = 0.085 / 52
    if regroup:
        start = k * (np.expm1(gamma * weeks) / gamma)
        increment = k * (np.exp(gamma * weeks) * (np.expm1(gamma) / gamma))
    else:
        start = k * np.expm1(gamma * weeks) / gamma
        increment = k * np.exp(gamma * weeks) * np.expm1(gamma) / gamma
    return model.weekly_gamma_hazard(start, increment, theta)


def fit(model, label, weeks, quiet, hazard):
    (k, theta), diag = model.fit_theta0_gompertz(
        hazard, weeks, quiet, 4, 0.085 / 52, joint_seed=True
    )
    flags = ["insufficient_signal", "insufficient_deaths", "bad_fit",
             "theta_hit_bound", "identifiability_flag", "delta_negative",
             "delta_negative_clamped"]
    print(label, f"k={k:.17g}", f"seed={diag['theta0_init']:.17g}",
          f"applied={theta:.17g}", diag["status"], f"success={diag['success']}",
          "reported raw deltas", diag["delta_raw_values"],
          "true flags", [key for key in flags if diag[key]])
    return theta, diag


def main(root):
    os.environ["DATASET"] = "Czech"
    os.environ.pop("SA_ANCHOR_WEEKS", None)
    os.chdir(root)
    sys.path.insert(0, str(root / "code"))
    import KCOR as model

    print("Python", sys.version.split()[0], "NumPy", np.__version__, "SciPy", scipy.__version__)
    # Half-open windows. Each synthetic history is shock-free at every week.
    for label, n, k, first_end, reentry, second_end in [
        ("blind 2", 90, .002, 30, 40, 90),
        ("blind 3", 60, .001, 20, 40, 60),
        ("original handoff 1", 100, .001, 30, 60, 90),
        ("original handoff 2", 100, .001, 40, 60, 100),
        ("original handoff 3", 100, .001, 24, 50, 80),
    ]:
        weeks = np.arange(n, dtype=float)
        first = weeks < first_end
        split = first | ((weeks >= reentry) & (weeks < second_end))
        observed = history(model, weeks, k, 4.)
        single_theta, _ = fit(model, label + " first", weeks, first, observed)
        np.testing.assert_allclose(single_theta, 4., atol=1e-6, rtol=0)
        theta, diag = fit(model, label + " split", weeks, split, observed)
        np.testing.assert_allclose(diag["theta0_init"], 4., atol=1e-6, rtol=0)
        if diag["insufficient_signal"]:
            assert theta == 0 and diag["delta_raw_values"][0] <= 0
            delta = diag["delta_raw_values"][0]
            print("  first-pass rejection for delta < -tol at tol=1e-14, 1e-10:",
                  delta < -1e-14, delta < -1e-10)
        if label == "original handoff 2":
            fit(model, label + " all", weeks, np.ones(n, bool), observed)
            regrouped = history(model, weeks, k, 4., regroup=True)
            print("  max change in input weekly hazard", np.max(abs(observed - regrouped)))
            fit(model, label + " regrouped split", weeks, split, regrouped)
        if label == "blind 2":
            reference = history(model, weeks, k, 0.)
            other, _ = fit(model, label + " homogeneous split", weeks, split, reference)
            numerator = model.invert_gamma_frailty(np.cumsum(observed), theta)
            denominator = model.invert_gamma_frailty(np.cumsum(reference), other)
            ratio = numerator / denominator
            anchor = model.KCOR_NORMALIZATION_WEEKS_EFFECTIVE - model.DYNAMIC_HVE_SKIP_WEEKS
            print("  endpoint / exact homogeneous denominator", numerator[-1] / sum(reference))
            print("  both fitted: raw", ratio[-1], "reference index", anchor,
                  "reference ratio", ratio[anchor], "anchored", ratio[-1] / ratio[anchor])


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

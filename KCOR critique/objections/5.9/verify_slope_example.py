"""Reproduce the specified-hazard example in Supplementary Information S5.3."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="KCOR checkout to verify")
    args = parser.parse_args()
    sys.path.insert(0, str((args.candidate / "code").resolve(strict=True)))
    from KCOR import fit_theta0_gompertz

    weeks = np.arange(40)
    quiet = weeks < 20
    c = 0.001
    cohort_a = np.where(quiet, c, 2 * c)
    cohort_b = np.full(weeks.size, c)

    print("annual slope | variance A | variance B | anchored KCOR | exact KCOR")
    for annual in (0.065, 0.085, 0.105):
        weekly = annual / 52
        growth = []
        variances = []
        for hazards in (cohort_a, cohort_b):
            (k, theta), diag = fit_theta0_gompertz(hazards, weeks, quiet, 4, weekly)
            assert diag["status"] == "ok", diag["status"]
            assert math.isclose(k, c, rel_tol=1e-6)
            assert math.isclose(theta, weekly / c, rel_tol=1e-6)
            variances.append(theta)
            # The factor 1/theta in the inverse cancels within each cohort.
            growth.append(math.expm1(theta * sum(hazards)) /
                          math.expm1(theta * sum(hazards[quiet])))
        anchored = growth[0] / growth[1]
        exact = math.expm1(60 * weekly) / math.expm1(40 * weekly)
        assert math.isclose(anchored, exact, rel_tol=1e-8)
        print(f"{annual:.3f} | {variances[0]:.5f} | {variances[1]:.5f} | "
              f"{anchored:.8f} | {exact:.8f}")


if __name__ == "__main__":
    main()

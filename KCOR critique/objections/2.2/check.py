"""Check the weekly formula and compare original-sample fits to iteration 1."""
import io
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1.3"))
import table8_bootstrap as replay
import numpy as np


def check_week(module):
    start = np.array([0.0, 0.03, 0.8])
    increment = np.array([0.0002, 0.004, 0.02])
    for theta in (0.0, 1e-10, 1.0, 10.0, 100.0):
        observed = module.weekly_gamma_hazard(start, increment, theta)
        recovered = ((1.0 + theta * start) * np.expm1(theta * observed) / theta
                     if theta else observed)
        np.testing.assert_allclose(recovered, increment, rtol=1e-13)
    # Generate observations independently from exact survival at 61 week boundaries.
    t = np.arange(61)
    k, theta, gamma = 0.004, 4.0, 0.01
    boundaries = k * np.expm1(gamma * t) / gamma
    weekly = np.diff(np.log1p(theta * boundaries) / theta)
    fit, diagnostics = module.fit_theta0_gompertz(
        weekly, t[:-1], np.ones_like(weekly, dtype=bool), 4, gamma, joint_seed=True)
    assert diagnostics["success"] and diagnostics["rmse_hazard"] < 1e-9
    np.testing.assert_allclose(fit, (k, theta), rtol=1e-5)


def main():
    book = io.BytesIO(subprocess.check_output(["git", "cat-file", "blob", replay.INPUT_BLOB], cwd=HERE))
    populations = [replay.read_population(book, sheet) for sheet in replay.SHEETS]
    paired = {}
    for version, mode in (("iteration_1", "revised"), ("iteration_2", "iteration-2")):
        replay.initialize(populations, mode)
        if mode == "iteration-2":
            check_week(replay.K)
        fit = replay.compare.joint_fit
        rows = []

        def record_fit(*args, **kwargs):
            result, diag = fit(*args, **kwargs)
            k, theta = result
            applied, _, _ = replay.K._apply_theta_degenerate_guard(theta, replay.GUARD)
            rows.append(dict(k=k, theta=theta, applied_theta=applied,
                             insufficient_signal=bool(diag.get("insufficient_signal"))))
            return result, diag

        replay.compare.joint_fit = record_fit
        for i, sheet in enumerate(replay.SHEETS):
            rows.clear()
            values, _, failed = replay.estimate(i, 0)
            assert len(rows) == 9 and failed == 0
            if mode == "revised":
                before = json.loads((HERE.parent / "1.6/table8_bootstrap.json").read_text())
                np.testing.assert_allclose(values, before["summary"][sheet]["joint"], rtol=1e-10)
            for (group, dose), row in zip(np.ndindex(3, 3), rows):
                key = f"{sheet}/{('two-dose', 'unvaccinated', 'all')[group]}/pseudo-dose-{dose}"
                paired.setdefault(key, {})[version] = row
    (HERE / "original_fits.json").write_text(json.dumps(paired, indent=2, allow_nan=False) + "\n")
    print("Weekly round trips, known-parameter fit and checkpoint point estimates verified.")


if __name__ == "__main__":
    main()

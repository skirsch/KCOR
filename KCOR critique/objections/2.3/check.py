"""Exercise the gap boundary in the saved N2 fitter and its N3 revision."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1.3/paired-seed-fit"))
import compare


def weekly_observations(time, pulses):
    boundaries = np.arange(len(time) + 1)
    baseline = 0.004 * np.expm1(0.01 * boundaries) / 0.01
    for week, excess in pulses:
        baseline[boundaries > week] += excess
    return np.diff(np.log1p(4.0 * baseline) / 4.0)


def main():
    time = np.arange(90)
    quiet = (time <= 25) | (time >= 41)
    for edition in ("iteration-2", "gap-boundary"):
        compare.initialize(edition)
        module = compare.adapter.K
        # Read the first-pass raw offsets before a refit can respond to a quiet-week pulse.
        config = dict(module.get_gompertz_theta_fit_config(), max_iterations=1)
        with patch.object(module, "get_gompertz_theta_fit_config", return_value=config):
            for week in (None, 39, 40, 41):
                obs = weekly_observations(time, [] if week is None else [(week, 0.010)])
                _, diag = compare.joint_fit(obs, time, quiet, 4, 0.01)
                captured = week == 39 or (week == 40 and edition == "gap-boundary")
                expected = 0.010 if captured else 0.0
                np.testing.assert_allclose(diag["delta_raw_values"], [expected], atol=1e-7)

    # Two separately located shocks, including a gap of just one week.
    quiet = ((time <= 25) | (time >= 41)) & (time != 60)
    obs = weekly_observations(time, [(40, 0.010), (60, 0.006)])
    parameters, diag = compare.joint_fit(obs, time, quiet, 4, 0.01)
    assert diag["success"] and not diag["insufficient_signal"]
    np.testing.assert_allclose(parameters, (0.004, 4.0), rtol=1e-5)
    np.testing.assert_allclose(diag["delta_applied_values"], [0.010, 0.006], atol=1e-7)
    assert diag["rmse_hazard"] < 1e-9
    print("Boundary controls, both gap offsets and the final residual calculation passed.")


if __name__ == "__main__":
    main()

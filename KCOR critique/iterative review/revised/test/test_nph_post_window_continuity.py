"""Check the revised post-wave correction through its production entry point."""

from importlib.util import module_from_spec, spec_from_file_location
from math import exp, inf, log
from pathlib import Path

import numpy as np
from scipy.integrate import quad


def test_nph_tail_continuity() -> None:
    source = Path(__file__).resolve().parents[1] / "code" / "KCOR.py"
    spec = spec_from_file_location("kcor_continuity_edition", source)
    assert spec is not None and spec.loader is not None
    edition = module_from_spec(spec)
    spec.loader.exec_module(edition)

    week = np.arange(5)
    dates = np.datetime64("2022-01-03") + week * np.timedelta64(7, "D")
    weekly_baseline = 0.02

    def correct(observed):
        # With theta = 1, ordinary gamma inversion is exp(H_obs) - 1.
        times = week[:len(observed)].astype(float)
        return edition.apply_nph_correction_post_inversion(
            np.expm1(observed), times, dates[:len(observed)],
            theta_applied=1.0, alpha=2.0, start_dt=dates[1], end_dt=dates[2],
            k_hat=weekly_baseline, gamma_per_week=0.0,
            t_rebased=times, observed_cumhaz=observed,
        )

    ordinary_to_exit = np.array([0.03, 0.10, 0.22])
    observed_to_exit = np.log1p(ordinary_to_exit)
    through_exit, exit_diagnostics = correct(observed_to_exit)
    wave = exit_diagnostics["post_wave_exposure"]
    assert wave > 0
    baseline_at_exit = ordinary_to_exit[0] + 2 * weekly_baseline

    def survival(baseline):
        # Unit exponential frailty z with individual cumulative hazard B*z + W*z².
        return quad(lambda z: exp(-(1 + baseline) * z - wave * z**2), 0, inf)[0]

    # Generate the observed hazard for a known baseline increase by integrating
    # survival independently of the production inverse.
    expected_increase = 0.05
    observed_increase = -log(survival(baseline_at_exit + expected_increase)
                             / survival(baseline_at_exit))
    observed = np.r_[observed_to_exit, observed_to_exit[-1],
                     observed_to_exit[-1] + observed_increase]
    corrected, diagnostics = correct(observed)

    np.testing.assert_allclose(corrected[:3], through_exit, rtol=0, atol=1e-12)
    assert corrected[3] == corrected[2]  # Zero deaths in the first post-wave week.
    np.testing.assert_allclose(corrected[4] - corrected[3], expected_increase,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(diagnostics["post_wave_level_shift"],
                               corrected[2] - ordinary_to_exit[-1], rtol=0, atol=1e-12)
    # The historical constant-offset rule would give a different increment here.
    assert not np.isclose(np.diff(np.expm1(observed))[-1], expected_increase)


if __name__ == "__main__":
    test_nph_tail_continuity()
    print("PASS revised NPH post-window continuity")

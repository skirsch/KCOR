"""Check the opening reference and exposure carried out of one- and two-week waves."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1.3/paired-seed-fit"))
import compare


def run(module, cumulative, first, last, *, alpha=1.18, skip=0):
    time = np.arange(len(cumulative))
    dates = np.datetime64("2021-01-04") + time * np.timedelta64(7, "D")
    cumulative = np.array(cumulative, dtype=float)
    return module.apply_nph_correction_post_inversion(
        cumulative, time, dates, 1.0, alpha, dates[first], dates[last], 0.001, 0.0,
        t_rebased=time - skip, observed_cumhaz=np.log1p(cumulative))


def check_positive(module, cumulative, first, last, opening, modeled_weeks, skip=0):
    curve, diag = run(module, cumulative, first, last, skip=skip)
    reference = opening + 0.001 * modeled_weeks
    exit_total = cumulative[last]
    factor = module.gamma_moment_alpha(module.propagate_theta(1.0, np.array([exit_total])), 1.18)[0]
    expected = (exit_total - reference) / factor
    assert diag["wave_points"] == diag["corrected_points"] == modeled_weeks
    np.testing.assert_allclose(diag["post_wave_exposure"], expected, rtol=1e-12)
    np.testing.assert_allclose(curve[last], reference + expected, rtol=1e-12)
    if first:
        np.testing.assert_array_equal(curve[:first], cumulative[:first])
    return curve


def main():
    series = [0.020, 0.026, 0.027]
    compare.initialize("gap-boundary")
    _, old = run(compare.adapter.K, series, 1, 1)
    assert old["corrected_points"] == 0 and old["post_wave_exposure"] == 0

    compare.initialize("wave-entry")
    module = compare.adapter.K
    check_positive(module, series, 1, 1, 0.020, 1)
    check_positive(module, [0.020, 0.026, 0.027, 0.028], 1, 2, 0.020, 2)
    check_positive(module, [0.006, 0.007], 0, 0, 0.0, 1)
    curve = check_positive(module, [0.0, 0.0, 0.006, 0.007], 0, 2, 0.0, 1, skip=2)
    np.testing.assert_array_equal(curve[:2], [0.0, 0.0])
    for series_at_entry in ([0.020, 0.021, 0.022], [0.020, 0.0205, 0.022]):
        curve, diag = run(module, series_at_entry, 1, 1)
        assert diag["corrected_points"] == 0 and diag["post_wave_exposure"] == 0
        np.testing.assert_allclose(curve, series_at_entry)
    curve, diag = run(module, series, 1, 1, alpha=1.0)
    np.testing.assert_array_equal(curve, series)
    assert diag["correction_mode"] == "identity"
    curve, diag = run(module, [0.0, 0.0, 0.006], 0, 1, skip=2)
    assert diag["wave_points"] == 0
    np.testing.assert_array_equal(curve, [0.0, 0.0, 0.006])
    print("Wave-entry, carry-forward, stabilization, nonpositive excess and identity checks passed.")


if __name__ == "__main__":
    main()

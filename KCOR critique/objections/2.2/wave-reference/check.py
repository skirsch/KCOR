"""Check the revised wave reference against independently integrated weekly hazards."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "1.3/paired-seed-fit"))
import compare


def check():
    compare.initialize("wave-week")
    model = compare.adapter.K
    time = np.arange(5, dtype=float)
    rebased = time - 2
    dates = np.datetime64("2022-01-03") + np.arange(5) * np.timedelta64(7, "D")
    rate = .02
    for gamma in (.085/52, .2, -.2, 1e-12, 0.):
        increments = [quad(lambda s: rate * np.exp(gamma*s), t, t+1)[0] for t in rebased[2:]]
        baseline = np.r_[0., 0., np.cumsum(increments)]
        for first, last in ((0, 2), (3, 3), (3, 4)):
            def run(curve, alpha):
                return model.apply_nph_correction_post_inversion(
                    curve, time, dates, 1., alpha, dates[first], dates[last], rate, gamma,
                    t_rebased=rebased, observed_cumhaz=np.log1p(curve))
            # With baseline alone, the entire integrated week must remain baseline.
            corrected, info = run(baseline, 1.18)
            np.testing.assert_allclose(corrected, baseline, rtol=0, atol=1e-12)
            assert info["post_wave_exposure"] == 0 and info["corrected_points"] == 0
            # Put a known excess at exit to test both sides of the threshold.
            observed = baseline.copy()
            observed[last:] += .004
            corrected, info = run(observed, 1.18)
            factor = model.gamma_moment_alpha(model.propagate_theta(1., observed), 1.18)[last]
            np.testing.assert_allclose(info["post_wave_exposure"], .004/factor, rtol=1e-12)
            np.testing.assert_allclose(corrected[last], baseline[last] + .004/factor, rtol=1e-12)
            np.testing.assert_array_equal(run(observed, 1.)[0], observed)
    print("Passed independent integration, wave entry and recurrence, zero excess and alpha=1 checks.")


def write_workbook(source, destination, edition):
    import table8_bootstrap as table
    os.environ["MONTE_CARLO"] = "0"
    os.chdir(source.parent)
    compare.initialize(edition)
    model = compare.adapter.K
    model.OVERRIDE_DOSE_PAIRS = list(table.PAIRS)
    model.fit_theta0_gompertz = compare.joint_fit
    model.process_workbook(str(source), str(destination))


def check_workbook(edition):
    import table8_bootstrap as table
    original = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB], cwd=HERE)
    populations = [table.read_population(io.BytesIO(original), sh) for sh in table.SHEETS]
    table.initialize(populations, edition)
    expected = [table.estimate(i, 0)[0] for i in range(2)]
    with TemporaryDirectory() as tmp:
        source, destination = (Path(tmp) / name for name in ("input.xlsx", "output.xlsx"))
        source.write_bytes(original)
        with ProcessPoolExecutor(1) as pool:
            pool.submit(write_workbook, source, destination, edition).result()
        actual = table.workbook_points(destination, populations)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    print("All six joint estimates matched the full workbook:", actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", action="store_true")
    parser.add_argument("--edition", choices=("wave-week", "entry-moment"), default="wave-week",
                        help="Revised edition to use for the independent workbook check.")
    args = parser.parse_args()
    check()
    if args.workbook:
        sys.path.insert(0, str(HERE.parents[1] / "1.3"))
        check_workbook(args.edition)

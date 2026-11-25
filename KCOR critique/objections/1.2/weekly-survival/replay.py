"""Paired Table 8 replay for the experimental weekly survival correction."""
import argparse
import io
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "1.3"))
import table8_bootstrap as table
from trial import make_correction


def initialize(populations):
    global records
    records = []
    table.initialize(populations, "entry-moment")
    correction = make_correction(table.K)

    def logged(*args, **kwargs):
        result, diag = correction(*args, **kwargs)
        if diag["correction_mode"] == "weekly_survival_trial":
            records.append((diag["wave_points"], diag["deficit_weeks"], diag["max_survival_residual"]))
        return result, diag

    table.K.apply_nph_correction_post_inversion = logged


def estimate(task):
    enrollment, draw = task
    outcomes = []
    for joint in (True, False):
        records.clear()
        points, flags, failed = table.estimate(enrollment, draw, joint=joint)
        outcomes.append((points, flags, failed, sum(r[0] for r in records), sum(r[1] for r in records),
                         max((r[2] for r in records), default=0.0)))
    return enrollment, draw, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    options = parser.parse_args()
    prior = HERE.parent / "entry-moment"
    report = json.loads((prior / "table8_bootstrap.json").read_text())
    assert (report["input_blob"], report["seed"], report["date"], report["draws_per_enrollment"]) == (
        table.INPUT_BLOB, table.SEED, str(table.REPORT.date()), 999)
    workbook = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB], cwd=HERE)
    populations = [table.read_population(io.BytesIO(workbook), sheet) for sheet in table.SHEETS]
    with np.load(prior / "table8-draws.npz") as archive:
        np.testing.assert_array_equal(archive["enrollments"], table.SHEETS)
        np.testing.assert_array_equal(archive["pairs"], table.PAIRS)
        np.testing.assert_array_equal(archive["draw"], np.arange(1000))
        np.testing.assert_array_equal(archive["fitters"], ["joint", "fixed"])
        before = archive["points"]

    table.initialize(populations, "entry-moment")
    for i in range(len(populations)):
        for draw in (0, 500, 999):
            actual = [table.estimate(i, draw, joint=joint)[0] for joint in (True, False)]
            np.testing.assert_allclose(actual, before[:, i, draw], rtol=1e-10, atol=1e-12)
    after = np.empty_like(before)
    diagnostics = np.zeros((2, 2, 1000, 4), dtype=int)
    residual = 0.0
    tasks = [(i, draw) for draw in range(1000) for i in range(2)]
    with ProcessPoolExecutor(options.workers, initializer=initialize, initargs=(populations,)) as pool:
        for n, (i, draw, outcomes) in enumerate(pool.map(estimate, tasks, chunksize=4), 1):
            for j, (point, flags, failed, weeks, deficits, error) in enumerate(outcomes):
                after[j, i, draw] = point
                diagnostics[j, i, draw] = flags, failed, weeks, deficits
                residual = max(residual, error)
            if n % 200 == 0:
                print(f"{n}/{len(tasks)} enrollment draws", flush=True)
    assert np.isfinite(after).all() and residual < 1e-10
    result = {key: report[key] for key in ("input_blob", "seed", "date", "pairs", "draws_per_enrollment")}
    result["max_survival_residual"] = residual
    result["summary"] = {}
    for i, sheet in enumerate(table.SHEETS):
        prior_summary = report["summary"][sheet]
        flags = diagnostics[0, i, :, 0]
        assert (flags[0], int((flags[1:] > 0).sum()), diagnostics[0, i, 1:, 1].sum()) == (
            prior_summary["original_insufficient_signal"], prior_summary["draws_with_insufficient_signal"],
            prior_summary["unsuccessful_stratum_fits"])
        result["summary"][sheet] = {}
        for j, fitter in enumerate(("joint", "fixed")):
            low, high = np.quantile(after[j, i, 1:], [.025, .975], axis=0)
            old_low, old_high = np.quantile(before[j, i, 1:], [.025, .975], axis=0)
            shift = after[j, i] - before[j, i]
            shift_low, shift_high = np.quantile(shift[1:], [.025, .975], axis=0)
            result["summary"][sheet][fitter] = dict(
                before=before[j, i, 0].tolist(), before_lower=old_low.tolist(), before_upper=old_high.tolist(),
                trial=after[j, i, 0].tolist(), lower=low.tolist(), upper=high.tolist(), width=(high-low).tolist(),
                change=shift[0].tolist(), change_lower=shift_low.tolist(), change_upper=shift_high.tolist(),
                original_weeks=int(diagnostics[j, i, 0, 2]), original_deficits=int(diagnostics[j, i, 0, 3]),
                bootstrap_weeks=int(diagnostics[j, i, 1:, 2].sum()),
                bootstrap_deficits=int(diagnostics[j, i, 1:, 3].sum()))
    np.savez_compressed(HERE / "table8-draws.npz", points=after, fitters=["joint", "fixed"],
                        enrollments=table.SHEETS, pairs=table.PAIRS, draw=np.arange(1000), diagnostics=diagnostics)
    (HERE / "table8-comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

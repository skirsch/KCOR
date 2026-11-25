"""Replay the Table 8 joint and fixed fits on the 3.1 person-bootstrap draws."""
import io
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CRITIQUE = HERE.parents[1]
sys.path.insert(0, str(CRITIQUE / "1.3"))
import table8_bootstrap as table
import numpy as np


def estimate(task):
    enrollment, draw = task
    return [table.estimate(enrollment, draw, joint=joint) for joint in (True, False)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--entry-moment", action="store_true",
                        help="Compare the wave-entry survivor correction with the full-week reference edition.")
    args = parser.parse_args()
    if args.entry_moment:
        prior, output = HERE, CRITIQUE / "1.2/entry-moment"
        previous, current = "wave-week", "entry-moment"
    else:
        prior, output = CRITIQUE / "3.1", HERE
        previous, current = "gap-refit", "wave-week"
    report = json.loads((prior / "table8_bootstrap.json").read_text())
    assert (report["input_blob"], report["seed"], report["date"], report["draws_per_enrollment"]) == (
        table.INPUT_BLOB, table.SEED, str(table.REPORT.date()), 999)
    original = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB], cwd=HERE)
    populations = [table.read_population(io.BytesIO(original), sh) for sh in table.SHEETS]
    with np.load(prior / "table8-draws.npz") as archive:
        np.testing.assert_array_equal(archive["enrollments"], table.SHEETS)
        np.testing.assert_array_equal(archive["pairs"], table.PAIRS)
        np.testing.assert_array_equal(archive["draw"], np.arange(1000))
        labels = ["joint", "fixed"] if args.entry_moment else ["corrected_joint", "corrected_fixed"]
        np.testing.assert_array_equal(archive["fitters"][-2:], labels)
        before = archive["points"][-2:]
    table.initialize(populations, previous)
    for enrollment in range(2):
        for draw in (0, 500, 999):
            old = [v[0] for v in estimate((enrollment, draw))]
            np.testing.assert_allclose(old, before[:, enrollment, draw], atol=1e-12, rtol=1e-10)
    after = np.empty_like(before)
    counts = np.empty((2, 1000, 2), dtype=int)
    tasks = [(i, draw) for draw in range(1000) for i in range(2)]
    with ProcessPoolExecutor(args.workers, initializer=table.initialize, initargs=(populations, current)) as pool:
        for n, ((i, draw), outcomes) in enumerate(zip(tasks, pool.map(estimate, tasks, chunksize=4)), 1):
            after[:, i, draw] = [outcome[0] for outcome in outcomes]
            counts[i, draw] = outcomes[0][1:]
            if n % 200 == 0:
                print(f"{n}/{len(tasks)} enrollment draws", flush=True)
    assert np.isfinite(after).all()
    for i, sheet in enumerate(table.SHEETS):
        row = report["summary"][sheet]
        assert (counts[i, 0, 0], (counts[i, 1:, 0] > 0).sum(), counts[i, 1:, 1].sum()) == (
            row["original_insufficient_signal"], row["draws_with_insufficient_signal"], row["unsuccessful_stratum_fits"])
        for j, label in enumerate(("joint", "fixed")):
            prefix = "" if j == 0 else "fixed_"
            lo, hi = np.quantile(after[j, i, 1:], [.025, .975], axis=0)
            shifts = after[j, i] - before[j, i]
            delta_lo, delta_hi = np.quantile(shifts[1:], [.025, .975], axis=0)
            row[label] = after[j, i, 0].tolist()
            for key, value in (("lower", lo), ("upper", hi), ("width", hi-lo),
                               ("change", shifts[0]), ("change_lower", delta_lo), ("change_upper", delta_hi)):
                row[prefix+key] = value.tolist()
    np.savez_compressed(output / "table8-draws.npz", points=after, fitters=["joint","fixed"],
                        enrollments=table.SHEETS, pairs=table.PAIRS, draw=np.arange(1000))
    (output / "table8_bootstrap.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

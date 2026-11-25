"""Compare the Table 8 fitters on identical person draws. Run from the repository root."""
import csv
import gzip
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1.3"))
import table8_bootstrap as table
import numpy as np

GROUPS = ("two-dose", "unvaccinated", "all-persons")
FLAGS = ("insufficient_signal", "insufficient_deaths", "theta_hit_bound", "bad_fit",
         "identifiability_flag", "delta_negative", "delta_negative_clamped")


def initialize(populations):
    global FITTERS
    table.initialize(populations, "wave-entry")
    previous = table.compare.joint_fit
    table.initialize(populations, "gap-refit")
    FITTERS = (previous, table.compare.joint_fit, table.compare.production_fit)


def replay(task):
    enrollment, draw = task
    results = []
    for fitter in FITTERS:
        diagnostics = []

        def logged_fit(*args, **kwargs):
            estimate, diag = fitter(*args, **kwargs)
            diagnostics.append((diag["status"], int(diag["success"]),
                                *(int(bool(diag.get(key))) for key in FLAGS),
                                int("negative delta on first reconstruction pass" in diag["message"])))
            return estimate, diag

        table.compare.joint_fit = logged_fit
        points, flags, failed = table.estimate(enrollment, draw)
        assert len(diagnostics) == 9
        results.append((points, flags, failed, diagnostics))
    return enrollment, draw, results


def main():
    workbook = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB])
    populations = [table.read_population(io.BytesIO(workbook), sh) for sh in table.SHEETS]
    tasks = [(i, d) for d in range(1000) for i in range(2)]
    editions = ("previous_joint", "corrected_joint", "corrected_fixed")
    points = np.empty((3, 2, 1000, 3))
    counts = np.empty((3, 2, 1000, 2), dtype=int)
    with gzip.open(HERE/"subgroup-diagnostics.csv.gz", "wt", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(("fitter", "enrollment", "draw", "subgroup", "pseudo_dose", "status",
                         "success", *FLAGS, "first_gap_reset"))
        with ProcessPoolExecutor(6, initializer=initialize, initargs=(populations,)) as pool:
            for n, (i, draw, results) in enumerate(pool.map(replay, tasks, chunksize=4), 1):
                for k, (values, flags, failed, diagnostics) in enumerate(results):
                    points[k, i, draw] = values
                    counts[k, i, draw] = flags, failed
                    for g, diag in enumerate(diagnostics):
                        writer.writerow((editions[k], table.SHEETS[i], draw, GROUPS[g//3], g%3, *diag))
                if n % 200 == 0:
                    print(f"{n}/{len(tasks)} enrollment draws", flush=True)

    assert np.isfinite(points).all()
    previous = json.loads((HERE.parent/"2.4/table8_bootstrap.json").read_text())
    assert previous["seed"] == table.SEED and previous["draws_per_enrollment"] == 999
    report = {key:previous[key] for key in ("input_blob", "seed", "draws_per_enrollment", "date", "pairs")}
    report["summary"] = {}
    for i, sheet in enumerate(table.SHEETS):
        lower, upper = np.quantile(points[:,i,1:], [.025, .975], axis=1)
        old = previous["summary"][sheet]
        for value, archived in ((points[0,i,0], old["joint"]), (lower[0], old["lower"]),
                                (upper[0], old["upper"])):
            np.testing.assert_allclose(value, archived, rtol=1e-10, atol=1e-12)
        assert (counts[0,i,1:,0] > 0).sum() == old["draws_with_insufficient_signal"]
        report["summary"][sheet] = dict(
            joint=points[1,i,0].tolist(), lower=lower[1].tolist(), upper=upper[1].tolist(),
            width=(upper[1]-lower[1]).tolist(), fixed=points[2,i,0].tolist(),
            fixed_lower=lower[2].tolist(), fixed_upper=upper[2].tolist(),
            original_insufficient_signal=int(counts[1,i,0,0]), people=int(populations[i][1].sum()),
            draws_with_insufficient_signal=int((counts[1,i,1:,0] > 0).sum()),
            unsuccessful_stratum_fits=int(counts[1,i,1:,1].sum()))
    np.savez_compressed(HERE/"table8-draws.npz", fitters=editions, enrollments=table.SHEETS,
                        pairs=table.PAIRS, points=points, draw=np.arange(1000))
    (HERE/"table8_bootstrap.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

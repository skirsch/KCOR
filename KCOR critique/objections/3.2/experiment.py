"""Exploratory entry-survival fit and paired Table 8 replay. No production overrides on disk."""
import hashlib
import inspect
import io
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1.3"))
import table8_bootstrap as table
import numpy as np


def entry_fit(model):
    """Use the additive gamma identity to translate the quiet-window rate to time zero."""
    original = model.fit_theta0_gompertz
    source = inspect.getsource(original)
    seed = "            H_gom_full = k_fixed * unit_start"
    boundaries = "    reentry_indices = [start for start, _ in segments[1:]]"
    assert source.count(seed) == source.count(boundaries) == 1
    source = source.replace(seed, """
            if first_seg_start > post_idx[0]:
                before = post_idx[post_idx < first_seg_start]
                observed = float(np.sum(h_arr[before]))
                # With effective quiet rate r, exp(theta H_entry) =
                # (k/r) * (1 + theta*r*F_entry). Keep the existing
                # nonnegative excess convention and the original seed rate bounds.
                log_ratio = theta_seed * observed - np.log1p(
                    theta_seed * k_fixed * unit_start[first_seg_start])
                k_fixed = np.exp(min(np.log(0.1), np.log(k_fixed) + max(0., log_ratio)))
""" + seed)
    source = source.replace(boundaries, boundaries + """
    if joint_seed and first_seg_start > post_idx[0]:
        reentry_indices.insert(0, first_seg_start)
""")
    namespace = dict(original.__globals__)
    exec(compile(source, "<3.2 entry-survival experiment>", "exec"), namespace)
    return partial(namespace[original.__name__], joint_seed=True)


def initialize(populations):
    table.initialize(populations, "gap-refit")
    table.compare.joint_fit = entry_fit(table.K)


def run_draw(task):
    enrollment, draw = task
    return enrollment, draw, table.estimate(enrollment, draw)


def main():
    book = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB], cwd=HERE)
    populations = [table.read_population(io.BytesIO(book), s) for s in table.SHEETS]
    archive = HERE.parent / "3.1/table8-draws.npz"
    with np.load(archive) as prior:
        assert list(prior["enrollments"]) == list(table.SHEETS)
        index = list(prior["fitters"]).index("corrected_joint")
        current = prior["points"][index].copy()
        np.testing.assert_array_equal(prior["draw"], np.arange(1000))
        np.testing.assert_array_equal(prior["pairs"], table.PAIRS)
    table.initialize(populations, "gap-refit")
    for i in range(2):
        for d in (0, 1, 501, 999):
            np.testing.assert_allclose(table.estimate(i, d)[0], current[i,d], rtol=1e-12)
    proposed = np.empty_like(current)
    flags = np.empty((2, 1000, 2), dtype=int)
    tasks = [(i,d) for d in range(1000) for i in range(2)]
    with ProcessPoolExecutor(6, initializer=initialize, initargs=(populations,)) as pool:
        for n,(i,d,(ratios, flagged, failed)) in enumerate(pool.map(run_draw, tasks, chunksize=5),1):
            proposed[i,d] = ratios
            flags[i,d] = flagged, failed
            if n % 400 == 0:
                print(f"{n}/{len(tasks)} enrollment draws", flush=True)
    assert np.isfinite(proposed).all()
    np.testing.assert_allclose(current[0], proposed[0], rtol=1e-13, atol=1e-14)
    summary = {}
    for i,s in enumerate(table.SHEETS):
        summary[s] = {}
        for label, points in (("current",current[i]), ("experimental",proposed[i]),
                              ("paired_change",proposed[i]-current[i])):
            low,high = np.quantile(points[1:], [.025,.975], axis=0)
            summary[s][label] = dict(point=points[0].tolist(), lower=low.tolist(),
                                     upper=high.tolist(), width=(high-low).tolist())
        summary[s]["experimental_original_flagged_failed"] = flags[i,0].tolist()
        summary[s]["experimental_draws_with_flagged_or_failed_separately"] = (flags[i,1:]>0).sum(axis=0).tolist()
    np.savez_compressed(HERE/"entry-draws.npz", proposed=proposed, flags=flags, pairs=table.PAIRS)
    (HERE/"empirical.json").write_text(json.dumps(dict(seed=table.SEED, input_blob=table.INPUT_BLOB,
        draws=999, archived_comparator_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        summary=summary), indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()

"""Recompute quantities used to adjudicate reviews from one frozen current edition."""
import io
import inspect
import json
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def check(root):
    table = root/"analysis/table8"
    sys.path.insert(0, str(table))
    import table8_bootstrap as replay

    book = io.BytesIO((table/"input.xlsx").read_bytes())
    populations = [replay.read_population(book, sheet) for sheet in replay.SHEETS]
    replay.initialize(populations, "revised")
    saved = json.loads((table/"table8_bootstrap.json").read_bytes())["summary"]
    for index, sheet in enumerate(replay.SHEETS):
        points, flags, failures = replay.estimate(index, 0)
        row = saved[sheet]
        np.testing.assert_allclose(points, row["joint"], rtol=1e-10)
        assert flags == row["original_insufficient_signal"] and failures == 0
        print(sheet, "endpoints", points, "flagged fits", flags, "bootstrap draws with flags",
              row["draws_with_insufficient_signal"])
        dates, time, quiet, keep, counts = replay.DATA[index]
        replay.DATA[index] = dates, time, quiet & (dates <= replay.REPORT.to_datetime64()), keep, counts
        print(sheet, "quiet weeks through reporting date", replay.estimate(index, 0))
        replay.DATA[index] = dates, time, quiet, keep, counts

    # Isolate the final quiet-only mask while preserving the seed and all data.
    fit = replay.compare.production_fit
    source = inspect.getsource(fit)
    mask = "fit_mask_theta = (quiet | anchor_mask) & post_skip"
    assert source.count(mask)==1
    namespace = dict(fit.__globals__)
    exec(compile(source.replace(mask, "fit_mask_theta = quiet & post_skip"), "<quiet-only check>", "exec"), namespace)
    replay.compare.joint_fit = partial(namespace[fit.__name__], joint_seed=True)
    for index, sheet in enumerate(replay.SHEETS):
        print(sheet, "quiet-only fit, full retrospective follow-up", replay.estimate(index, 0))

    paired = json.loads((root/"validation/person_bootstrap/paired.json").read_bytes())["paired_results"]
    for method in ["fixed", "joint"]:
        rows = [pair[method] for pair in paired]
        print(method, "covered", sum(row["covered"] for row in rows), "/", len(rows),
              "mean width", np.mean([row["high"]-row["low"] for row in rows]))
    scan = pd.read_csv(root/"analysis/quiet_windows/windows.csv")
    print("quiet windows passed", int(scan.passed.sum()), "/", len(scan),
          "theta range", scan.theta_raw.min(), scan.theta_raw.max())
    cox = pd.read_csv(root/"analysis/cox_intervals/figure_intervals.csv")
    print("Cox experiment: KCOR intervals including 1", int(((cox.lower95<=1)&(cox.upper95>=1)).sum()), "/", len(cox))
    pilot = pd.read_csv(root/"test/sim_grid/out/bootstrap_coverage_replicates.csv")
    print("distinct outer estimates by pilot scenario", pilot.groupby("scenario_name").point_estimate.nunique().to_dict())
    sweep = pd.read_csv(root/"test/quiet_window_contamination/out/quiet_contamination_long.csv")
    zero = sweep[(sweep.contamination_type=="comorbidity") & (sweep.epsilon_comorb==0) & (sweep.rep>=0)]
    assert len(zero)==20
    print("zero contamination: above one", int((zero.KCOR_asymptote>1).sum()), "/", len(zero),
          "mean", zero.KCOR_asymptote.mean())


if __name__=="__main__":
    check(Path(sys.argv[1]).resolve())

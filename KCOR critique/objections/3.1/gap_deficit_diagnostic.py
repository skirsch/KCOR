"""Report pointwise bootstrap intervals for signed first-pass gap offsets."""
import csv
import io
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from replay import GROUPS, table


HERE = Path(__file__).resolve().parent


def initialize(populations):
    global fit_joint
    table.initialize(populations, "gap-refit")
    fit_joint = table.compare.joint_fit


def first_pass_offsets(hazard, time, quiet, gamma, diag):
    """Reconstruct the first pass from the seed parameters retained by the fit."""
    post_skip = (time >= 0) & np.isfinite(hazard)
    quiet_weeks = np.flatnonzero(quiet & post_skip)
    reentry = quiet_weeks[1:][np.diff(quiet_weeks) > 1]
    observed = np.cumsum(np.where(post_skip, hazard, 0.0))[reentry - 1]
    theta = diag["theta0_init"]
    individual = np.expm1(theta * observed) / theta if theta else observed
    expected = diag["k_hat"] * (np.expm1(gamma * time[reentry]) / gamma)
    return reentry, np.diff(individual - expected, prepend=0.0)


def replay_draw(task):
    enrollment, draw = task
    rows = []
    dates = table.DATA[enrollment][0]

    def record(hazard, time, quiet, anchor_weeks, gamma, **kwargs):
        args = (hazard, time, quiet, anchor_weeks, gamma)
        estimate, diag = fit_joint(*args, **kwargs)
        reentry, offsets = first_pass_offsets(hazard, time, quiet, gamma, diag)
        if draw == 0:
            get_config = table.K.get_gompertz_theta_fit_config
            try:
                table.K.get_gompertz_theta_fit_config = lambda: {**get_config(), "max_iterations": 1}
                _, one_pass = fit_joint(*args, **kwargs)
            finally:
                table.K.get_gompertz_theta_fit_config = get_config
            np.testing.assert_allclose(offsets, one_pass["delta_raw_values"], rtol=0, atol=1e-12)
        group, dose = divmod(len(rows), 3)
        rows.append([(GROUPS[group], dose, str(dates[i])[:10], offset)
                     for i, offset in zip(reentry, offsets)])
        return estimate, diag

    table.compare.joint_fit = record
    points, _, _ = table.estimate(enrollment, draw)
    assert len(rows) == 9
    return enrollment, draw, points, [offset for group in rows for offset in group]


def main():
    workbook = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB], cwd=HERE)
    populations = [table.read_population(io.BytesIO(workbook), sheet) for sheet in table.SHEETS]
    with np.load(HERE / "table8-draws.npz") as archive:
        assert tuple(archive["enrollments"]) == table.SHEETS
        assert np.array_equal(archive["draw"], np.arange(1000))
        archived_points = archive["points"][list(archive["fitters"]).index("corrected_joint")]
    tasks = [(i, draw) for draw in range(1000) for i in range(len(populations))]
    offsets_by_fit = {}
    with ProcessPoolExecutor(6, initializer=initialize, initargs=(populations,)) as pool:
        for n, (i, draw, points, rows) in enumerate(pool.map(replay_draw, tasks, chunksize=4), 1):
            np.testing.assert_allclose(points, archived_points[i, draw], rtol=1e-10, atol=1e-12)
            for group, dose, date, offset in rows:
                key = (table.SHEETS[i], group, dose, date)
                if key not in offsets_by_fit:
                    offsets_by_fit[key] = np.full(1000, np.nan)
                offsets_by_fit[key][draw] = offset
            if n % 400 == 0:
                print(f"{n}/{len(tasks)} Table 8 draws verified", flush=True)

    summary = []
    for key, offsets in offsets_by_fit.items():
        # A missing seed fit is not evidence of zero gap excess.
        draws = offsets[1:]
        finite = np.isfinite(draws)
        complete = np.isfinite(offsets[0]) and finite.all()
        if complete:
            lower, upper = np.quantile(draws, [.025, .975])
            flag = "deficit" if upper < 0 else ""
        else:
            lower, upper = np.nan, np.nan
            flag = "incomplete"
        summary.append((*key, offsets[0], lower, upper, int(finite.sum()), flag))

    path = HERE / "gap-deficit-summary.csv"
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("enrollment", "subgroup", "pseudo_dose", "quiet_reentry",
                         "first_pass_offset", "lower95", "upper95", "finite_draws", "flag"))
        writer.writerows(summary)
    print(f"{len(summary)} gap estimates, {sum(row[-1] == 'deficit' for row in summary)} deficit flags.")


if __name__ == "__main__":
    main()

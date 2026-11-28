"""Person-bootstrap percentile intervals for the archived Czech Table 8 joint-fit trial."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "paired-seed-fit"))
import compare
import numpy as np
import pandas as pd

INPUT_BLOB = "f2dbb3d7a39d47a06821048de042285df46785ba"
SHEETS = ("2021_24", "2022_06")
PAIRS = ((1, 0), (2, 0), (2, 1))
REPORT = pd.Timestamp("2022-12-26")
SEED = 20261121


def read_population(workbook, sheet):
    frame = pd.read_excel(workbook, sheet_name=sheet, parse_dates=["DateDied"])
    start = pd.Timestamp.fromisocalendar(*map(int, sheet.split("_")), 1)
    frame = frame[frame.DateDied >= start]
    group = frame.groupby(["YearOfBirth", "Dose", "DateDied"])[["Alive", "Dead"]].sum()
    dates = pd.date_range(start, frame.DateDied.max(), freq="7D")
    index = pd.MultiIndex.from_product([[1940, 1950, 1960], range(3), dates], names=group.index.names)
    assert group.index.equals(index)
    alive, dead = (group[c].to_numpy(dtype=int).reshape(3, 3, -1) for c in ("Alive", "Dead"))
    np.testing.assert_array_equal(alive[:, :, :-1] - dead[:, :, :-1], alive[:, :, 1:])
    outcomes = np.concatenate([dead, (alive-dead)[:, :, -1:]], axis=2)
    # Rows are dose-2 people, unvaccinated people, and everyone. Remove overlap
    # before sampling death weeks and survivors, then rebuild it after each draw.
    disjoint = outcomes.copy()
    disjoint[2] -= outcomes[0] + outcomes[1]
    assert disjoint.min() >= 0 and disjoint.sum() == alive[2, :, 0].sum()
    return dates, disjoint


def initialize(populations, weekly_update="revised"):
    global K, CONFIG, WAVE, GUARD, DATA
    os.environ["MONTE_CARLO"] = "0"
    compare.initialize(weekly_update)
    K = compare.adapter.K
    CONFIG = K.get_gompertz_theta_fit_config()
    WAVE = K.get_covid_correction_config()
    GUARD = K.get_theta_degenerate_fit_config()
    assert WAVE["apply_correction"] and WAVE["apply_alpha"] == 1.18
    DATA = []
    for dates, counts in populations:
        time = np.arange(len(dates), dtype=float)
        quiet, _, _ = K._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(), time)
        target = dates.get_loc(REPORT)
        # All in-wave dates and the anchor are retained. After wave exit the
        # production inversion depends only on survival at exit and at target.
        keep = (dates <= WAVE["end_dt"]) | (time == K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE) | (time == target)
        DATA.append((dates.to_numpy(), time, quiet, keep, counts))


def estimate(index, draw, joint=True):
    dates, time, quiet, keep, disjoint = DATA[index]
    counts = disjoint.copy()
    if draw:
        rng = np.random.default_rng([SEED, index, draw])
        total = int(disjoint.sum())
        counts = rng.multinomial(total, disjoint.ravel() / total).reshape(disjoint.shape)
    counts[2] += counts[0] + counts[1]
    alive = np.cumsum(counts[:, :, ::-1], axis=2)[:, :, ::-1][:, :, :-1]
    dead = counts[:, :, :-1]
    mr = dead / (alive + K.EPS)
    early = mr[:, :, :4]
    weights = np.nanmean(np.where(early > K.EPS, early, np.nan), axis=(1, 2)) * alive[:, :, :4].sum(axis=(1, 2))
    growth = np.empty((3, 3))
    flags = 0
    failed = 0
    skip = K.DYNAMIC_HVE_SKIP_WEEKS
    fit = compare.joint_fit if joint else compare.production_fit
    for a, d in np.ndindex(growth.shape):
        hazard = K.hazard_from_mr(np.clip(mr[a, d], 0, .999))
        effective = np.where(time >= skip, hazard, 0.)
        (k, theta), diag = fit(effective, time-skip, quiet, CONFIG["k_anchor_weeks"],
                               CONFIG["gamma_per_week"], deaths_arr=dead[a, d])
        failed += int(not diag["success"])
        flags += int(bool(diag.get("insufficient_signal")))
        theta = theta if diag["success"] and np.isfinite(theta) and theta >= 0 else 0.
        theta, _, _ = K._apply_theta_degenerate_guard(theta, GUARD)
        observed = np.cumsum(effective)[keep]
        inverted = K.invert_gamma_frailty(observed, theta)
        corrected, _ = K.apply_nph_correction_post_inversion(
            inverted, time[keep], dates[keep], theta, WAVE["apply_alpha"],
            WAVE["start_dt"], WAVE["end_dt"], k, CONFIG["gamma_per_week"], observed_cumhaz=observed)
        anchor = np.flatnonzero(time[keep] == K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE).item()
        growth[a, d] = corrected[-1] / corrected[anchor]
    pooled = [float(np.exp(np.average(np.log(growth[:, n]/growth[:, d]), weights=weights))) for n, d in PAIRS]
    return pooled, flags, failed


def one_draw(task):
    index, draw = task
    return index, draw, estimate(index, draw)


def workbook_points(path, populations):
    frame = pd.read_excel(path, sheet_name="dose_pairs")
    rows = frame[(frame.Date == REPORT) & (frame.YearOfBirth > 0)]
    points = []
    for sheet, (_, cells) in zip(SHEETS, populations):
        strata = cells.copy()
        strata[2] = cells.sum(axis=0)
        alive = np.flip(np.cumsum(np.flip(strata, axis=2), axis=2), axis=2)
        early_rate = strata[:, :, :4] / alive[:, :, :4]
        weights = np.nanmean(np.where(early_rate > 0, early_rate, np.nan), axis=(1, 2)) * alive[:, :, :4].sum(axis=(1, 2))
        cohort = rows[rows.EnrollmentDate == sheet]
        ratios = []
        for n, d in PAIRS:
            by_age = cohort[(cohort.Dose_num == n) & (cohort.Dose_den == d)].set_index("YearOfBirth")
            ratios.append(float(np.exp(np.average(np.log(by_age.loc[[1940,1950,1960], "KCOR"]), weights=weights))))
        points.append(ratios)
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=999)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-workbook", type=Path, help="Full-workbook reference for a later revision of the fitter.")
    edition = parser.add_mutually_exclusive_group()
    for mode in ("iteration-2", "gap-boundary", "wave-entry"):
        edition.add_argument("--" + mode, dest="edition", action="store_const", const=mode)
    args = parser.parse_args()
    if args.edition and not args.check_workbook:
        parser.error("Iteration 2 replays require --check-workbook")
    book = io.BytesIO(subprocess.check_output(["git", "cat-file", "blob", INPUT_BLOB], cwd=HERE))
    populations = [read_population(book, sh) for sh in SHEETS]
    weekly_update = args.edition or ("revised" if args.check_workbook else "archived")
    initialize(populations, weekly_update)
    point = [estimate(i, 0) for i in range(2)]
    # Permanent regression check against the six independently reconstructed
    # reporting-date values from the full ordinary-mode joint-fit workbook.
    reference = [[1.00906808859867, 1.0317663863960909, 1.0224943173348615],
                 [1.0503159510304514, 1.0574772450702619, 1.0068182283938316]]
    fixed = [estimate(i, 0, joint=False)[0] for i in range(2)]
    print("Joint-fit points:", [x[0] for x in point], flush=True)
    tasks = [(i, j) for j in range(1, args.draws+1) for i in range(2)]
    results = [[] for _ in SHEETS]
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(populations, weekly_update)) as pool:
        for n, (i, draw, value) in enumerate(pool.map(one_draw, tasks, chunksize=4), 1):
            results[i].append(value)
            if n % 200 == 0:
                print(f"{n}/{len(tasks)} enrollment draws finished", flush=True)
    if args.check_workbook:
        reference = workbook_points(args.check_workbook, populations)
    np.testing.assert_allclose([x[0] for x in point], reference, rtol=1e-10, atol=1e-12)
    print("All six original estimates match the workbook.", flush=True)
    summary = {}
    for i, sheet in enumerate(SHEETS):
        values = np.array([r[0] for r in results[i]])
        assert values.shape == (args.draws, 3) and np.isfinite(values).all()
        summary[sheet] = dict(joint=point[i][0], fixed=fixed[i], original_insufficient_signal=point[i][1],
                              people=int(populations[i][1].sum()), lower=np.quantile(values, .025, axis=0).tolist(),
                              upper=np.quantile(values, .975, axis=0).tolist(),
                              draws_with_insufficient_signal=sum(r[1]>0 for r in results[i]),
                              unsuccessful_stratum_fits=sum(r[2] for r in results[i]))
    report = dict(input_blob=INPUT_BLOB, seed=SEED, draws_per_enrollment=args.draws,
                  date=str(REPORT.date()), pairs=PAIRS, summary=summary)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

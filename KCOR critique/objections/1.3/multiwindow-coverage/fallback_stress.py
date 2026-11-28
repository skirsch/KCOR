"""Exploratory coverage check in a design where upper-bound fallbacks occur."""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import study

SCENARIO = "nonlinear_wave"
REPETITIONS = 50
DRAWS = 99


def initialize():
    study.POPULATION = 50_000
    study.WINDOWS = ((2, 5), (35, 52))
    study.SEED = 20261126
    study.initialize()


def main():
    values = []
    tasks = [(SCENARIO, i, DRAWS) for i in range(REPETITIONS)]
    with ProcessPoolExecutor(7, initializer=initialize) as pool:
        for n, (_, i, _, weights, estimates) in enumerate(pool.map(study.run_dataset, tasks), 1):
            assert i == len(values)
            values.append(estimates)
            if n % 10 == 0:
                print(f"{n}/{REPETITIONS} fallback-stress datasets", flush=True)
    values = np.asarray(values)
    report = study.summarize(values, 1.)
    for j, label in enumerate(("fixed", "joint")):
        lower, upper = np.quantile(values[:, 1:, j, 0], [.025, .975], axis=1)
        covered = (lower <= 1) & (upper >= 1)
        flags = values[:, 0, j, 3] > 0
        report[label]["coverage_among_flagged"] = [int(covered[flags].sum()), int(flags.sum())]
        report[label]["coverage_among_unflagged"] = [int(covered[~flags].sum()), int((~flags).sum())]
        report[label]["median_width"] = float(np.median(upper-lower))
        report[label]["maximum_width"] = float(np.max(upper-lower))
    np.savez_compressed(study.HERE / "fallback-stress-draws.npz", points=values)
    (study.HERE / "fallback-stress.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

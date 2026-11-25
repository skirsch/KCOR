"""Replay the documented unequal-frailty scenario with the selected revised fitter."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent.parent / "iterative review"
STUDY = HERE.parent / "1.3" / "objection3_corrected_validation"
sys.path.insert(0, str(STUDY))
import production_adapter as adapter
from simulation import bootstrap


def main():
    spec = importlib.util.spec_from_file_location("revised_kcor", REVIEW / "revised/code/KCOR.py")
    revised = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revised)
    adapter.K = revised
    design = json.loads((STUDY / "design.json").read_text())
    case = next(s for s in design["scenarios"] if s["id"] == "unequal_large")
    folder = STUDY / "final_2026-11-10" / case["id"]
    windows, target = design["common"]["windows"], design["common"]["target"]
    covered = accepted = 0
    max_point_error = 0.0
    files = sorted(folder.glob("????.npz"))
    assert len(files) == design["outer_repetitions"]
    for path in files:
        record = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as data:
            counts, draws = data["counts"], data["draws"]
        assert len(draws) == design["bootstrap_draws"]
        fit = adapter.estimate(counts, windows, target)
        np.testing.assert_allclose(fit["point"], record["point"], rtol=1e-9)
        assert fit["accepted"] == record["accepted"]
        accepted += int(fit["accepted"])
        max_point_error = max(max_point_error, abs(fit["point"] - record["point"]))
        if path.stem == "0000":
            rng = np.random.default_rng(np.random.SeedSequence([*record["seed_components"], 1]))
            replay = [adapter.estimate(bootstrap(counts, rng), windows, target)["point"]
                      for _ in draws]
            np.testing.assert_allclose(replay, draws, rtol=1e-9)
            print(f"Refitted all {len(draws)} draws of the first dataset.", flush=True)
        lo, hi = adapter.intervals(fit["point"], draws)["percentile"]
        np.testing.assert_allclose([lo, hi], record["intervals"]["percentile"], rtol=1e-12)
        covered += int(lo <= record["truth"] <= hi)
    print(f"Accepted {accepted}/{len(files)}. Percentile coverage {covered}/{len(files)}. "
          f"Maximum point-estimate difference {max_point_error:.3g}.")


if __name__ == "__main__":
    main()

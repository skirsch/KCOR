"""Cross-check the study adapter at the actual MC workbook entry point for both fits."""
import contextlib
import io
import gc
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import compare


def main():
    compare.initialize()
    a = compare.adapter
    a.configure(compare.windows)
    folder = compare.STUDY / "final_2026-11-10/unequal_large"
    record = json.loads((folder / "0000.json").read_text())
    with np.load(folder / "0000.npz") as archive:
        original = archive["counts"]
    rng = np.random.default_rng(np.random.SeedSequence([*record["seed_components"], 1]))
    samples = [original] + [compare.bootstrap(original, rng) for _ in range(5)]
    with tempfile.TemporaryDirectory(prefix="kcor-paired-fit-") as tmp:
        base = Path(tmp)
        input_path = base / "input.xlsx"
        pd.concat([a.workbook_frame(c, i) for i, c in enumerate(samples)]).to_excel(
            input_path, sheet_name="2021_24", index=False)
        for name, function in (("fixed", compare.production_fit), ("joint", compare.joint_fit)):
            a.K.fit_theta0_gompertz = function
            outputs = base / name
            outputs.mkdir()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                a.K.process_workbook(str(input_path), str(outputs / "output.xlsx"))
            frame = pd.read_excel(outputs / "output.xlsx", sheet_name="2021_24")
            selected = frame[(frame.YearOfBirth == 1950) & (frame.Dose_num == 1) & (frame.Dose_den == 0)]
            assert selected.mc_id.nunique() == len(samples)
            for i, counts in enumerate(samples):
                actual = selected[selected.mc_id == i].sort_values("Date").KCOR.to_numpy()
                expected = a.estimate(counts, compare.windows, compare.target, return_curves=True)["curve"]
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
            gc.collect()  # Release workbook-reader cycles before temporary cleanup on Windows.
            print(f"{name}: matched all {len(selected)} workbook rows.", flush=True)


if __name__ == "__main__":
    main()

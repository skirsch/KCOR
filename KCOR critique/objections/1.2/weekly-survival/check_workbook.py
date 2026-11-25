"""Check the trial through the ordinary workbook entry point."""
import io
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from replay import initialize, table


def populations_from_bytes(data):
    return [table.read_population(io.BytesIO(data), sheet) for sheet in table.SHEETS]


def write_workbook(source, destination):
    initialize(populations_from_bytes(source.read_bytes()))
    table.K.OVERRIDE_DOSE_PAIRS = list(table.PAIRS)
    table.K.fit_theta0_gompertz = table.compare.joint_fit
    table.K.process_workbook(str(source), str(destination))


def main():
    data = subprocess.check_output(["git", "cat-file", "blob", table.INPUT_BLOB])
    populations = populations_from_bytes(data)
    initialize(populations)
    expected = [table.estimate(i, 0)[0] for i in range(len(populations))]
    with tempfile.TemporaryDirectory(prefix="kcor-weekly-wave-") as folder:
        root = Path(folder)
        source, output = root / "input.xlsx", root / "trial.xlsx"
        source.write_bytes(data)
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), str(source), str(output)],
                       cwd=root, check=True)
        actual = table.workbook_points(output, populations)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    print("All six trial points matched the ordinary workbook.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        write_workbook(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        main()

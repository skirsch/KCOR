"""Check Table 9 in a frozen KCOR edition. No writes to the edition."""
import argparse
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import load_workbook


def raw_endpoint(sheet):
    rows = iter(sheet.values)
    header = next(rows)
    columns = [header.index(key) for key in ("ISOweekDied", "Dose", "Dead", "Alive")]
    totals = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        week, dose, dead, alive = (row[i] for i in columns)
        group = totals[week, dose]
        group[0] += dead
        group[1] += alive
    cumulative = [0.0, 0.0]
    ratios = []
    for week in sorted({week for week, _ in totals}):
        for dose in (0, 1):
            dead, alive = totals[week, dose]
            cumulative[dose] += -math.log1p(-dead / alive)
        ratios.append(cumulative[1] / cumulative[0])
    return ratios[80] / ratios[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edition", type=Path)
    parser.add_argument("--replay", action="store_true", help="also run fitter and plotter")
    args = parser.parse_args()
    root = args.edition.resolve()
    control = root / "test/positive_control"
    source = control / "data/KCOR_positive_control.xlsx"
    paper = (root / "documentation/preprint/paper.md").read_text(encoding="utf-8")
    table = paper.split("{#tbl:pos_control_summary}", 1)[1].split("\n\n", 2)[1]
    published = {}
    for row in table.splitlines():
        fields = [field.strip() for field in row.strip("|").split("|")]
        if fields[0] in ("Harm", "Benefit"):
            published[fields[0].lower()] = fields[-1]
    assert published.keys() == {"harm", "benefit"}
    workbook = load_workbook(source, read_only=True, data_only=True)
    try:
        for scenario, endpoint in published.items():
            sheet, = [name for name in workbook.sheetnames if name.endswith("_" + scenario)]
            value = raw_endpoint(workbook[sheet])
            print(f"{sheet}  {value:.12f}  Table 9 {endpoint}")
            assert f"{value:.3f}" == endpoint
    finally:
        workbook.close()
    if args.replay:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "KCOR_processed_positive_control.xlsx"
            env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLCONFIGDIR=tmp)
            options = dict(cwd=tmp, env=env, capture_output=True, encoding="utf-8")
            fit = subprocess.run([sys.executable, "-B", str(root / "code/KCOR.py"),
                                  str(source), str(output), "Positive Control",
                                  "KCOR_summary_positive_control.log"], **options)
            print(f"Fitter exit {fit.returncode}, processed workbook exists {output.exists()}")
            print("\n".join(fit.stderr.splitlines()[-4:]))
            if fit.returncode != 1 or "too many values to unpack" not in fit.stderr:
                raise RuntimeError("Unexpected fitter outcome. Inspect the output above.")
            plot = subprocess.run([sys.executable, "-B", str(control / "code/plot_positive_control.py"),
                                   "--input", str(source), "--output", str(Path(tmp) / "raw.png")],
                                  **options, check=True)
            print(plot.stdout.strip())
            assert "Computing KCOR from raw data" in plot.stdout


if __name__ == "__main__":
    main()

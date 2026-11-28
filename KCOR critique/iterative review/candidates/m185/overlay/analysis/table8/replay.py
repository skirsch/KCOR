"""Regenerate Table 8 with the revised weekly update and verify its points against the workbook."""
import argparse
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STUDY = HERE
sys.path.insert(0, str(STUDY))
import table8_bootstrap as study


def write_workbook(source, workbook):
    os.chdir(source.parent)
    os.environ["MONTE_CARLO"] = "0"
    os.environ["KCOR_BOOTSTRAP_LAYOUT"] = "table8"
    os.environ["KCOR_BOOTSTRAP_DRAWS"] = "0"  # this runner performs its own matched bootstrap below
    study.compare.initialize("revised")
    module = study.compare.adapter.K
    module.OVERRIDE_DOSE_PAIRS = list(study.PAIRS)
    module.fit_theta0_gompertz = study.compare.joint_fit
    module.process_workbook(str(source), str(workbook))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=999)
    parser.add_argument("--output", type=Path, default=HERE / "table8_bootstrap.json")
    args = parser.parse_args()
    output = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="kcor-weekly-table8-") as scratch:
        source = Path(scratch) / "input.xlsx"
        workbook = Path(scratch) / "joint.xlsx"
        source.write_bytes((HERE/"input.xlsx").read_bytes())
        # The workbook writer can hold Excel handles until its process exits.
        writer = get_context("spawn").Process(target=write_workbook, args=(source, workbook))
        writer.start()
        writer.join()
        if writer.exitcode:
            raise RuntimeError(f"Workbook generation exited with status {writer.exitcode}")
        subprocess.run([sys.executable, str(STUDY / "table8_bootstrap.py"),
                        "--check-workbook", str(workbook), "--draws", str(args.draws),
                        "--output", str(output)], cwd=scratch, check=True)


if __name__ == "__main__":
    main()

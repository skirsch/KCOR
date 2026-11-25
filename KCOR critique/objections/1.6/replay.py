"""Regenerate Table 8 with the revised weekly update and verify its points against the workbook."""
import argparse
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "1.3"
sys.path.insert(0, str(STUDY))
import table8_bootstrap as study


def write_workbook(source, workbook, weekly_update):
    os.chdir(source.parent)
    os.environ["MONTE_CARLO"] = "0"
    study.compare.initialize(weekly_update)
    module = study.compare.adapter.K
    module.OVERRIDE_DOSE_PAIRS = list(study.PAIRS)
    module.fit_theta0_gompertz = study.compare.joint_fit
    module.process_workbook(str(source), str(workbook))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=999)
    parser.add_argument("--output", type=Path)
    edition = parser.add_mutually_exclusive_group()
    rounds = {"iteration-2": "2.2", "gap-boundary": "2.3", "wave-entry": "2.4"}
    for mode in rounds:
        edition.add_argument("--" + mode, dest="edition", action="store_const", const=mode)
    args = parser.parse_args()
    mode = args.edition or "revised"
    output_dir = HERE.parent / rounds[mode] if args.edition else HERE
    output = (args.output or output_dir / "table8_bootstrap.json").resolve()
    extra_args = ["--" + mode] if args.edition else []
    with tempfile.TemporaryDirectory(prefix="kcor-weekly-table8-") as scratch:
        source = Path(scratch) / "input.xlsx"
        workbook = Path(scratch) / "joint.xlsx"
        source.write_bytes(subprocess.check_output(["git", "cat-file", "blob", study.INPUT_BLOB], cwd=HERE))
        # The workbook writer can hold Excel handles until its process exits.
        writer = get_context("spawn").Process(target=write_workbook, args=(source, workbook, mode))
        writer.start()
        writer.join()
        if writer.exitcode:
            raise RuntimeError(f"Workbook generation exited with status {writer.exitcode}")
        subprocess.run([sys.executable, str(STUDY / "table8_bootstrap.py"),
                        "--check-workbook", str(workbook), "--draws", str(args.draws),
                        "--output", str(output), *extra_args], cwd=scratch, check=True)


if __name__ == "__main__":
    main()

"""Rebuild the checkpoint PDFs in a separate output directory."""
import argparse
import hashlib
import json
from pathlib import Path

import build_pdfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    output = parser.parse_args().output_directory.resolve()
    here = Path(__file__).resolve().parent
    if here == output or here in output.parents:
        parser.error("Choose an output directory outside the saved checkpoint.")
    for name, digest in json.loads((here / "checkpoint.json").read_text()).items():
        if hashlib.sha256((here / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checkpoint digest mismatch for {name}")
    build_pdfs.PREPRINT = here.parents[3] / "documentation/preprint"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("clean", "redline"):
        build_pdfs.build(name == "redline", output / f"KCOR-{name}.pdf")

"""
Aggregate per-variant analysis metrics into a single falsification summary CSV.

Expected layout:
  <base>/<variant_name>/analysis_metrics.csv
Writes:
  <out>
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base directory containing variant subdirs")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)

    paths = sorted(glob.glob(str(base / "*" / "analysis_metrics.csv")))
    if not paths:
        raise SystemExit(f"ERROR: No analysis_metrics.csv files found under {base}")

    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df.insert(0, "variant_dir", str(Path(p).parent))
        dfs.append(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
    print(f"Wrote {out} rows={sum(len(d) for d in dfs)} variants={len(dfs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


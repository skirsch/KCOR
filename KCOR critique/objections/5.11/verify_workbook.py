"""Check the ordinary report when a cohort exhausts its risk set."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("candidate", type=Path)
args = parser.parse_args()
os.environ.update(DATASET="Czech", KCOR_NPH_MODE="identified_only",
                  KCOR_BOOTSTRAP_DRAWS="0", SA_DOSE_PAIRS="1,0", MONTE_CARLO="0")
sys.path.insert(0, str(args.candidate / "code"))
import KCOR as K

K.OVERRIDE_DOSE_PAIRS = [(1, 0)]
original = K.build_kcor_rows
saved = {}

def capture(df, *args, **kwargs):
    saved["df"] = df.copy()
    return original(df, *args, **kwargs)

K.build_kcor_rows = capture
source = Path(__file__).resolve().parent.parent / "5.7/wave-reporting-input.xlsx"
df = pd.read_excel(source)
mask = (df.Dose == 1) & (df.DateDied == df.DateDied.max())
df.loc[mask, "Dead"] = df.loc[mask, "Alive"]
with tempfile.TemporaryDirectory() as directory:
    p = Path(directory)
    df.to_excel(p / "input.xlsx", sheet_name="2021_24", index=False)
    out = K.process_workbook(str(p / "input.xlsx"), str(p / "out.xlsx"),
                             str(p / "run.log"), return_rows=True)
frame = saved["df"]
mask = (frame.Dose == 1) & (frame.DateDied == frame.DateDied.max())
assert frame.loc[mask, "numerical_range_unavailable"].all()
assert frame.loc[mask, "CH"].isna().all()
rows = out[(out.Dose_num == 1) & (out.Dose_den == 0) & (out.Date == out.Date.max())]
rows = rows[rows.YearOfBirth.isin([1950, -2])]
assert set(rows.YearOfBirth) == {1950, -2}
assert rows[["KCOR", "KCOR_unanchored"]].isna().all().all()
print(rows[["YearOfBirth", "KCOR", "KCOR_unanchored"]].to_string(index=False))

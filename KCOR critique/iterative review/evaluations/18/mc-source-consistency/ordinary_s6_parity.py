"""Reconstruct the paper's S6 worked example; compare complete ordinary rows.

Run with a candidate root and scratch pickle path. Never output to candidate.
"""
from pathlib import Path
import os, sys, tempfile, contextlib, io
import pandas as pd
root=Path(sys.argv[1]).resolve(); dst=Path(sys.argv[2]).resolve()
os.environ.update(DATASET="Czech", KCOR_NPH_MODE="identified_only", KCOR_BOOTSTRAP_DRAWS="0", KCOR_AGGREGATE_SCOPE="partial", SA_DOSE_PAIRS="2,0", MONTE_CARLO="0", KCOR_BOOTSTRAP_LAYOUT="disjoint", KCOR_RISK_HISTORY_POLICY="death_only")
sys.path.insert(0,str(root/"code"))
import KCOR as K
source=root/"data/Czech/KCOR_CMR.xlsx"
raw=pd.read_excel(source,sheet_name="2021_24",usecols=["DateDied","ISOweekDied","YearOfBirth","Dose","Alive","Dead"])
week=raw.ISOweekDied.astype(str).str.replace("-","",regex=False).str[:6].astype(int)
raw=raw[week.between(202124,202416)&raw.Dose.isin([0,2])].copy()
known=raw[raw.YearOfBirth.gt(0)].copy(); known["YearOfBirth"]=(known.YearOfBirth-5)//10*10+5
known=known[known.YearOfBirth.between(1925,1975)]
frame=pd.concat([known,raw[raw.YearOfBirth.eq(-2)]]).groupby(["YearOfBirth","Dose","DateDied","ISOweekDied"],as_index=False)[["Alive","Dead"]].sum()
frame["Sex"]="O"; K.AGE_RANGE=5; K.YEAR_RANGE=None; K.OVERRIDE_DOSE_PAIRS=[(2,0)]; K.ENROLLMENT_DATES=["2021_24"]
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
    p=Path(t); frame.to_excel(p/"input.xlsx",sheet_name="2021_24",index=False)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        result=K.process_workbook(str(p/"input.xlsx"),str(p/"output.xlsx"),str(p/"run.log"),return_rows=True)
assert result.shape==(1192,46),result.shape
result.to_pickle(dst)
print(root.name,result.shape,result[result.YearOfBirth.eq(-2)].sort_values("Date").iloc[-1].KCOR)

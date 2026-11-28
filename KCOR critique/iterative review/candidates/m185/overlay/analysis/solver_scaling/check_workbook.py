from pathlib import Path
import sys,json,pandas as pd,numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"test"))
from test_round7_workbook import run
points={}
for mode in ("identified_only","forced_alpha"):
    source=ROOT/("analysis/finite_reference/table8-primary" if mode=="identified_only" else "analysis/round11/forced-alpha-smoke")/(mode+"-draws.npz")
    expected=np.load(source)["points"][:,0]
    points[mode]={}
    for i,sheet in enumerate(("2021_24","2022_06")):
        f=pd.read_excel(ROOT/"analysis/table8/input.xlsx",sheet_name=sheet)
        rows=run(sheet,f,mode=mode,layout="table8")
        sys.path.insert(0,str(ROOT/"analysis/table8"))
        import table8_bootstrap as T
        pooled=rows[rows.YearOfBirth.eq(0)]
        all_dates=pd.to_datetime(rows[rows.YearOfBirth.eq(1940)].Date).unique()
        for n,d in [(1,0),(2,0),(2,1)]:
            pp=pooled[(pooled.Dose_num==n)&(pooled.Dose_den==d)]
            assert len(pp)==len(all_dates),(sheet,mode,n,d,len(pp),len(all_dates))
            assert set(pd.to_datetime(pp.Date))==set(all_dates)
        exported=pooled[pd.to_datetime(pooled.Date)==pd.Timestamp("2022-12-26")]
        actual=[float(exported[(exported.Dose_num==n)&(exported.Dose_den==d)].KCOR.iloc[0]) for n,d in [(1,0),(2,0),(2,1)]]
        assert np.max(np.abs(np.array(actual)-expected[i]))<1e-8,(sheet,mode,actual,expected[i])
        maxdiff=float(np.max(np.abs(np.array(actual)-expected[i])))
        assert maxdiff<1e-8
        points[mode][sheet]=dict(points=actual,exported_points=actual,pooled_dates=len(all_dates),max_difference_from_full_replay=maxdiff)
        print(mode,sheet,points[mode][sheet],flush=True)
(HERE/"workbook-verification.json").write_text(json.dumps(points,indent=2))

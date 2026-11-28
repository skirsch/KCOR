from pathlib import Path
import sys,json,pandas as pd,numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"test"))
from test_round7_workbook import run
points={}
for mode in ("identified_only","forced_alpha"):
    source=HERE/(mode+"-draws.npz")
    if not source.exists(): source=HERE/(mode+"-partial.npz")
    expected=np.load(source)["points"][:,0]
    points[mode]={}
    for i,sheet in enumerate(("2021_24","2022_06")):
        f=pd.read_excel(ROOT/"analysis/table8/input.xlsx",sheet_name=sheet)
        rows=run(sheet,f,mode=mode,layout="table8")
        sys.path.insert(0,str(ROOT/"analysis/table8"))
        import table8_bootstrap as T
        _,cells=T.read_population(ROOT/"analysis/table8/input.xlsx",sheet)
        strata=cells.copy();strata[2]=cells.sum(axis=0)
        alive=np.flip(np.cumsum(np.flip(strata,axis=2),axis=2),axis=2)
        early=strata[:,:,:4]/alive[:,:,:4]
        weights=np.nanmean(np.where(early>0,early,np.nan),axis=(1,2))*alive[:,:,:4].sum(axis=(1,2))
        p=rows[(rows.YearOfBirth.isin([1940,1950,1960]))&(pd.to_datetime(rows.Date)==pd.Timestamp("2022-12-26"))]
        values=[float(np.exp(np.average(np.log(p[(p.Dose_num==n)&(p.Dose_den==d)].set_index("YearOfBirth").loc[[1940,1950,1960],"KCOR"]),weights=weights))) for n,d in [(1,0),(2,0),(2,1)]]
        maxdiff=float(np.max(np.abs(np.array(values)-expected[i])))
        assert maxdiff<1e-8 and p.normalization_status.eq("ok").all()
        points[mode][sheet]=dict(points=values,max_difference_from_full_replay=maxdiff)
        print(mode,sheet,points[mode][sheet],flush=True)
(HERE/"workbook-verification.json").write_text(json.dumps(points,indent=2))

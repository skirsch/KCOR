import os,sys,tempfile,io
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="identified_only",KCOR_BOOTSTRAP_DRAWS="0",MONTE_CARLO="0",KCOR_BOOTSTRAP_LAYOUT="disjoint")
import KCOR as K
def input_frame(sheet,missing=None):
    start=pd.Timestamp.fromisocalendar(*map(int,sheet.split("_")),1)
    dates=pd.date_range(start,periods=20,freq="7D")
    frames=[]
    for age in (1940,1950):
        for dose in (0,1):
            dead=np.arange(20)%3+20+dose*4+(age-1940)
            alive=20000-np.r_[0,np.cumsum(dead[:-1])]
            f=pd.DataFrame(dict(YearOfBirth=age,Dose=dose,Sex="F",ISOweekDied=dates.strftime("%G-%V"),DateDied=dates,Alive=alive,Dead=dead))
            if missing is not None and age==1940:f=f.drop(missing)
            frames.append(f)
    return pd.concat(frames,ignore_index=True)
def run(sheet,frame,mode="identified_only",layout="disjoint"):
    os.environ["KCOR_NPH_MODE"]=mode;os.environ["KCOR_BOOTSTRAP_LAYOUT"]=layout
    K.ENROLLMENT_DATES=[sheet];K.OVERRIDE_DOSE_PAIRS=[(1,0)] if layout=="disjoint" else [(1,0),(2,0),(2,1)]
    K.OVERRIDE_YOBS=None
    with tempfile.TemporaryDirectory() as t:
        src=Path(t)/"input.xlsx"
        with pd.ExcelWriter(src) as w:frame.to_excel(w,sheet_name=sheet,index=False)
        try:
            with redirect_stdout(io.StringIO()) as log,redirect_stderr(io.StringIO()):
                out=K.process_workbook(str(src),str(Path(t)/"out.xlsx"),return_rows=True)
        except Exception:
            print(log.getvalue()[-7000:]);raise
    return out
def main():
    for sheet,date in [("2025_24","2025-07-14"),("2020_24","2020-07-13"),("2021_24","2021-07-19")]:
        out=run(sheet,input_frame(sheet))
        row=out[(out.YearOfBirth==1940)&(out.Dose_num==1)&(out.Dose_den==0)]
        assert len(row)==20,(sheet,len(row))
        assert str(pd.to_datetime(row.anchor_date).dropna().iloc[0].date())==date,(sheet,row.anchor_date.unique())
        print(sheet,"anchor",date,"rows",len(row))
    out=run("2021_24",input_frame("2021_24",5))
    for age in [1940,1950,0,-2]:
        x=out[out.YearOfBirth==age]
        print("missing",age,"status",x.normalization_status.unique(),"last",x.KCOR.iloc[-1] if len(x) else None)
    assert out[out.YearOfBirth==1940].KCOR.isna().all()
    assert out[out.YearOfBirth==1940].anchor_status.eq("missing_reference_week").all()
    assert out[out.YearOfBirth==1950].KCOR.notna().any()
    assert out[out.YearOfBirth==0].KCOR.isna().all()
    assert out[out.YearOfBirth==-2].KCOR.isna().all()
    f=pd.read_excel(ROOT/"analysis/table8/input.xlsx",sheet_name="2021_24")
    captured={};build=K.build_kcor_rows
    def capture(frame,*a,**kw):
        captured["frame"]=frame.copy()
        return build(frame,*a,**kw)
    K.build_kcor_rows=capture
    try:clean=run("2021_24",f,layout="table8")
    finally:K.build_kcor_rows=build
    frame=captured["frame"]
    agg=frame[frame.YearOfBirth==-2].sort_values(["Dose","DateDied"])
    inc=frame[frame.YearOfBirth==1960].sort_values(["Dose","DateDied"])
    np.testing.assert_array_equal(agg[["Alive","Dead"]],inc[["Alive","Dead"]])
    a=clean[clean.YearOfBirth==1960].sort_values(["Dose_num","Dose_den","Date"])
    b=clean[clean.YearOfBirth==-2].sort_values(["Dose_num","Dose_den","Date"])
    np.testing.assert_allclose(a.KCOR,b.KCOR,equal_nan=True,rtol=1e-10)
    print("overlap direct aggregate equals inclusive group",len(b))
    old=K.apply_nph_correction_post_inversion
    calls=0
    def fail_once(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:raise ArithmeticError("controlled numerical failure")
        return old(*args,**kwargs)
    K.apply_nph_correction_post_inversion=fail_once
    try:bad=run("2021_24",f,mode="forced_alpha",layout="table8")
    finally:K.apply_nph_correction_post_inversion=old
    print("failure",bad.groupby("YearOfBirth").normalization_status.unique().to_dict())
    flagged=bad[bad.normalization_status.eq("normalization_failed")]
    assert len(flagged)>0 and flagged.KCOR.isna().all() and flagged.abnormal_fit.eq(True).all()
    from bootstrap_reporting import flagged as boot_flagged
    assert boot_flagged(flagged).all()
    assert bad[bad.normalization_status.eq("ok")].KCOR.notna().any()
    assert bad[(bad.YearOfBirth==0)&(bad.Dose_den==0)].KCOR.isna().all()
    print("PASS failure flags and unaffected cohorts")
if __name__=="__main__":main()

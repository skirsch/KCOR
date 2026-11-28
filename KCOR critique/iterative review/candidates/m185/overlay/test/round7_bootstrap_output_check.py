from pathlib import Path
import os,sys,tempfile,io
from contextlib import redirect_stdout,redirect_stderr
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"test"))
from test_round7_workbook import input_frame,K
def main():
    os.environ.update(KCOR_BOOTSTRAP_DRAWS="2",KCOR_BOOTSTRAP_WORKERS="2",KCOR_NPH_MODE="forced_alpha",KCOR_BOOTSTRAP_LAYOUT="disjoint")
    K.ENROLLMENT_DATES=["2021_24"];K.OVERRIDE_DOSE_PAIRS=[(1,0)];K.OVERRIDE_YOBS=None
    old=K.apply_nph_correction_post_inversion
    calls=0
    def fail_once(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==1:raise ArithmeticError("controlled numerical failure")
        return old(*a,**kw)
    K.apply_nph_correction_post_inversion=fail_once
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            src=Path(t)/"input.xlsx";dst=Path(t)/"out.xlsx"
            input_frame("2021_24").to_excel(src,sheet_name="2021_24",index=False)
            with redirect_stdout(io.StringIO()) as log,redirect_stderr(io.StringIO()):
                K.process_workbook(str(src),str(dst))
            out=pd.read_excel(dst,sheet_name="dose_pairs")
            assert "normalization_status" in out
            fail=out[out.normalization_status.eq("normalization_failed")]
            assert len(fail)>0 and fail.KCOR.isna().all()
            assert fail.CI_point_flagged.eq(True).all()
            assert fail.CI_lower.isna().all() and fail.CI_upper.isna().all()
            assert out[(out.normalization_status=="ok") & out.KCOR.notna()].shape[0]>0
            print("PASS written workbook flags, unavailable point/interval, successful cohort")
    finally:K.apply_nph_correction_post_inversion=old
if __name__=="__main__":main()

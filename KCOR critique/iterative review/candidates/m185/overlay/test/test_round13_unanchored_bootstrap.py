from pathlib import Path
import os,sys,tempfile,io
from contextlib import redirect_stdout,redirect_stderr
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'test'),str(ROOT/'code')]
from test_round7_workbook import K,input_frame
if __name__=='__main__':
 s='2023_01';f=input_frame(s);ix=f[(f.YearOfBirth==1940)&(f.Dose==1)].index;a=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
 f.loc[ix[a],'Dead']=f.loc[ix[a],'Alive'];f.loc[ix[a+1:],['Alive','Dead']]=0
 os.environ.update(KCOR_BOOTSTRAP_DRAWS='3',KCOR_BOOTSTRAP_WORKERS='2',KCOR_BOOTSTRAP_LAYOUT='disjoint');K.ENROLLMENT_DATES=[s];K.OVERRIDE_DOSE_PAIRS=[(1,0)]
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
  src=Path(tmp)/'in.xlsx';dst=Path(tmp)/'out.xlsx';f.to_excel(src,sheet_name=s,index=False)
  with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):K.process_workbook(str(src),str(dst))
  q=pd.read_excel(dst,sheet_name='dose_pairs');q=q[q.YearOfBirth.eq(1940)].sort_values('Date')
  assert q.KCOR.isna().all() and q.iloc[a-1].CI_unanchored_valid_draws==3 and np.isfinite(q.iloc[a-1].CI_unanchored_lower) and q.iloc[a:].CI_unanchored_lower.isna().all()
  print(q.iloc[a-1][['KCOR_unanchored','CI_unanchored_lower','CI_unanchored_upper','CI_unanchored_valid_draws']].to_dict())

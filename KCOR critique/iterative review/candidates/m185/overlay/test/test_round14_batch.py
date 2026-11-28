from pathlib import Path
import os,sys,tempfile,io
from contextlib import redirect_stdout,redirect_stderr
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'test'),str(ROOT/'code')]
from test_round7_workbook import K,input_frame
from bootstrap_reporting import histories
from fixed_cohort_history import risk_history_policy

def main():
 s='2022_26'; dates=pd.date_range('2022-06-27',periods=85,freq='7D'); fs=[]
 for age in (1940,1950):
  for dose in (0,2):
   dead=np.full(85,15 if dose==0 else 16)
   if age==1940 and dose==2:
    dead[60]=10000-16*60;dead[61:]=0
   fs.append(pd.DataFrame(dict(YearOfBirth=age,Dose=dose,Sex='F',ISOweekDied=dates.strftime('%G-%V'),DateDied=dates,Alive=10000-np.r_[0,np.cumsum(dead[:-1])],Dead=dead)))
 f=pd.concat(fs,ignore_index=True)
 histories(f,s,'disjoint'); redundant=f.assign(PT=f.Alive)
 histories(redundant,s,'disjoint')
 try:histories(f.assign(PT=f.Alive-.5),s,'disjoint')
 except ValueError:pass
 else:raise AssertionError('exposure accepted')
 try:risk_history_policy('Disjoint')
 except ValueError:pass
 else:raise AssertionError('unknown layout accepted')
 assert K._finite_hazard_ratio(0,1e-15)==0 and np.isfinite(K._finite_hazard_ratio(.23,7.432377e-14))
 os.environ.update(KCOR_BOOTSTRAP_DRAWS='3',KCOR_BOOTSTRAP_WORKERS='2',KCOR_BOOTSTRAP_LAYOUT='disjoint');K.ENROLLMENT_DATES=[s];K.OVERRIDE_DOSE_PAIRS=[(2,0)];K.OVERRIDE_YOBS=None
 results=[]
 for ref in (59,60):
  K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE=ref;K.KCOR_NORMALIZATION_WEEKS=ref-K.DYNAMIC_HVE_SKIP_WEEKS+1
  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
   src=Path(tmp)/'in.xlsx';dst=Path(tmp)/'out.xlsx';f.to_excel(src,sheet_name=s,index=False)
   with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):K.process_workbook(str(src),str(dst))
   q=pd.read_excel(dst,sheet_name='dose_pairs');q=q[q.YearOfBirth.eq(1940)].sort_values('Date')
   r=q.iloc[59];cols=['KCOR','KCOR_unanchored','CI_unanchored_lower','CI_unanchored_upper','CI_unanchored_valid_draws','CI_unanchored_point_flagged','CI_unanchored_flagged_draws','CI_unanchored_method','CI_point_flagged','CI_flagged_draws'];print(ref,r[cols].to_dict());results.append(r)
 a,b=results;cols=['KCOR_unanchored','CI_unanchored_lower','CI_unanchored_upper','CI_unanchored_valid_draws','CI_unanchored_point_flagged','CI_unanchored_flagged_draws','CI_unanchored_method']
 assert a[cols].equals(b[cols]),(a[cols],b[cols])
 assert np.isnan(b.KCOR) and b.CI_point_flagged and not b.CI_unanchored_point_flagged and b.CI_unanchored_flagged_draws==0
 print('PASS matching raw result and independent flags')

def ancillary_reports():
 s='2023_01';f=input_frame(s);idx=f[(f.YearOfBirth==1940)&(f.Dose==1)].index
 f.loc[idx[8],'Dead']=f.loc[idx[8],'Alive'];f.loc[idx[9:],['Alive','Dead']]=0
 K.NEGATIVE_CONTROL_MODE=1;K.ENROLLMENT_DATES=[s];K.OVERRIDE_DOSE_PAIRS=[(1,0)]
 K.KCOR_NORMALIZATION_WEEKS=4;K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE=K.DYNAMIC_HVE_SKIP_WEEKS+3
 original_date=K.get_reporting_date
 K.get_reporting_date=lambda _:str(f.loc[idx[10],'DateDied'].date())
 os.environ['KCOR_BOOTSTRAP_DRAWS']='0'
 try:
  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
   p=Path(t);f.to_excel(p/'in.xlsx',sheet_name=s,index=False)
   with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):K.process_workbook(str(p/'in.xlsx'),str(p/'out.xlsx'))
   r=pd.read_excel(p/'negative_control_test_summary.xlsx');r=r[(r.YoB1==1940)&(r.Dose==1)].iloc[0]
   assert r.ReportDate==f.loc[idx[10],'DateDied'] and r.endpoint_status=='unavailable' and pd.isna(r.KCOR),r
   print('PASS exact optional negative endpoint')
 finally:K.NEGATIVE_CONTROL_MODE=0;K.get_reporting_date=original_date
 s='2021_24';ds=pd.date_range('2021-06-14',periods=160,freq='7D');rows=[]
 for age in (1940,1950):
  for dose in (0,1):
   alive=10**15
   for i,date in enumerate(ds):
    dead=1 if dose==1 and i<6 else int(alive*(.01 if dose==0 else .003))
    rows.append(dict(YearOfBirth=age,Dose=dose,Sex='F',ISOweekDied=date.strftime('%G-%V'),DateDied=date,Alive=alive,Dead=dead));alive-=dead
 f=pd.DataFrame(rows);K.ENROLLMENT_DATES=[s];K.OVERRIDE_DOSE_PAIRS=[(1,0)]
 K.KCOR6_QUIET_START_ISO='2023-01';K.KCOR6_QUIET_END_ISO='2023-52'
 os.environ.update(KCOR_NPH_MODE='off',KCOR_BOOTSTRAP_DRAWS='0',SENSITIVITY_ANALYSIS='1',SA_BASELINE_WEEKS='4',SA_QUIET_START_OFFSETS='0')
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
  p=Path(t);f.to_excel(p/'in.xlsx',sheet_name=s,index=False)
  with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):K.process_workbook(str(p/'in.xlsx'),str(p/'out.xlsx'))
  r=pd.read_csv(p/'KCOR_SA_cells.csv').iloc[0]
  assert r.status=='ok' and np.isfinite(r.kcor) and r.kcor>1e10,r
  print('PASS finite sensitivity endpoint below old reference floor')
if __name__=='__main__':
 main();ancillary_reports()


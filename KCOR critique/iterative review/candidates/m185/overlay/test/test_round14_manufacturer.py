from pathlib import Path
import os, sys, tempfile, io
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'test'),str(ROOT/'code')]
from test_round7_workbook import K
from manufacturer_history import build_history,reconcile

def main():
 s='2021_24';start=pd.Timestamp('2021-06-14')
 ds=pd.date_range(start,periods=85,freq='7D')
 people=pd.DataFrame(dict(dose_group=[2,2,3,2],Dose2_MFG=['M','P','M','X'],
   YearOfBirth=[1940]*4,Sex=['F']*4,DCCI=[0]*4,
   DateOfDeath=[pd.NaT,ds[1],ds[1],pd.NaT],WeekOfDeath=[None,'2021-25','2021-25',None]))
 h=build_history(people,start,ds.strftime('%G-%V').tolist(),2)
 assert h[h.DateDied.eq('2021-06-14')].Alive.sum()==3 and h.Dead.sum()==1 and h[h.MFG.eq('M')].Dead.sum()==0
 ordinary=h.groupby(['DateDied','ISOweekDied','YearOfBirth','Sex','DCCI','Dose'],as_index=False)[['Alive','Dead']].sum()
 reconcile(h,ordinary,start,2)
 bad=h.copy();bad.loc[bad.index[0],'Alive']+=1
 try:reconcile(bad,ordinary,start,2)
 except ValueError:pass
 else:raise AssertionError('mismatched archive accepted')
 print('PASS exact dose split and reconciliation')
 rows=[]
 for age in (1940,1950):
  for mfg,weekly in [('P',15),('M',16),('O',10)]:
   dead=np.full(len(ds),weekly)
   rows.append(pd.DataFrame(dict(DateDied=ds,ISOweekDied=ds.strftime('%G-%V'),YearOfBirth=age,Sex='F',DCCI=0,Dose=2,MFG=mfg,Alive=20000-np.r_[0,np.cumsum(dead[:-1])],Dead=dead)))
 h=pd.concat(rows,ignore_index=True)
 ordinary=h.groupby(['DateDied','ISOweekDied','YearOfBirth','Sex','DCCI','Dose'],as_index=False)[['Alive','Dead']].sum()
 # Extra ordinary comparison group; source M/P should still reconcile only dose 2.
 other=ordinary.copy();other['Dose']=0
 ordinary=pd.concat([ordinary,other],ignore_index=True)
 os.environ.update(KCOR_BOOTSTRAP_DRAWS='3',KCOR_BOOTSTRAP_WORKERS='2',KCOR_BOOTSTRAP_LAYOUT='disjoint')
 K.ENROLLMENT_DATES=[s];K.OVERRIDE_DOSE_PAIRS=[(2,0)]
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
  src=Path(tmp)/'input.xlsx';dst=Path(tmp)/'output.xlsx'
  earlier=h[h.DateDied.eq(start)].copy();earlier["DateDied"]=start-pd.Timedelta(weeks=1);earlier["ISOweekDied"]=earlier.DateDied.dt.strftime("%G-%V")
  with pd.ExcelWriter(src) as w:
   ordinary.to_excel(w,sheet_name=s,index=False);pd.concat([earlier,h]).to_excel(w,sheet_name=s+'_MFG_D2',index=False)
  with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
   K.process_workbook(str(src),str(dst))
  out=pd.read_excel(dst,sheet_name='MFG_MP_2021_24_D2_decades')
  q=out[out.YearOfBirth.eq(1940)].sort_values('Date')
  assert len(q)==85 and pd.to_datetime(q.Date).min()==start
  assert q.CI_valid_draws.iloc[-1]==3 and np.isfinite(q.CI_lower.iloc[-1])
  assert q.Dose_num.eq(2).all() and q.Dose_den.eq(2).all()
  assert q.KCOR_MP.equals(q.KCOR)
  print('PASS integrated manufacturer workbook',q.iloc[-1][['KCOR_MP','CI_lower','CI_upper','CI_valid_draws','theta_num','theta_den']].to_dict())
  log=(Path(tmp)/'KCOR_summary.log').read_text(encoding='utf-8')
  report=q[pd.to_datetime(q.Date).eq(pd.Timestamp(K.get_reporting_date(s)))].iloc[0]
  expected=f'{report.KCOR_MP:.4f} [{report.CI_lower:.4f}, {report.CI_upper:.4f}]'
  assert expected in log,expected
  print('PASS log equals workbook at reporting date',expected)
if __name__=='__main__':main()

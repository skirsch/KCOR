"""Round 15: preserve distinction between full and available age populations."""
from pathlib import Path
import sys,os,tempfile,io
from contextlib import redirect_stdout,redirect_stderr
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'test'),str(ROOT/'code')]
from test_round14_aggregate import K,input_frame,total,ending
S='2023_01'
def written(f,scope):
 os.environ.update(KCOR_BOOTSTRAP_DRAWS='3',KCOR_BOOTSTRAP_WORKERS='2',KCOR_BOOTSTRAP_LAYOUT='disjoint',KCOR_AGGREGATE_SCOPE=scope)
 K.ENROLLMENT_DATES=[S];K.OVERRIDE_DOSE_PAIRS=[(1,0)];K.OVERRIDE_YOBS=None
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
  p=Path(tmp);src=p/'in.xlsx';dst=p/'out.xlsx';f.to_excel(src,sheet_name=S,index=False)
  with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):K.process_workbook(str(src),str(dst))
  out=pd.read_excel(dst,sheet_name='dose_pairs');by=pd.read_excel(dst,sheet_name='by_dose')
  log=(p/'KCOR_summary.log').read_text(encoding='utf-8')
  if scope=='partial' and not f.YearOfBirth.eq(-2).any():
   assert by[by.YearOfBirth.eq(-2)].population_label.eq('All available ages').all()
   assert 'All available ages' in log
  return out
def main():
 f=input_frame(S);f.loc[f.YearOfBirth.eq(1950),'YearOfBirth']=1960
 idx=f[f.YearOfBirth.eq(1960)&f.Dose.eq(1)].index;f.loc[idx[8:],'Dead']*=4;f.loc[idx,'Alive']=20000-np.r_[0,np.cumsum(f.loc[idx,'Dead'].to_numpy()[:-1])]
 older=f[f.YearOfBirth.eq(1940)];young=total(f[f.YearOfBirth.eq(1960)]).assign(YearOfBirth=-60)
 full=total(f);cols=['KCOR','CI_lower','CI_upper','CI_valid_draws']
 a=written(f,'exhaustive');b=written(pd.concat([older,young]),'partial')
 ar=ending(a,-2);br=ending(b,-2);old=ending(b,1940)
 assert ar.population_label=='All Ages' and br.population_label=='All available ages'
 np.testing.assert_allclose(br[cols].to_numpy(float),old[cols].to_numpy(float),rtol=1e-13)
 assert br.CI_valid_draws==3 and br.CI_population_scope_changed_draws==0 and abs(ar.KCOR-br.KCOR)>.5
 c=written(pd.concat([f,full[full.Dose.eq(1)]]),'exhaustive')
 cr=ending(c,-2);assert cr.population_label=='All Ages'
 np.testing.assert_allclose(ar[cols].to_numpy(float),cr[cols].to_numpy(float),rtol=1e-13)
 d=written(pd.concat([older,full]),'partial');dr=ending(d,-2)
 assert dr.population_label=='All Ages' and dr.KCOR==ar.KCOR and dr.CI_valid_draws==0 and pd.isna(dr.CI_lower)
 print('full',ar[cols].to_dict(),ar.population_label)
 print('partial',br[cols].to_dict(),br.population_label)
 print('exhaustive one supplied dose',cr[cols].to_dict(),cr.population_label)
 print('partial supplied full total',dr[cols].to_dict(),dr.population_label)
if __name__=='__main__':main()

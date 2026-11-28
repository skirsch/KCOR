"""Round 16.2: a Monte Carlo replicate is independent of its workbook batch."""
from pathlib import Path
import os,sys,tempfile,io
from contextlib import redirect_stdout,redirect_stderr
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'test'),str(ROOT/'code')]
from test_round7_workbook import K,input_frame
S='2023_01'
def run(f,write=False):
 os.environ.update(MC_ENROLLMENT_DATE=S,MONTE_CARLO='1',KCOR_BOOTSTRAP_DRAWS='0',KCOR_BOOTSTRAP_LAYOUT='disjoint',KCOR_AGGREGATE_SCOPE='exhaustive',KCOR_NPH_MODE='identified_only')
 K.MONTE_CARLO_MODE=True;K.MC_INCLUDE_AGGREGATES=True;K.MC_YOB_START=1940;K.MC_YOB_END=1950
 K.ENROLLMENT_DATES=[S];K.OVERRIDE_DOSE_PAIRS=[(1,0)];K.OVERRIDE_YOBS=None
 with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
  p=Path(tmp);src=p/'in.xlsx';f.to_excel(src,sheet_name=S,index=False)
  with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):o=K.process_workbook(str(src),str(p/'out.xlsx'),return_rows=not write)
  if write:o=pd.read_excel(p/'out.xlsx',sheet_name=S)
  weights={int(i):K._asmr_weights_for(S,int(i)).copy() for i in f.mc_id.unique()}
  return o,weights
def chosen(out,draw):
 return out[out.mc_id.eq(draw)&out.YearOfBirth.isin([0,-3])].sort_values(['YearOfBirth','Date']).reset_index(drop=True)
def main():
 f=input_frame(S);ix=f[f.YearOfBirth.eq(1940)&f.Dose.eq(1)].index
 f.loc[ix[8:],'Dead']*=3;f.loc[ix,'Alive']=20000-np.r_[0,np.cumsum(f.loc[ix,'Dead'].to_numpy()[:-1])]
 a=f.assign(mc_id=1);b=input_frame(S).assign(mc_id=2);b.loc[b.YearOfBirth.eq(1950),['Alive','Dead']]*=15
 alone,wa=run(a);second,wb=run(b);batch,w=run(pd.concat([a,b],ignore_index=True));reverse,wr=run(pd.concat([b,a],ignore_index=True).sample(frac=1,random_state=12))
 assert wa[1]==w[1]==wr[1] and wb[2]==w[2]==wr[2]
 assert wa[1]!=wb[2]
 for draw,individual in [(1,alone),(2,second)]:
  expected=chosen(individual,draw);assert set(expected.YearOfBirth)=={0}
  for actual in (chosen(batch,draw),chosen(reverse,draw)):
   pd.testing.assert_frame_equal(expected,actual,check_exact=True)
  print('replicate',draw,'weights',w[draw],'pooled endpoint',expected[expected.YearOfBirth.eq(0)].iloc[-1].KCOR,'interval unavailable (MC replicate point)')
 print('PASS standalone, batched, reordered and shuffled equality for reported pooled path')
 written_batch,ww=run(pd.concat([a,b],ignore_index=True),write=True)
 assert ww==w
 for draw in (1,2):
  p=chosen(batch,draw);q=chosen(written_batch,draw)
  np.testing.assert_allclose(p.KCOR,q.KCOR,rtol=1e-14,equal_nan=True)
  np.testing.assert_allclose(p.pool_retained_weight_fraction,q.pool_retained_weight_fraction,rtol=1e-14,equal_nan=True)
 # Controlled normalization failure: only the affected replicate and its
 # positive-weight contributor may propagate the flag.
 probe=pd.concat([a,b],ignore_index=True)
 probe['t']=(probe.DateDied-pd.Timestamp('2023-01-02')).dt.days//7
 probe['normalization_status']='ok'
 probe.loc[probe.mc_id.eq(1)&probe.YearOfBirth.eq(1940)&probe.Dose.eq(1),'normalization_status']='normalization_failed'
 flagged=K._flag_unavailable_normalization(batch,probe,S)
 assert chosen(flagged,1).normalization_status.eq('normalization_failed').all()
 assert chosen(flagged,2).normalization_status.eq('ok').all()
 original=K.ASMR_WEIGHTS_BY_SHEET[(S,1)]
 K.ASMR_WEIGHTS_BY_SHEET[(S,1)]={1940:0.,1950:1.}
 try:
  flagged=K._flag_unavailable_normalization(batch,probe,S)
  assert chosen(flagged,1).normalization_status.eq('ok').all()
 finally:K.ASMR_WEIGHTS_BY_SHEET[(S,1)]=original
 print('PASS written workbook and replicate-specific normalization-flag lookup')

if __name__=='__main__':main()

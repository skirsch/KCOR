"""9.5: unexplained losses are rejected by default; opt-in preserves the declared analysis."""
from pathlib import Path
import os,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'test'));sys.path.insert(0,str(ROOT/'code'))
from test_source_component_history import fixture
from test_round7_workbook import run
from fixed_cohort_history import invalid_components
from bootstrap_reporting import histories, draw_history

def main():
    os.environ.pop('KCOR_RISK_HISTORY_POLICY',None)
    frame=fixture();start=pd.Timestamp('2021-06-14');sheet='2021_24'
    assert not invalid_components(frame,start);histories(frame,sheet,'disjoint')
    altered=frame.copy();mask=altered.YearOfBirth.eq(1950)&altered.Dose.eq(1)&altered.DateDied.ge('2021-10-04')
    altered.loc[mask,'Alive']-=12000
    bad=invalid_components(altered,start);assert (1950,1) in bad;print(bad,flush=True)
    try:histories(altered,sheet,'disjoint')
    except ValueError:pass
    else:raise AssertionError('default bootstrap accepted unexplained loss')
    out=run(sheet,altered);assert out.risk_history_policy.eq('death_only').all()
    for age in (1950,0,-2):
        x=out[out.YearOfBirth.eq(age)&out.Dose_num.eq(1)];assert x.KCOR.isna().all() and x.normalization_status.eq('invalid_risk_history').all(),(age,x[['KCOR','normalization_status']].tail())
    x=out[out.YearOfBirth.eq(1940)&out.Dose_num.eq(1)];assert abs(x.KCOR.iloc[-1]-1)<1e-10
    print('default rejects affected age and pools; unaffected point unchanged',flush=True)
    os.environ['KCOR_RISK_HISTORY_POLICY']='allow_exits'
    try:
        assert not invalid_components(altered,start)
        h=histories(altered,sheet,'disjoint');draw=draw_history(h,np.random.default_rng(5));assert not invalid_components(draw,start)
        x=run(sheet,altered);assert x.risk_history_policy.eq('allow_exits').all();a=x[x.YearOfBirth.eq(1950)&x.Dose_num.eq(1)];assert a.normalization_status.eq('ok').all() and abs(a.KCOR.iloc[-1]-1.45793)<1e-5
        print('explicit exit-enabled endpoint',a.KCOR.iloc[-1],flush=True)
        try:invalid_components(frame,start,layout='table8')
        except ValueError:pass
        else:raise AssertionError('overlap allowed exits')
    finally:os.environ.pop('KCOR_RISK_HISTORY_POLICY',None)
if __name__=='__main__':main()

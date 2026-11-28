import os,sys
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'test'));sys.path.insert(0,str(ROOT/'code'))
from test_round7_workbook import run
from fixed_cohort_history import invalid_components
from bootstrap_reporting import histories
def fixture():
    dates=pd.date_range('2021-06-14',periods=56,freq='7D')
    rows=[]
    for yob in (1940,1950):
        for dose in (0,1):
            for sex,d in [('M',20),('F',1)]:
                rows.append(pd.DataFrame(dict(YearOfBirth=yob,Dose=dose,Sex=sex,DateDied=dates,ISOweekDied=dates.strftime('%G-%V'),Alive=30000-d*np.arange(len(dates)),Dead=d)))
    return pd.concat(rows,ignore_index=True)
def main():
    frame=fixture()
    base=run('2021_24',frame)
    assert abs(base[(base.YearOfBirth==1950)&(base.Dose_num==1)].KCOR.iloc[-1]-1)<1e-10
    mask=(frame.YearOfBirth==1950)&(frame.Dose==1)&(frame.Sex=='M')
    cases={'interior':frame[~(mask&frame.DateDied.eq('2021-12-27'))],
           'trailing':frame[~(mask&frame.DateDied.ge('2021-12-27'))],
           'age_final':frame[~((frame.YearOfBirth==1950)&(frame.Dose==1)&frame.DateDied.eq(frame.DateDied.max()))],
           'duplicate':pd.concat([frame,frame[mask].iloc[[8]]],ignore_index=True)}
    for label,x in cases.items():
        bad=invalid_components(x,pd.Timestamp('2021-06-14'));assert (1950,1) in bad,(label,bad)
        try:histories(x,'2021_24','disjoint')
        except ValueError:pass
        else:raise AssertionError(label+' accepted by bootstrap')
        out=run('2021_24',x)
        for age in [1950,0,-2]:
            g=out[(out.YearOfBirth==age)&(out.Dose_num==1)]
            assert not g.empty and g.KCOR.isna().all() and g.normalization_status.eq('missing_week_history').all(),(label,age,g[['KCOR','normalization_status']].tail())
        u=out[(out.YearOfBirth==1940)&(out.Dose_num==1)]
        assert np.isfinite(u.KCOR.iloc[-1]) and abs(u.KCOR.iloc[-1]-1)<1e-10
        print(label,'affected unavailable; independent age unchanged; bootstrap rejected',flush=True)
    exhausted=frame.copy()
    cutoff=pd.Timestamp('2021-12-20')
    exhausted.loc[mask&exhausted.DateDied.eq(cutoff),'Dead']=exhausted.loc[mask&exhausted.DateDied.eq(cutoff),'Alive']
    exhausted=exhausted[~(mask&exhausted.DateDied.gt(cutoff))]
    assert not invalid_components(exhausted,pd.Timestamp('2021-06-14'))
    histories(exhausted,'2021_24','disjoint')
    out=run('2021_24',exhausted)
    assert out[out.YearOfBirth.eq(1950)].normalization_status.eq('ok').all()
    print('genuine exhaustion accepted by both paths',flush=True)
if __name__=='__main__':main()

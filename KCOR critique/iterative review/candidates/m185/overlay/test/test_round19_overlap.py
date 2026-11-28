"""Table 8 point and person-bootstrap validation share nested-history feasibility."""
from pathlib import Path
import os,sys,io,tempfile
from contextlib import redirect_stdout,redirect_stderr
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'test'),str(ROOT/'code')]
from test_round7_workbook import K
from fixed_cohort_history import invalid_components,rectangular_overlap_counts
from bootstrap_reporting import histories

SHEET='2021_24'
DATES=pd.date_range('2021-06-14',periods=20,freq='7D')


def fixture():
    parts=[]
    for age in (1940,1950,1970):
        for dose in (0,1,2):
            dead=np.arange(len(DATES))%3+20+dose*4+(age-1940)//10
            alive=20000-np.r_[0,np.cumsum(dead[:-1])]
            parts.append(pd.DataFrame(dict(YearOfBirth=age,Dose=dose,Sex='F',DateDied=DATES,
                ISOweekDied=DATES.strftime('%G-%V'),Alive=alive,Dead=dead)))
    f=pd.concat(parts,ignore_index=True)
    inclusive=f.groupby(['Dose','Sex','DateDied','ISOweekDied'],as_index=False)[['Alive','Dead']].sum()
    return pd.concat([f[f.YearOfBirth.ne(1970)],inclusive.assign(YearOfBirth=1960)],ignore_index=True)


def written(frame):
    os.environ.update(KCOR_BOOTSTRAP_LAYOUT='table8',KCOR_BOOTSTRAP_DRAWS='0',KCOR_NPH_MODE='identified_only')
    K.ENROLLMENT_DATES=[SHEET];K.OVERRIDE_DOSE_PAIRS=[(1,0),(2,0),(2,1)];K.OVERRIDE_YOBS=None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src=Path(tmp)/'input.xlsx';dst=Path(tmp)/'out.xlsx'
        frame.to_excel(src,sheet_name=SHEET,index=False)
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            K.process_workbook(str(src),str(dst))
        return pd.read_excel(dst,sheet_name='dose_pairs')


def rejected(frame):
    try:histories(frame,SHEET,'table8')
    except ValueError as exc:
        assert 'invalid overlapping history counts' in str(exc),str(exc)
    else:raise AssertionError('bootstrap accepted impossible nested history')


def main():
    archive=ROOT/'analysis/table8/input.xlsx'
    for sheet,expected in [('2021_24',2511927),('2022_06',2450449)]:
        source=pd.read_excel(archive,sheet_name=sheet,parse_dates=['DateDied'])
        start=pd.Timestamp.fromisocalendar(*map(int,sheet.split('_')),1)
        post=source[source.DateDied.ge(start)]
        assert not invalid_components(post,start,layout='table8')
        _,dates,cells=histories(source,sheet,'table8')
        assert cells.sum()==expected
        # Preserve every archived death and survivor cell, not just the total.
        grouped=post.groupby(['YearOfBirth','Dose','DateDied'])[['Alive','Dead']].sum()
        _,rectangle=rectangular_overlap_counts(grouped,start,post.DateDied.max())
        a=rectangle.Alive.to_numpy().reshape(3,3,-1);d=rectangle.Dead.to_numpy().reshape(3,3,-1)
        prior=np.concatenate([d,(a-d)[:,:,-1:]],axis=2);prior[2]-=prior[0]+prior[1]
        np.testing.assert_array_equal(cells,prior)
        print('Archived layout accepted:',sheet,len(dates),'weeks',cells.sum(),'people',flush=True)
    f=fixture();assert not invalid_components(f,DATES[0],layout='table8')
    _,_,cells=histories(f,SHEET,'table8');assert cells.sum()==180000
    baseline=written(f)
    inc=f.YearOfBirth.eq(1960)&f.Dose.eq(1)
    idx=f.loc[inc].sort_values('DateDied').index
    for kind in ('opening','weekly_deaths','final_survivors'):
        x=f.copy()
        if kind=='opening':x.loc[idx,'Alive']-=30000
        elif kind=='weekly_deaths':
            # One death-week cell is too small, despite ample risk and survivors.
            x.loc[idx[7],'Dead']=1
            x.loc[idx[8:],'Alive']+=f.loc[idx[7],'Dead']-1
        else:
            # Positive opening remainder, but insufficient members survive
            # to account for both constituent final survivors.
            x.loc[idx,'Alive']-=19999
        bad=invalid_components(x,DATES[0],layout='table8')
        assert set(bad)=={(1960,1)},bad
        assert 'invalid overlapping history counts' in bad[(1960,1)][0][1]
        rejected(x)
        out=written(x)
        involving_one=out.Dose_num.eq(1)|out.Dose_den.eq(1)
        affected=out.YearOfBirth.isin([1960,-2,0])&involving_one
        assert affected.any() and out.loc[affected,'KCOR'].isna().all()
        assert not out.loc[affected,'normalization_status'].eq('ok').any()
        constituents=out.YearOfBirth.isin([1940,1950])
        unchanged=constituents | (out.YearOfBirth.isin([1960,-2])&~involving_one)
        np.testing.assert_allclose(out.loc[unchanged,'KCOR'],baseline.loc[unchanged,'KCOR'],equal_nan=True)
        unaffected_pool=out.YearOfBirth.eq(0)&~involving_one
        assert np.isfinite(out.loc[unaffected_pool,'KCOR'].iloc[-1])
        print(kind,'affected statuses',out.loc[affected,'normalization_status'].unique(),
              'independent constituent and direct comparison values unchanged',flush=True)
    fractional=f.copy();fractional[['Alive','Dead']]=fractional[['Alive','Dead']]/10
    assert not invalid_components(fractional,DATES[0],layout='table8')
    rejected(fractional)
    print('Fractional expected-count point layout retained; person bootstrap requires integers.',flush=True)


if __name__=='__main__':main()

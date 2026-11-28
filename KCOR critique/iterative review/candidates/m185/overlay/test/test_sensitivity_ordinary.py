"""Verify the sensitivity sweep against ordinary workbook outputs."""
import os,sys,io,tempfile
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
os.environ.update(DATASET='Czech',KCOR_BOOTSTRAP_DRAWS='0',MONTE_CARLO='0',SA_BASELINE_WEEKS='2,4,8',SA_QUIET_START_OFFSETS='0',KCOR_BOOTSTRAP_LAYOUT='disjoint')
import KCOR as K
SHEET='2021_24'
def fixture():
    dates=pd.date_range('2021-06-14',periods=160,freq='7D')
    rows=[]
    for age in [1940,1970]:
        for dose in [0,1]:
            n=200000;theta=2+age/1000+dose
            G=.0005*np.expm1(.085*np.arange(161)/52)/(.085/52)
            alive=np.round(n*(1+theta*G)**(-1/theta)).astype(int)
            rows.append(pd.DataFrame(dict(YearOfBirth=age,Dose=dose,Sex='O',DateDied=dates,ISOweekDied=dates.strftime('%G-%V'),Alive=alive[:-1],Dead=-np.diff(alive))))
    return pd.concat(rows,ignore_index=True)
def aggregate(frame,age):
    source=frame if age==-2 else frame[frame.YearOfBirth>=1960]
    return source.groupby(['Dose','DateDied','ISOweekDied'],as_index=False)[['Alive','Dead']].sum().assign(YearOfBirth=age,Sex='O')
def run(src,dest,mode,sa):
    os.environ['KCOR_NPH_MODE']=mode;os.environ['SENSITIVITY_ANALYSIS']='1' if sa else '0'
    K.ENROLLMENT_DATES=[SHEET];K.OVERRIDE_DOSE_PAIRS=[(1,0)];K.OVERRIDE_YOBS=[-2]
    with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
        return K.process_workbook(str(src),str(dest/'KCOR_SA.xlsx'),'run.log',return_rows=True)
def main():
    frame=fixture();variants=[frame,pd.concat([frame,aggregate(frame,-2)],ignore_index=True),pd.concat([frame,aggregate(frame,-2),aggregate(frame,-60)],ignore_index=True)]
    K.get_theta_estimation_windows()
    saved=K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE
    window=dict(start_iso='2023-01',end_iso='2023-52',start_int=202301,end_int=202352,label='SA test')
    with tempfile.TemporaryDirectory() as t:
        t=Path(t)
        for mode in ['identified_only','forced_alpha']:
            all_values=[]
            for i,f in enumerate(variants):
                dest=t/f'{mode}-{i}';dest.mkdir()
                src=dest/'input.xlsx';f.to_excel(src,sheet_name=SHEET,index=False)
                K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE=saved
                K.KCOR_NORMALIZATION_WEEKS=4;K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE=4+K.DYNAMIC_HVE_SKIP_WEEKS-1
                run(src,dest,mode,True)
                cells=pd.read_csv(dest/'KCOR_SA_cells.csv')
                assert len(cells)==3 and cells.status.eq('ok').all(),cells
                all_values.append(cells.kcor.to_numpy())
                # Direct ordinary comparisons at all three reference lengths.
                for bw,value in zip(cells.reference_weeks,cells.kcor):
                    K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE=[window]
                    K.KCOR_NORMALIZATION_WEEKS=int(bw);K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE=int(bw)+K.DYNAMIC_HVE_SKIP_WEEKS-1
                    rows=run(src,dest,mode,False)
                    point=rows[rows.YearOfBirth.eq(-2)&rows.Dose_num.eq(1)&rows.Dose_den.eq(0)&pd.to_datetime(rows.Date).eq(pd.Timestamp(K.get_reporting_date(SHEET)))].KCOR
                    assert len(point)==1 and abs(point.iloc[0]-value)<1e-10,(mode,i,bw,point,value)
            assert np.max(np.abs(np.array(all_values)-all_values[0]))<1e-10
            print(mode, 'all reference lengths match ordinary calculation; redundant aggregates invariant',flush=True)
            # Keep the full disjoint data and remove only the supplied total's
            # last positive-risk week. Source horizon must not shrink in SA.
            f=variants[1];f=f[~(f.YearOfBirth.eq(-2)&f.DateDied.eq(f.DateDied.max()))]
            dest=t/f'{mode}-missing-total';dest.mkdir();src=dest/'input.xlsx';f.to_excel(src,sheet_name=SHEET,index=False)
            K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE=saved
            K.KCOR_NORMALIZATION_WEEKS=4;K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE=4+K.DYNAMIC_HVE_SKIP_WEEKS-1
            run(src,dest,mode,True)
            cells=pd.read_csv(dest/'KCOR_SA_cells.csv');assert len(cells)==3 and cells.status.eq('unavailable').all() and cells.kcor.isna().all(),cells
            K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE=[window]
            rows=run(src,dest,mode,False)
            point=rows[rows.YearOfBirth.eq(-2)&rows.Dose_num.eq(1)&rows.Dose_den.eq(0)&pd.to_datetime(rows.Date).eq(pd.Timestamp(K.get_reporting_date(SHEET)))];assert len(point)==1 and point.KCOR.isna().all() and point.normalization_status.eq('missing_week_history').all(),point
            print(mode,'incomplete supplied total unavailable for all reference lengths',flush=True)
    K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE=saved
if __name__=='__main__':main()

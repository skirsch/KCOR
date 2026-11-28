"""Isolate overlapping-population aggregation at identical current settings."""
import os,sys,tempfile,io,json
from pathlib import Path
import pandas as pd
from contextlib import redirect_stdout,redirect_stderr
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'));os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only',KCOR_BOOTSTRAP_DRAWS='0',SENSITIVITY_ANALYSIS='0')
import KCOR as K
def main():
    sh='2021_24';src=ROOT/'data/Czech/KCOR_CMR.xlsx';raw=pd.read_excel(src,sheet_name=sh)
    raw['DateDied']=pd.to_datetime(raw.DateDied);raw=raw[raw.YearOfBirth.le(2020)&raw.Dose.isin([0,1,2])]
    K.ENROLLMENT_DATES=[sh];K.OVERRIDE_YOBS=[-2];K.OVERRIDE_DOSE_PAIRS=[(1,0),(2,0),(2,1)]
    K.get_theta_estimation_windows()
    K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE=[dict(start_iso='2023-01',end_iso='2023-52',start_int=202301,end_int=202352,label='SA comparison')]
    old=raw.groupby(['Dose','DateDied','ISOweekDied'],as_index=False)[['Alive','Dead']].sum().assign(YearOfBirth=-2,Sex='O')
    out={}
    for label,frame in [('double_counted',old),('correct',raw[raw.YearOfBirth.eq(-2)])]:
        with tempfile.TemporaryDirectory() as t:
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                rows=K.process_workbook(str(src),str(Path(t)/'out.xlsx'),bootstrap_frames={sh:frame},return_rows=True)
        row=rows[rows.YearOfBirth.eq(-2)&pd.to_datetime(rows.Date).eq(pd.Timestamp(K.get_reporting_date(sh)))]
        out[label]=[{'num':int(x.Dose_num),'den':int(x.Dose_den),'point':float(x.KCOR)} for x in row.itertuples()]
    print(out);(HERE/'population-only-comparison.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()

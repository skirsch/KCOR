"""Exact expected counts: the ordinary workbook retains a flagged near-ceiling fit."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'test'));sys.path.insert(0,str(ROOT/'code'))
from test_round7_workbook import run,K
def main():
    K.get_theta_estimation_windows()
    K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE=[dict(start_iso='2021-24',end_iso='2023-01',start_int=202124,end_int=202301,label='exact test')]
    dates=pd.date_range('2021-06-14',periods=80,freq='7D')
    t=np.maximum(np.arange(81)-K.DYNAMIC_HVE_SKIP_WEEKS,0)
    G=.001*np.expm1(.085*t/52)/(.085/52)
    frames=[]
    for dose,theta in [(0,0),(1,499.95)]:
        H=np.log1p(theta*G)/theta if theta else G
        risk=1e7*np.exp(-H)
        frames.append(pd.DataFrame(dict(YearOfBirth=1950,Dose=dose,Sex='O',DateDied=dates,ISOweekDied=dates.strftime('%G-%V'),Alive=risk[:-1],Dead=-np.diff(risk))))
    x=run('2021_24',pd.concat(frames,ignore_index=True))
    a=x[x.YearOfBirth.eq(1950)].iloc[-1]
    print('ordinary flagged',a.KCOR,a.theta_num,a.theta0_status_num,a.normalization_status)
    assert abs(a.KCOR-1)<1e-5 and abs(a.theta_num-499.95)<1e-3 and a.theta0_status_num=='THETA_UPPER_BOUND'
if __name__=='__main__':main()

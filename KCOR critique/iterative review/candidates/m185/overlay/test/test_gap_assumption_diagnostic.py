"""9.7: signed material deficit flag without modifying the constrained estimator."""
from pathlib import Path
import os,sys,io,tempfile
from contextlib import redirect_stdout,redirect_stderr
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'test'));sys.path.insert(0,str(ROOT/'code'))
from test_round7_workbook import K
from bootstrap_reporting import gap_diagnostic

def direct():
    gamma=.085/52;t=np.arange(120,dtype=float);G=.002*np.expm1(gamma*np.arange(121))/gamma;theta=4.
    quiet=(t<10)|(t>=70);key=('2021_24',1950,1);rng=np.random.default_rng(9);N=10_000_000
    for q in (.3,1.,1.7):
        E=np.r_[0.,np.cumsum(np.diff(G)*np.where((t>=10)&(t<70),q,1.))];H=np.log1p(theta*E)/theta
        h=np.diff(H);(_,fit),d=K.fit_theta0_gompertz(h,t,quiet,4,gamma);v=K.invert_gamma_frailty(np.cumsum(h),fit);point=(v[-1]/v[3])/(G[-1]/G[4]);draws=[]
        S=np.exp(-H);probs=np.r_[-np.diff(S),S[-1]];counts=np.r_[N*probs[:-1],N*probs[-1]]
        for j in range(49):
            c=rng.multinomial(N,probs);risk=np.cumsum(c[::-1])[::-1][:-1];hd=K.hazard_from_mr(c[:-1]/risk)
            _,diag=K.fit_theta0_gompertz(hd,t,quiet,4,gamma,deaths_arr=c[:-1],risk_arr=risk);draws.append({key:diag.get('gap_first_pass')})
        rows,status=gap_diagnostic({key:d['gap_first_pass']},draws)
        print(q,point,rows,status,flush=True)
        if q==.3:assert status[key]=='violation' and abs(point-.58847707717)<1e-8
        else:assert status[key]!='violation'
        assert d['status']=='ok'
    _,status=gap_diagnostic({key:[(70,-.1)]},[{}, {key:[(70,-.1)]}]);assert status[key]=='not_assessed'

def workbook():
    os.environ.update(KCOR_BOOTSTRAP_DRAWS='19',KCOR_BOOTSTRAP_WORKERS='4',KCOR_NPH_MODE='identified_only',KCOR_BOOTSTRAP_LAYOUT='disjoint')
    sheet='2021_24';K.ENROLLMENT_DATES=[sheet];K.OVERRIDE_DOSE_PAIRS=[(1,0)];K.OVERRIDE_YOBS=None
    dates=pd.date_range('2021-06-14',periods=160,freq='7D');t=np.maximum(np.arange(161)-K.DYNAMIC_HVE_SKIP_WEEKS,0);gamma=.085/52;G=.002*np.expm1(gamma*t)/gamma
    protective=(dates>=pd.Timestamp('2021-09-13'))&(dates<pd.Timestamp('2022-05-30'));frames=[]
    for dose in [0,1]:
        E=np.r_[0.,np.cumsum(np.diff(G)*np.where(protective,.3 if dose==1 else 1.,1.))];S=(1+4*E)**(-.25);alive=np.rint(10_000_000*S).astype(int)
        frames.append(pd.DataFrame(dict(YearOfBirth=1950,Dose=dose,Sex='O',DateDied=dates,ISOweekDied=dates.strftime('%G-%V'),Alive=alive[:-1],Dead=-np.diff(alive))))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        src=Path(t)/'input.xlsx';dst=Path(t)/'out.xlsx';pd.concat(frames).to_excel(src,sheet_name=sheet,index=False)
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):K.process_workbook(str(src),str(dst))
        gaps=pd.read_excel(dst,sheet_name='gap_diagnostics');out=pd.read_excel(dst,sheet_name='dose_pairs');print(gaps.to_string(index=False),flush=True)
        affected=out[out.YearOfBirth.eq(1950)&out.Dose_num.eq(1)&out.Dose_den.eq(0)]
        print('endpoint',affected[['KCOR','CI_lower','CI_upper','gap_assumption_status','theta0_status_num','CI_point_flagged']].tail(1).to_string(index=False),flush=True)
        assert gaps.status.eq('violation').any() and affected.gap_assumption_status.eq('violation').all() and affected.CI_point_flagged.all()
        assert affected.KCOR.notna().any() and affected.CI_lower.notna().any()
if __name__=='__main__':direct();workbook()

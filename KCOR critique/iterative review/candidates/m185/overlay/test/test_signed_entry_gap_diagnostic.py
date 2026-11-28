"""12.4: identify an actual protective gap without flagging a later ordinary gap."""
from pathlib import Path
import os,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'));os.environ['DATASET']='Czech';os.environ['KCOR_NPH_MODE']='identified_only'
import KCOR as K
from bootstrap_reporting import gap_diagnostic
def main():
    t=np.arange(65,dtype=float);gamma=.085/52;k=.006;theta=8.
    G=k*np.expm1(gamma*np.arange(66))/gamma
    increments=np.diff(G);increments[4:15]*=.3
    X=np.r_[0.,np.cumsum(increments)]
    H=np.log1p(theta*X)/theta;quiet=((t>=15)&(t<35))|(t>=45)
    fit,d=K.fit_theta0_gompertz(np.diff(H),t,quiet,4,gamma)
    expected=-(.7)*np.diff(G)[4:15].sum()
    print('fitted parameters',fit,'seed',d['theta0_init'],'diagnostics',d['gap_first_pass'])
    assert d['gap_seed_status']=='ok' and abs(d['theta0_init']-theta)<1e-6
    assert [v[0] for v in d['gap_first_pass']]==[15.,45.]
    np.testing.assert_allclose([v[1] for v in d['gap_first_pass']],[expected,0],atol=1e-8)
    rng=np.random.default_rng(921);S=np.exp(-H);prob=np.r_[-np.diff(S),S[-1]];key=('synthetic',1950,0);draws=[]
    for _ in range(50):
        cells=rng.multinomial(5_000_000,prob);risk=np.cumsum(cells[::-1])[::-1][:-1]
        _,a=K.fit_theta0_gompertz(K.hazard_from_mr(cells[:-1]/risk),t,quiet,4,gamma,deaths_arr=cells[:-1],risk_arr=risk)
        draws.append({key:a['gap_first_pass']})
    rows,status=gap_diagnostic({key:d['gap_first_pass']},draws)
    print(rows)
    assert rows[0]['status']=='violation' and rows[1]['status']=='no_deficit_detected'
    assert rows[1]['lower95']<=0<=rows[1]['upper95']
if __name__=='__main__':main()

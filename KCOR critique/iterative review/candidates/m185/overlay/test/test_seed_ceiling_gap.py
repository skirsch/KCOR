"""10.3: seed and joint fit use the same configured ceiling."""
from pathlib import Path
import sys,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'));os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only');import KCOR as K
from bootstrap_reporting import gap_diagnostic

def main():
    t=np.arange(120,dtype=float);gamma=.085/52;G=.002*np.expm1(gamma*np.arange(121))/gamma;quiet=(t<20)|(t>=70);N=10_000_000;key=('2021_24',1950,1)
    for theta in (200.,550.):
        H=np.log1p(theta*G)/theta;h=np.diff(H);(_,fit),d=K.fit_theta0_gompertz(h,t,quiet,4,gamma)
        print('true',theta,'seed',d['theta0_init'],'final',fit,'seed status',d['gap_seed_status'],'gap',d['gap_first_pass'],flush=True)
        if theta==200:
            assert abs(d['theta0_init']-200)<1e-6 and abs(fit-200)<1e-6 and abs(d['gap_first_pass'][0][1])<1e-9 and d['gap_seed_status']=='ok'
            rng=np.random.default_rng(423);S=np.exp(-H);prob=np.r_[-np.diff(S),S[-1]];draws=[]
            for i in range(49):
                c=rng.multinomial(N,prob);risk=np.cumsum(c[::-1])[::-1][:-1];_,diag=K.fit_theta0_gompertz(K.hazard_from_mr(c[:-1]/risk),t,quiet,4,gamma,deaths_arr=c[:-1],risk_arr=risk);draws.append({key:diag['gap_first_pass']})
            s,status=gap_diagnostic({key:d['gap_first_pass']},draws);print('zero gap interval',s,flush=True);assert status[key]=='no_deficit_detected' and s[0]['lower95']<=0<=s[0]['upper95']
        else:assert d['seed_theta_hit_bound'] and d['gap_seed_status']=='bounded_seed' and len(d['gap_first_pass'])==1 and d['gap_first_pass'][0][0]==70 and np.isnan(d['gap_first_pass'][0][1])
    # A small configured ceiling also applies to the initial fit.
    native=K.get_theta_degenerate_fit_config;K.get_theta_degenerate_fit_config=lambda:{**native(),'theta0_max':.25}
    try:
        H=np.log1p(.1*G)/.1;(_,fit),d=K.fit_theta0_gompertz(np.diff(H),t,quiet,4,gamma);assert abs(d['theta0_init']-.1)<1e-5 and abs(fit-.1)<1e-5
    finally:K.get_theta_degenerate_fit_config=native
if __name__=='__main__':main()

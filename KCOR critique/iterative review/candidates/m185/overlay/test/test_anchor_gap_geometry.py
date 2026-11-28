"""10.4: anchors are baseline fitting weeks, not gaps."""
from pathlib import Path
import os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'));os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only')
import KCOR as K

def main():
    t=np.arange(55,dtype=float); gamma=.085/52; k=.002; theta=4.
    G=k*np.expm1(gamma*np.arange(56))/gamma
    H=np.log1p(theta*G)/theta
    rng=np.random.default_rng(104); c=rng.multinomial(100000,np.r_[-np.diff(np.exp(-H)),np.exp(-H[-1])]); risk=np.cumsum(c[::-1])[::-1][:-1]; h=K.hazard_from_mr(c[:-1]/risk)
    fits=[]
    for start in (0,1,2,3,4):
        fit,d=K.fit_theta0_gompertz(h,t,t>=start,4,gamma,deaths_arr=c[:-1],risk_arr=risk)
        assert d['success'] and d['n_fit']==55 and d['gap_first_pass']==[] and d['delta_raw_values']==[]
        assert d['n_quiet_bins']==55-start
        fits.append(fit);print('quiet starts',start,'fit',fit,'gaps',d['gap_first_pass'])
    assert np.allclose(fits,fits[0],rtol=1e-6,atol=1e-8)
    # Exact multiweek excess: anchors 0..3, gap 4..9, quiet from 10.
    excess=.02; X=G+excess*(np.arange(56)>=10)
    hh=np.diff(np.log1p(theta*X)/theta)
    fit,d=K.fit_theta0_gompertz(hh,t,t>=10,4,gamma)
    print('genuine first gap',fit,d['gap_first_pass'],d['delta_applied_values'])
    assert d['success'] and len(d['gap_first_pass'])==1 and d['gap_first_pass'][0][0]==10
    assert abs(fit[1]-theta)<1e-5 and abs(d['delta_applied_values'][0]-excess)<1e-7
    q=(t<20)|(t>=30); X=G+excess*(np.arange(56)>=30); hh=np.diff(np.log1p(theta*X)/theta)
    fit,d=K.fit_theta0_gompertz(hh,t,q,4,gamma)
    assert d['success'] and len(d['gap_first_pass'])==1 and d['gap_first_pass'][0][0]==30 and abs(d['delta_applied_values'][0]-excess)<1e-7
    print('internal gap',fit,d['gap_first_pass'],d['delta_applied_values'])
if __name__=='__main__':main()

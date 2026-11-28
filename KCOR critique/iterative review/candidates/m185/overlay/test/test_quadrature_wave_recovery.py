"""Exact-wave recovery through the production fit and correction entry points."""
from pathlib import Path
import os,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from test_survival_quadrature import reference
import KCOR as K
def test_quadrature_wave_recovery():
    gamma=.085/52;k=.001
    t=np.arange(80,dtype=float);quiet=(t<15)|(t>=25)
    dates=np.datetime64('2021-06-14')+t.astype(int)*np.timedelta64(7,'D')
    B=k*np.expm1(gamma*np.arange(81))/gamma
    for theta,alpha,exposure in [(2.,1.18,.08),(20.,1.3,.5),(99.,1.18,2.1270330577)]:
        W=exposure*np.clip((np.arange(81)-15)/10,0,1)
        H=np.array([reference(b,w,theta,alpha) for b,w in zip(B,W)])
        h=np.diff(H)
        (kfit,thfit),diag=K.fit_theta0_gompertz(h,t,quiet,4,gamma,deaths_arr=np.full(len(t),1000),wave=(15,24),alpha=alpha)
        assert diag['success'],diag
        # Known input parameters recover the full wave baseline and post-wave curve.
        result,info=K.apply_nph_correction_post_inversion(K.invert_gamma_frailty(H[1:],theta),t,dates,theta,alpha,dates[15],dates[24],k,gamma,t_rebased=t,observed_cumhaz=H[1:])
        max_err=float(np.max(abs(result-B[1:]-W[1:])))
        np.testing.assert_allclose(result,B[1:]+W[1:],atol=2e-8,rtol=1e-8)
        assert abs(kfit-k)<2e-7,(kfit,k)
        assert abs(thfit-theta)<.05,(thfit,theta)
        print('theta',theta,'fitted',thfit,'k',kfit,'alpha',alpha,'max curve error',max_err,'max wave solver residual',info['max_survival_residual'],flush=True)
if __name__=='__main__':test_quadrature_wave_recovery()

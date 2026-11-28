"""Exact-data regression for the proportional least-squares solver scale."""
from pathlib import Path
import sys,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="identified_only")
import KCOR as K
def test_scaled_proportional_solver():
    gamma=.085/52
    for k in (1e-6,1e-5,1e-4,.001):
        for theta in (.5,4.,50.):
            for n in (12,52):
                t=np.arange(n,dtype=float)
                B=k*np.expm1(gamma*np.arange(n+1))/gamma
                h=np.diff(np.log1p(theta*B)/theta)
                (kh,th),diag=K.fit_theta0_gompertz(h,t,np.ones(n,bool),4,gamma)
                assert diag["success"],diag
                np.testing.assert_allclose(kh,k,atol=1e-12,rtol=1e-8)
                np.testing.assert_allclose(th,theta,atol=1e-6,rtol=0)
                assert diag["rmse_hazard"]<1e-12
    print("PASS 24 exact low-rate and frailty grid cases")
if __name__=="__main__":test_scaled_proportional_solver()

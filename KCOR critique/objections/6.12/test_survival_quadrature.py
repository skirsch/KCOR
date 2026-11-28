"""Regression tests for the shared survival integral (objection 6.12)."""
import os,sys,math,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kcor_survivor import Survival,cumulative_hazard

def reference(B,W,theta,alpha):
    if theta==0:return B+W
    shape=1/theta
    scale=theta/(1+theta*B)
    # For high shape, integrate near the concentrated gamma law on its
    # standard-deviation scale; otherwise integrate the loss against density.
    if shape>=50:
        def integrand(x):
            y=shape+math.sqrt(shape)*x
            if y<=0:return 0.
            return math.exp((shape-1)*math.log(y)-y-gammaln(shape)+.5*math.log(shape)-W*(scale*y)**alpha)
        S,_=quad(integrand,-min(16,math.sqrt(shape)),16,epsabs=1e-14,epsrel=1e-12,limit=300)
        return math.log1p(theta*B)/theta-math.log(S)
    def loss(y):
        if y==0:return 0.
        return math.exp((shape-1)*math.log(y)-y-gammaln(shape))*(-math.expm1(-W*(scale*y)**alpha))
    L,_=quad(loss,0,math.inf,epsabs=1e-13,epsrel=1e-12,limit=300)
    return math.log1p(theta*B)/theta-math.log1p(-L)

def test_survival_quadrature():
    s=Survival(99,1.18);w=2.1270330577*np.array([1-1e-9,1+1e-9])
    out=s.hazard(.01,w)
    assert out[1]>out[0]
    np.testing.assert_allclose(out,[reference(.01,x,99,1.18) for x in w],atol=1e-11,rtol=0)
    np.testing.assert_allclose(Survival(99,1.3).hazard(.01,1.),reference(.01,1.,99,1.3),atol=1e-11,rtol=0)
    worst=0.;n=0
    for th in [.01,.1,.5,1,5,20,99,100]:
        for a in [.5,1.18,1.3,2,3,4]:
            s=Survival(th,a)
            for B in [0.,.01,.5,2.]:
                for W in [0.,1e-5,.01,1.,10.]:
                    actual=s.hazard(B,W);expected=reference(B,W,th,a)
                    worst=max(worst,abs(actual-expected));n+=1
                    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)
    print("Independent integration cases",n,"max absolute error",worst)
    n=0
    for th in [0,1e-9,1e-5,.001,.01,.1,1,5,20,99,100]:
        for a in [.5,1.,1.18,1.3,2.,3.,4.]:
            s=Survival(th,a)
            x=np.r_[0,np.geomspace(1e-8,10,300)]
            for fixed in [0.,.01,.5,2.]:
                for axis in [0,1]:
                    y=s.hazard(x,fixed) if axis==0 else s.hazard(fixed,x)
                    assert np.isfinite(y).all()
                    assert np.min(np.diff(y))>=-5e-12,(th,a,fixed,axis,np.min(np.diff(y)))
                    n+=len(y)-1
    print("Monotonic consecutive comparisons",n)
    for theta in [0,1e-9,1.,99.]:
        a=1.
        B=np.array([0,.01,.5,2]);W=np.array([.01,.05,1.,2.])
        expected=np.log1p(theta*(B+W))/theta if theta else B+W
        np.testing.assert_array_equal(Survival(theta,a).hazard(B,W),expected)
    for theta in [.01,1.,99.]:
        for alpha in [1.18,1.3,3.]:
            s=Survival(theta,alpha)
            for axis in [0,1]:
                B,W=.01,.5;dx=.123
                target=reference(B+dx,W,theta,alpha) if axis==0 else reference(B,W+dx,theta,alpha)
                solved=s.increase(B,W,target,axis,.05)
                np.testing.assert_allclose(solved,dx,atol=1e-8,rtol=0)
    print("Identity and root recovery passed")
if __name__=="__main__":test_survival_quadrature()

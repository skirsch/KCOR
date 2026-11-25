"""Numerical checks on exact survival and information scaling."""
from limits import *
from scipy.integrate import quad_vec
def reference(B,W,theta,alpha):
    from scipy.special import gamma
    def integrand(u):
        x=u**theta; z=theta*x
        return np.exp(-x-z*B-z**alpha*W)
    value,error=quad_vec(integrand,0,np.inf,epsabs=1e-11,epsrel=1e-11)
    return np.log(value*theta/gamma(1/theta))
def main():
    B=np.array([0,.01,.1,.5,1.])
    W=np.array([0,.001,.02,.06,.2])
    for th in (.1,.5,1.,4.,15.):
        np.testing.assert_allclose(log_survival(B,W,th,1.18),reference(B,W,th,1.18),rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(log_survival(B,W,th,1.),-np.log1p(th*(B+W))/th,rtol=1e-12)
    t=np.arange(2,53)
    q=(t<=5)|(t>=35);wave=(t>=27)&(t<=34)
    m=Model(12500,q,wave,4,len(t),1.18)
    p=m.reference(.002,.5,known=[0,.048])
    a=np.array(m.info(p)[:3]);m.N*=4;b=np.array(m.info(p)[:3])
    np.testing.assert_allclose(a/2,b,rtol=1e-10)
    # Short/low mortality no-wave case: analytical discrete weak-hazard limit.
    L=52;k=1e-5;N=1e8
    m=Model(N,np.ones(L,bool),np.zeros(L,bool),4,L,1.)
    p=m.reference(k,.5,known=[])
    # Small nonzero Gompertz growth changes the approximation at ~10%.
    approx=np.sqrt(12/(N*k**3*L*(L**2-1)))
    exact=m.info(p)[0]
    assert abs(exact/approx-1)<.15
    print("Validated nonlinear survival against independent adaptive integration, alpha=1 identity, N^-1/2 scaling and weak-hazard approximation.",exact,approx)
if __name__=="__main__":main()

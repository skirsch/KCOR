"""Local-identification counterexample to a universal external-scale requirement."""
from math import exp,log
from pathlib import Path
import numpy as np,json
from scipy.integrate import quad
from scipy.optimize import root
g=.085/52
B=.001*np.expm1(g*np.array([22.,35.]))/g
W=np.array([.025,.065])
def model(c,a):
    H=[];J=[]
    for b,w in zip(B,W):
        def q(z):return exp(-(1+b)*z-c*w*z**a)
        L=quad(q,0,np.inf,epsabs=1e-12)[0]
        d1=quad(lambda z:z**a*q(z),0,np.inf,epsabs=1e-12)[0]
        d2=quad(lambda z:z**a*log(z)*q(z) if z>0 else 0,0,np.inf,epsabs=1e-12)[0]
        H.append(-log(L));J.append([w*d1/L,c*w*d2/L])
    return np.array(H),np.array(J)
def main():
    h,j=model(.7,1.18)
    fit=root(lambda p:model(*p)[0]-h,[.8,1.1])
    assert fit.success and np.allclose(fit.x,[.7,1.18],rtol=1e-7)
    det=float(np.linalg.det(j));assert abs(det)>1e-6
    out=dict(baseline=B.tolist(),wave=W.tolist(),reference=(B+W).tolist(),observed=h.tolist(),jacobian=j.tolist(),determinant=det,fitted=fit.x.tolist())
    (Path(__file__).parent/'counterexample.json').write_text(json.dumps(out,indent=2))
    print(out)
if __name__=='__main__':main()

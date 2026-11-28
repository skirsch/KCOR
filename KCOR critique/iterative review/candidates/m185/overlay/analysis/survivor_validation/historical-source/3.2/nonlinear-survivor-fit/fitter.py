"""Isolated exact nonlinear survivor-law fitter; preserves weekly LS objective.

Wave exposures use the same baseline/deficit attribution as the 1.2 trial.
Between quiet blocks, nonwave excess is proportional and nonnegative, and
profiled from observed cumulative survival. The first gap is included.
"""
from pathlib import Path
import os,sys,importlib.util
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import numpy as np
import math
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammaln
from scipy.optimize import brentq,least_squares
from functools import lru_cache
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"1.2/weekly-survival"),str(ROOT/"2.7/low-frailty/all-window-joint")]
from trial import cumulative_hazard as reference_hazard, make_correction
from experiment import candidate
# Numerically stable, normalized gamma quadrature. Same integral as 1.2.
@lru_cache(maxsize=256)
def nodes(theta):
    j=np.arange(1,64.)
    z,v=eigh_tridiagonal(1+2*theta*np.arange(64.),np.sqrt(theta*j*(1+theta*(j-1))))
    return z,v[0]**2

class Survival:
    def __init__(self,theta,alpha):
        self.th=theta;self.a=alpha
        if theta>=1e-5 and alpha!=1:
            z,w=nodes(float(theta));self.za=np.maximum(z,0.)**alpha;self.w=w
            self.m1=np.exp(alpha*np.log(theta)+gammaln(1/theta+alpha)-gammaln(1/theta))
            self.m2=np.exp(2*alpha*np.log(theta)+gammaln(1/theta+2*alpha)-gammaln(1/theta))
    def hazard(self,B,W):
        if np.isscalar(B) and np.isscalar(W):
            th,a=self.th,self.a
            if th<1e-5:
                return B+W-th/2*((B+a*W)**2-a*(a-1)*W)
            if W==0 or a==1:
                return math.log1p(th*(B+W))/th
            scale=1+th*B;u=W/scale**a
            v=u*self.za
            expectation=1-u*self.m1+u*u*self.m2/2+float(np.dot(np.expm1(-v)+v-v*v/2,self.w))
            if expectation>1e-6 and u*self.m1<2 and math.isfinite(expectation):
                return math.log1p(th*B)/th-math.log(expectation)
            return reference_hazard(float(B),float(W),th,a)
        B,W=np.broadcast_arrays(np.asarray(B,dtype=float),np.asarray(W,dtype=float))
        shape=B.shape;B=B.ravel();W=W.ravel();th=self.th;a=self.a
        if th<1e-5:
            out=B+W-th/2*((B+a*W)**2-a*(a-1)*W)
        elif a==1:
            out=np.log1p(th*(B+W))/th
        else:
            scale=1+th*B;u=W/scale**a
            # Exact first two noninteger moments; integrate smooth remainder.
            v=u[:,None]*self.za
            remainder=(np.expm1(-v)+v-v*v/2)@self.w
            expectation=1-u*self.m1+u*u*self.m2/2+remainder
            out=np.log1p(th*B)/th
            good=(expectation>1e-6)&(u*self.m1<2.)&np.isfinite(expectation)
            out[good]-=np.log(expectation[good])
            for i in np.flatnonzero(~good):
                out[i]=reference_hazard(float(B[i]),float(W[i]),th,a)
        out=out.reshape(shape)
        return float(out) if not shape else out
    def increase(self,B,W,target,axis,guess):
        # Solve cumulative survival on the monotone indicated exposure axis.
        base=self.hazard(B,W)
        if target<=base+1e-15:return 0.
        def f(x):return self.hazard(B+x,W)-target if axis==0 else self.hazard(B,W+x)-target
        upper=max(guess,1e-10)
        while f(upper)<0:
            upper*=2
            if not np.isfinite(upper) or upper>1e150:raise ArithmeticError("exposure not bracketed")
        return brentq(f,0.,upper,xtol=1e-12,rtol=1e-13)

class Profile:
    def __init__(self,h,t,quiet,anchors,gamma,wave):
        self.idx=np.flatnonzero(np.isfinite(h)&np.isfinite(t)&(t>=0))
        self.h=h[self.idx];self.t=t[self.idx]
        self.obs=np.r_[0.,np.cumsum(np.maximum(self.h,0.))]
        self.F=np.expm1(gamma*np.r_[self.t,self.t[-1]+1])/gamma
        assert np.allclose(np.diff(self.t),1.)
        self.wave=(self.t>=wave[0])&(self.t<=wave[1])
        self.fit=(quiet[self.idx]|(self.t<anchors))&~self.wave
        self.segments=[]
        kind=np.where(self.wave,2,np.where(self.fit,0,1))
        bounds=np.r_[0,np.flatnonzero(np.diff(kind)!=0)+1,len(kind)]
        # Only data through final fitting week matter, unless wave is later.
        last=np.flatnonzero(self.fit)[-1]+1 if self.fit.any() else 0
        for a,b in zip(bounds[:-1],bounds[1:]):
            if a>=last:break
            self.segments.append((int(a),int(min(b,last)),int(kind[a])))
    def predict(self,k,theta,alpha):
        sv=Survival(theta,alpha); B=W=0.;pred=[];rawgap=[];applied=[];deficits=0
        for a,b,kind in self.segments:
            increments=k*np.diff(self.F[a:b+1])
            if kind==0:
                before=B+np.r_[0.,np.cumsum(increments)]
                q=np.diff(sv.hazard(before,np.full(len(before),W)))
                pred.extend(q);B=before[-1]
            elif kind==1:
                # Reconstruct observed survival at quiet re-entry. As in
                # proportional KCOR, nonpositive additional gap exposure is zero.
                B+=float(increments.sum())
                target=self.obs[b];raw=target-sv.hazard(B,W)
                extra=sv.increase(B,W,target,0,max(raw,1e-10))
                rawgap.append(raw);applied.append(extra);B+=extra
            else:
                # At wave opening, the observed pre-wave survival implies an
                # exact proportional accumulated exposure (same as trial 1.2).
                # If the wave starts at modeled origin this is zero.
                B=np.expm1(theta*self.obs[a])/theta if theta else self.obs[a]
                W=0.
                # A sufficient no-deficit condition: with positive alpha, prior
                # wave selection cannot increase the mean surviving frailty.
                # If observations exceed the no-wave conditional baseline in
                # every week, every sequential W increment is nonnegative.
                # Then one exact solve at exit equals the weekly recursion.
                path=B+np.r_[0.,np.cumsum(increments)]
                baseline_only=np.diff(sv.hazard(path,np.zeros(len(path))))
                if np.all(self.h[a:b]>=baseline_only):
                    B=float(path[-1])
                    W=sv.increase(B,0.,self.obs[b],1,max(self.obs[b]-sv.hazard(B,0.),1e-10))
                    continue
                for j,inc in zip(range(a,b),increments):
                    planned=B+inc;target=self.obs[j+1]
                    loss=sv.hazard(planned,W)
                    if target>loss:
                        B=planned;W+=sv.increase(B,W,target,1,max(target-loss,1e-10))
                    else:
                        deficits+=int(target<loss-1e-12)
                        B+=sv.increase(B,W,target,0,inc)
        return np.array(pred),rawgap,applied,deficits

def make_fitter(model,wave,alpha=1.18):
    base=candidate(model,"multistart")
    def fit(h_arr,t_rebased,quiet_mask,k_anchor_weeks,gamma_per_week,theta0_init=1.,deaths_arr=None,**kwargs):
        h=np.asarray(h_arr);t=np.asarray(t_rebased);quiet=np.asarray(quiet_mask,bool)
        # Identity for no nonlinear wave within the modeled fit history.
        if abs(alpha-1)<=model.NPH_IDENTITY_TOL or not np.any((t>=max(0,wave[0]))&(t<=wave[1])):
            return base(h,t,quiet,k_anchor_weeks,gamma_per_week,theta0_init,deaths_arr)
        initial,diag=base(h,t,quiet,k_anchor_weeks,gamma_per_week,theta0_init,deaths_arr)
        if diag.get("insufficient_deaths") or not np.isfinite(initial[0]):return initial,diag
        p=Profile(h,t,quiet,k_anchor_weeks,gamma_per_week,wave)
        if not p.fit.any() or len(p.h[p.fit])<3:return initial,diag
        target=p.h[p.fit]
        # Scale residuals for numerical optimization only (same LS minimum).
        scale=max(float(target.mean()),1e-8)
        def residual(x):
            k,th=np.exp(x[0]),x[1]
            try:
                pred,_,_,_=p.predict(k,th,alpha)
                e=(target-pred)/scale
                if np.isfinite(e).all():return e
            except (ArithmeticError,ValueError,OverflowError,FloatingPointError):
                pass
            return np.full(len(target),1e6)
        thmax=float(model.get_theta_degenerate_fit_config().get("theta0_max",100.))
        k0=float(np.clip(initial[0],1e-8,.1));th0=float(np.clip(initial[1],0.,thmax)) if np.isfinite(initial[1]) else .5
        candidates=[]
        for th in (th0,.5):
            if candidates and abs(th-th0)<1e-7:continue
            r=least_squares(residual,[np.log(k0),th],bounds=([np.log(1e-8),0.],[np.log(.1),thmax]),
                              ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=200)
            candidates.append(r)
        valid=[r for r in candidates if r.success and np.isfinite(r.cost)]
        r=min(valid or candidates,key=lambda r:r.cost)
        k,theta=float(np.exp(r.x[0])),float(r.x[1])
        pred,gaps,applied,deficits=p.predict(k,theta,alpha)
        e=target-pred;rel=float(np.sqrt(np.mean((e/(target+1e-12))**2)))
        guard=model.get_theta_degenerate_fit_config()
        bound=theta>=thmax*.999
        bad=rel>float(guard.get("bad_fit_relrmse_threshold",1e5))
        flag=bound or bad
        mask=np.zeros_like(quiet);mask[p.idx]=p.fit
        diag.update(success=bool(r.success),status="insufficient_signal" if flag else ("ok" if r.success else "error"),message="nonlinear survivor trial",
                    n_obs=len(target),n_fit=len(target),fit_mask_theta=mask,k_hat=k,theta0_raw=theta,
                    theta_hit_bound=bound,bad_fit=bad,insufficient_signal=flag,
                    rmse_hazard=float(np.sqrt(np.mean(e**2))),fit_residual_max=float(np.max(abs(e))),relRMSE_hazard_fit=rel,
                    delta_negative_clamped=any(x<=0 for x in gaps),delta_negative=any(x<=0 for x in gaps),
                    nonlinear_gap_observed_deficit=gaps,nonlinear_gap_exposure=applied,nonlinear_wave_deficits=deficits,
                    trial_nfev=sum(x.nfev for x in candidates))
        return (k,0. if flag else theta),diag
    return fit


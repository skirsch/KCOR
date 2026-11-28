"""Model-based expected Fisher limits, exact gamma survival with known alpha.
Gap totals are free nuisance parameters; no information from assuming within-gap
hazard shape. Endpoints at normalization and report are retained.
"""
import os
for x in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[x]="1"
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import roots_genlaguerre, gammaln
from scipy.optimize import brentq, least_squares
from scipy.stats import chi2
from functools import lru_cache
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import multiwindow_information as old
sys.path.insert(0,str(HERE.parents[2]/"1.3/multiwindow-coverage"))
import study
GAMMA=.085/52

@lru_cache(maxsize=512)
def quadrature(theta):
    nodes,weights=roots_genlaguerre(128,1/theta-1)
    return theta*nodes,weights/np.exp(gammaln(1/theta))

def log_survival(B,W,theta,alpha):
    if alpha==1:
        return -np.log1p(theta*(B+W))/theta if theta else -(B+W)
    if theta < 1e-5:
        # Taylor limit around the mean-one degenerate frailty law.
        # f''(1)/f(1)=(B+alpha W)^2-alpha(alpha-1)W.
        return -(B+W)+theta/2*((B+alpha*W)**2-alpha*(alpha-1)*W)
    z,w=quadrature(float(theta))
    # Exponential tilt by B is exact for gamma. Integrate only the smooth
    # remainder after removing the first two noninteger-power terms.
    scale=1+theta*B
    v=np.outer(W/scale**alpha,z**alpha)
    first=W/scale**alpha*np.exp(alpha*np.log(theta)+gammaln(1/theta+alpha)-gammaln(1/theta))
    second=(W/scale**alpha)**2*np.exp(2*alpha*np.log(theta)+gammaln(1/theta+2*alpha)-gammaln(1/theta))
    expectation=1-first+second/2+(np.expm1(-v)+v-v*v/2)@w
    return -np.log1p(theta*B)/theta+np.log(expectation)

def design(quiet,wave,anchor,target):
    assert len(quiet)==len(wave) and not np.any(quiet&wave)
    # Quiet weeks individually, arbitrary gap totals split at exponent changes
    # and the endpoints defining KCOR.
    cuts={0,len(quiet),anchor,target}
    for i in range(len(quiet)):
        if quiet[i] or i==0 or quiet[i]!=quiet[i-1] or wave[i]!=wave[i-1]:
            cuts.add(i)
        if quiet[i]: cuts.add(i+1)
    cuts=np.array(sorted(cuts))
    a,b=cuts[:-1],cuts[1:]
    nuisance=np.flatnonzero(~quiet[a])
    F=np.expm1(GAMMA*cuts)/GAMMA
    P=np.zeros((len(cuts),len(nuisance)))
    W=P.copy()
    for j,i in enumerate(nuisance):
        (W if wave[a[i]] else P)[i+1:,j]=1.
    return cuts,F,P,W,nuisance

class Model:
    def __init__(self,N,quiet,wave,anchor,target,alpha):
        self.N=N; self.alpha=alpha
        self.cuts,self.F,self.P,self.W,self.nuisance=design(quiet,wave,anchor,target)
        self.anchor=np.flatnonzero(self.cuts==anchor).item()
        self.target=np.flatnonzero(self.cuts==target).item()
    def paths(self,p):
        B=np.exp(p[0])*self.F+self.P@p[2:]
        W=self.W@p[2:]
        return B,W
    def logS(self,p):
        B,W=self.paths(p)
        return log_survival(B,W,p[1],self.alpha)
    def q(self,p):
        return -np.diff(self.logS(p))
    def growth(self,p):
        B,W=self.paths(p); X=B+W
        return np.log(X[self.target]/X[self.anchor])
    def reference(self,k,theta,observed=None,known=None):
        p=np.r_[np.log(k),theta,np.zeros(len(self.nuisance))]
        if known is not None:
            p[2:]=known
        elif observed is not None:
            for j,i in enumerate(self.nuisance):
                desired=observed[self.cuts[i+1]]-observed[self.cuts[i]]
                def f(x):
                    p[j+2]=x
                    return self.q(p)[i]-desired
                if f(0)<0:
                    high=.01
                    while f(high)<0: high*=2
                    p[j+2]=brentq(f,0,high,xtol=1e-13)
                else: p[j+2]=0
        return p
    def info(self,p):
        q=self.q(p); prob=-np.expm1(-q)
        n=self.N*np.exp(self.logS(p)[:-1])
        assert np.all(prob>0)
        J=[];g=[]
        for j in range(len(p)):
            step=1e-5 if j<2 else 1e-7
            a=p.copy();b=p.copy();a[j]+=step;b[j]-=step
            J.append((self.q(a)-self.q(b))/(2*step))
            g.append((self.growth(a)-self.growth(b))/(2*step))
        A=np.array(J).T*np.sqrt((n*(1-prob)/prob)[:,None])
        scale=np.linalg.norm(A,axis=0)
        U,s,V=np.linalg.svd(A/scale,full_matrices=False)
        assert s[-1]>s[0]*1e-10,(s,self.cuts)
        C=(V.T/s**2)@V/scale[:,None]/scale[None,:]
        g=np.array(g)
        return np.sqrt(C[1,1]),np.sqrt(g@C@g),np.sqrt(C[0,0]),C
    def expected_profile(self,p,grid):
        """Expected likelihood-ratio profile (not exact repeated-sample CI)."""
        q0=self.q(p)
        prob=-np.expm1(-q0); n=self.N*np.exp(self.logS(p)[:-1])
        keep=np.arange(len(p))!=1
        def residual(x,theta):
            trial=p.copy();trial[keep]=x;trial[1]=theta
            q=self.q(trial); pr=-np.expm1(-q)
            # signed sqrt binomial deviance against expected counts
            dev=2*n*(prob*np.log(prob/pr)+(1-prob)*(q-q0))
            return np.sign(q-q0)*np.sqrt(np.maximum(dev,0))
        results=[]
        for th in grid:
            r=least_squares(residual,p[keep],args=(th,),bounds=(np.r_[-20.,np.zeros(len(p)-2)],np.r_[-1.,np.full(len(p)-2,10.)]),max_nfev=500,ftol=1e-10,gtol=1e-10,xtol=1e-10)
            results.append((th,np.sum(r.fun**2),r.success))
        return results

def row(model,p,label,**metadata):
    st,sg,sk,C=model.info(p)
    theta=p[1]
    return dict(label=label,**metadata,N=model.N,k=np.exp(p[0]),theta=theta,alpha=model.alpha,quiet_weeks=int(np.sum(np.diff(model.cuts)[np.setdiff1d(np.arange(len(model.cuts)-1),model.nuisance)])),gap_parameters=len(model.nuisance),theta_se=st,theta_half95=1.96*st,log_growth_se=sg,k_relative_half95=1.96*sk,theta_k_corr=C[0,1]/(st*sk),growth=np.exp(model.growth(p)),theta_N_50=model.N*(1.96*st/(.5*theta))**2,theta_N_25=model.N*(1.96*st/(.25*theta))**2)

def main():
    old.table.initialize([],"entry-moment")
    baseline=pd.read_csv(HERE.parent/"multiwindow-information.csv")
    rows=[]; studies=[]; pairs=[]
    # Known truth: exact stress and main synthetic configurations. 
    for design_name,windows,N in (("stress",((2,5),(35,52)),12500),("main",((2,26),(35,52)),250000)):
        for alpha in (1.,1.18):
            t=np.arange(2,53)
            quiet=np.logical_or.reduce([(t>=a)&(t<=b) for a,b in windows])
            wave=(t>=27)&(t<=34)
            for age,k in ((1940,.002),(1950,.001)):
                for theta in (.1,.25,.5,1.,2.,4.):
                    m=Model(N,quiet,wave,4,len(t),alpha)
                    excess=np.array([3*k*(m.cuts[i+1]-m.cuts[i]) if wave[m.cuts[i]] and alpha!=1 else 0. for i in m.nuisance])
                    p=m.reference(k,theta,known=excess)
                    r=row(m,p,design_name,age=age)
                    rows.append(r)
                    if design_name=="stress" and theta==.5 and alpha==1.18:
                        grid=np.unique(np.r_[np.linspace(0,5,51),.5])
                        studies.extend(dict(label=design_name,age=age,theta_true=theta,theta=x,expected_LR=v,success=ok) for x,v,ok in m.expected_profile(p,grid))
    # Actual sample sizes and schedules, model-based references (not actual fitted theta).
    for kind,enroll,group,dose,dates,risk,dead in list(old.read_table8())+list(old.read_czech()):
        skip=old.table.K.DYNAMIC_HVE_SKIP_WEEKS
        time=np.arange(len(dates))
        quiet,_,_=old.table.K._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(),time)
        target=(dates.get_loc(old.table.REPORT)+1 if kind=="Table8" else len(dates))-skip
        anchor=old.table.K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE+1-skip
        wave=(dates>=old.table.WAVE["start_dt"])&(dates<=old.table.WAVE["end_dt"])
        quiet=quiet[skip:]; wave=np.asarray(wave[skip:]); risk=risk[skip:]; dead=dead[skip:]
        observed=np.r_[0.,np.cumsum(-np.log1p(-np.clip(dead/risk,0,.999)))]
        match=baseline[(baseline.source==kind)&(baseline.enrollment==enroll)&(baseline.group.astype(str)==group)&(baseline.dose==dose)]
        k=float(match.reference_rate.iloc[0])
        for alpha in (1.,1.18):
            m=Model(int(risk[0]),quiet,wave,anchor,target,alpha)
            for theta in (.1,.25,.5,1.,2.,4.):
                p=m.reference(k,theta,observed)
                rows.append(row(m,p,kind,enrollment=enroll,group=group,dose=dose))
    df=pd.DataFrame(rows)
    df.to_csv(HERE/"information-limits.csv",index=False)
    pd.DataFrame(studies).to_csv(HERE/"expected-profiles.csv",index=False)
    # Independent cohorts within age and independent synthetic strata.
    s=df[(df.label.isin(["stress","main"]))&(df.theta.isin([.5,4.]))]
    for (name,alpha,age),g in s.groupby(["label","alpha","age"]):
        se=np.sqrt(np.sum(g.log_growth_se**2))
        pairs.append(dict(label=name,alpha=alpha,age=age,log_ratio_se=se,lo=np.exp(-1.96*se),hi=np.exp(1.96*se)))
    real=df[(df.label.isin(["Table8","Czech"]))&(df.theta==.5)]
    for (name,alpha,enroll,group),g in real.groupby(["label","alpha","enrollment","group"]):
        for a,b in ((1,0),(2,0),(2,1),(3,0),(3,2)):
            if a not in g.dose.values or b not in g.dose.values: continue
            u=g[g.dose==a].iloc[0];v=g[g.dose==b].iloc[0]
            se=np.hypot(u.log_growth_se,v.log_growth_se);ratio=u.growth/v.growth
            pairs.append(dict(label=name,alpha=alpha,enrollment=enroll,group=group,pair=f"{a}/{b}",ratio=ratio,log_ratio_se=se,lo=ratio*np.exp(-1.96*se),hi=ratio*np.exp(1.96*se)))
    pd.DataFrame(pairs).to_csv(HERE/"ratio-limits.csv",index=False)
    print(df[(df.label.isin(["stress","main"]))&(df.theta.isin([.5,4.]))].round(4).to_string(index=False))
if __name__=="__main__": main()


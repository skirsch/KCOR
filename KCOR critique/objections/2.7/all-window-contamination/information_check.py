"""Locate the original experiment relative to the local information limit."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"low-frailty/theoretical-limits"))
from limits import Model
def main():
    pars=pd.read_csv(HERE/"parameter-summary.csv")
    # Original fit uses 53 configured quiet weeks (rebased 4..56 inclusive)
    # plus four early anchor weeks (0..3). Keep them in the information model.
    k=.00045616902672594684
    length=128;quiet=np.arange(length)<=56
    rows=[]
    for N in (50000,500000,5000000):
        for name,theta in (("theta_a",.5),("theta_b",4.)):
            m=Model(N,quiet,np.zeros(length,bool),5,length,1.)
            p=m.reference(k,theta,known=[0.])
            se,g,sk,C=m.info(p)
            # Local Gaussian approximation for an unconstrained parameter,
            # projected onto theta>=0. Not an exact expected estimator.
            z=theta/se
            predicted=theta*norm.cdf(z)+se*norm.pdf(z)
            h=pars[(pars["size"]==N)&(pars.fitter=="all_window")&(pars.level==0)&(pars.parameter==name)].iloc[0]
            rows.append(dict(N=N,parameter=name,theta_true=theta,theta_se=se,theta_half95=1.96*se,
                N_50=N*(1.96*se/(.5*theta))**2,N_25=N*(1.96*se/(.25*theta))**2,
                gaussian_clipped_mean=predicted,gaussian_clipped_zero_probability=norm.cdf(-z),
                observed_mean=h["mean"],observed_lo=h.lower95,observed_hi=h.upper95,observed_near_zero=h.near_zero/200,
                log_growth_se=g))
    df=pd.DataFrame(rows)
    df.to_csv(HERE/"information-comparison.csv",index=False)
    print(df.round(4).to_string(index=False))
    for N,g in df.groupby("N"):
        sd=np.linalg.norm(g.log_growth_se)
        print(N,"KCOR info interval (exact model benchmark including gap)",np.exp(np.array([-1,1])*1.96*sd))
if __name__=="__main__":main()

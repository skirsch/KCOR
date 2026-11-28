import argparse,sys,numpy as np
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument("candidate",type=Path);args=parser.parse_args()
sys.path.insert(0,str(args.candidate/"code"));import KCOR as K
g=.085/52;t=np.arange(52.)
for k in (.2,.001):
    B=k*np.expm1(g*np.arange(53))/g;h=np.diff(np.log1p(2*B)/2)
    for wave,alpha in [(None,None),((20,22),1.18)]:
        quiet=np.ones(52,bool) if wave is None else ~((t>=20)&(t<=22))
        v,d=K.fit_theta0_gompertz(h,t,quiet,4,g,wave=wave,alpha=alpha)
        print(k,alpha,v,d["status"],d["k_hit_bound"])
        assert d["k_hit_bound"]==(k==.2)
        if k==.2:assert d["status"]=="baseline_rate_bound" and v[1]>0

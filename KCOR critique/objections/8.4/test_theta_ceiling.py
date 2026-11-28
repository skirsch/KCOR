from pathlib import Path
import sys,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only')
import KCOR as K
def main():
    assert K.get_theta_degenerate_fit_config()['theta0_max']==500
    gamma=.085/52;t=np.arange(80,dtype=float);G=.001*np.expm1(gamma*np.arange(81))/gamma
    for truth in [99.95,120,450,499.95,550]:
        H=np.log1p(truth*G)/truth
        (k,theta),d=K.fit_theta0_gompertz(np.diff(H),t,np.ones(len(t),bool),4,gamma)
        assert d['success'] and theta>0 and theta==d['theta0_raw']
        if truth<=500:assert abs(theta-truth)<1e-5
        else:assert abs(theta-500)<1e-5
        assert d['theta_hit_bound']==(truth>=499.5)
        assert (d['status']=='theta_upper_bound')==(truth>=499.5)
        assert not d['insufficient_signal']
        print(truth,theta,d['status'],flush=True)
    # Alpha=1 marked-wave pathway uses the same constrained estimate and diagnostic.
    W=.05*np.clip((np.arange(81)-15)/10,0,1)
    truth=499.95;H=np.log1p(truth*(G+W))/truth;quiet=(t<15)|(t>=25)
    (k,theta),d=K.fit_theta0_gompertz(np.diff(H),t,quiet,4,gamma,deaths_arr=np.full(len(t),1000),wave=(15,24),alpha=1)
    assert d['success'] and abs(theta-truth)<.05 and d['theta_hit_bound'] and d['status']=='theta_upper_bound'
    print('marked alpha=1 wave',theta,d['status'],flush=True)
if __name__=='__main__':main()

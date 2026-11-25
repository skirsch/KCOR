"""Planning grid under stipulated multi-window schedules, not a proposed change."""
from limits import *

def main():
    rows=[]; thresholds=[]
    for k in (.0005,.001,.002):
      for theta in (.25,.5,1.,4.):
       for alpha in (1.,1.18):
        seen=set()
        for L in range(4,209):
            t=np.arange(2,35+L)
            quiet=(t<=5)|(t>=35);wave=(t>=27)&(t<=34)
            m=Model(100000,quiet,wave,4,len(t),alpha)
            extra=[3*k*(m.cuts[i+1]-m.cuts[i]) if wave[m.cuts[i]] and alpha!=1 else 0. for i in m.nuisance]
            p=m.reference(k,theta,known=extra)
            r=row(m,p,"planning",last_quiet_weeks=L)
            for N in (100000,500000,1000000):
                factor=np.sqrt(100000/N)
                z=dict(r,N=N)
                for col in ("theta_se","theta_half95","log_growth_se","k_relative_half95"): z[col]*=factor
                if L in (18,31,44,70,122,208): rows.append(z)
                for target in (.5,.25):
                    if (N,target) not in seen and z["theta_half95"]<=target*theta:
                        thresholds.append(dict(k=k,theta=theta,N=N,alpha=alpha,target_relative_halfwidth=target,last_quiet_weeks=L,total_quiet_weeks=L+4,modeled_weeks=len(t)))
                        seen.add((N,target))
        for N in (100000,500000,1000000):
            for target in (.5,.25):
                if (N,target) not in seen: thresholds.append(dict(k=k,theta=theta,N=N,alpha=alpha,target_relative_halfwidth=target,last_quiet_weeks=np.nan,total_quiet_weeks=np.nan,modeled_weeks=np.nan))
    pd.DataFrame(rows).to_csv(HERE/"planning-grid.csv",index=False)
    pd.DataFrame(thresholds).to_csv(HERE/"planning-thresholds.csv",index=False)
    print(pd.DataFrame(thresholds).query("theta==0.5 and alpha==1.18").to_string(index=False))
if __name__=="__main__": main()

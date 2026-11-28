from limits import *
import json
from scipy.stats import chi2
def main():
    f=pd.read_csv(HERE/"information-limits.csv")
    p=pd.read_csv(HERE/"expected-profiles.csv")
    for age,g in p.groupby("age"):
        print("profile",age,"range",g[g.expected_LR<=chi2.ppf(.95,1)].theta.agg(["min","max"]).to_dict(),"max grid LR",g.expected_LR.max())
    print(f[(f.label=="Czech")&(f.theta==.5)&(f.alpha==1.18)][["enrollment","group","dose","N","theta_half95","theta_N_50","theta_N_25"]].round(3).to_string(index=False))
    # Exact synthetic survival validation vs quadrature in the generator.
    study.initialize()
    probs,_=study.population("nonlinear_wave")
    t=np.arange(2,53)
    quiet=(t<=5)|(t>=35);wave=(t>=27)&(t<=34)
    for i,j in np.ndindex((2,2)):
        k=study.RATES[i];theta=study.THETA[i,j]
        m=Model(12500,quiet,wave,4,len(t),1.18)
        extra=[3*k*(m.cuts[x+1]-m.cuts[x]) if wave[m.cuts[x]] else 0. for x in m.nuisance]
        p=m.reference(k,theta,known=extra)
        S=np.r_[1.,np.cumsum(probs[i,j,::-1])[::-1][1:]]
        S=S[2:]/S[2]
        np.testing.assert_allclose(np.exp(m.logS(p)),S[m.cuts],rtol=2e-7,atol=2e-9)
    print("Exact nonlinear survival agrees with independent generator.")
    # Fixed expected weights in the synthetic design; weights random in the actual
    # estimator, but their first-order derivatives vanish at equal true ratios.
    weights=study.pooling_weights((probs*12500)[None])
    a=f[(f.label=="stress")&(f.alpha==1.18)&(f.theta.isin([.5,4.]))]
    s=np.array([np.sqrt(np.sum(g.log_growth_se**2)) for _,g in a.groupby("age")])
    se=np.linalg.norm(weights*s)
    draw=np.load(HERE.parent/"all-window-joint/multistart-stress.npz")["points"]
    y=draw[:,:,1,0];low,high=np.quantile(y[:,1:],[.025,.975],axis=1)
    print("pooled weights",weights,"theory SE",se,"ratio interval",np.exp(np.array([-1,1])*1.96*se),"empirical point sd",np.std(np.log(y[:,0]),ddof=1),"median bootstrap sd",np.median(np.std(np.log(y[:,1:]),axis=1,ddof=1)),"median width",np.median(high-low))
    pd.DataFrame([dict(theory_log_se=se,theory_lower=np.exp(-1.96*se),theory_upper=np.exp(1.96*se),experimental_log_point_sd=np.std(np.log(y[:,0]),ddof=1),experimental_median_bootstrap_log_sd=np.median(np.std(np.log(y[:,1:]),axis=1,ddof=1)),experimental_median_bootstrap_width=np.median(high-low))]).to_csv(HERE/"synthetic-pooled-comparison.csv",index=False)
    summary=[]
    for i,j in np.ndindex((2,2)):
        true=study.THETA[i,j];r=a[(a.age==study.STRATA[i])&(a.theta==true)].iloc[0]
        v=draw[:,:,1,6+2*i+j]
        print("theta",study.STRATA[i],true,"theory SE",r.theta_se,"actual across datasets SD",v[:,0].std(ddof=1),"median bootstrap interval",np.median(np.quantile(v[:,1:],[.025,.975],axis=1),axis=1))
        lo,hi=np.median(np.quantile(v[:,1:],[.025,.975],axis=1),axis=1)
        summary.append(dict(age=study.STRATA[i],theta_true=true,theory_se=r.theta_se,experimental_point_sd=v[:,0].std(ddof=1),median_bootstrap_lo=lo,median_bootstrap_hi=hi))
    pd.DataFrame(summary).to_csv(HERE/"synthetic-frailty-comparison.csv",index=False)
    # Table 8 actual model-reference points based on the experimental parameter fits.
    old.table.initialize([],"entry-moment")
    pars=pd.read_csv(HERE.parent/"all-window-joint/table-parameters.csv")
    rows=[]
    for _,enroll,group,dose,dates,risk,dead in old.read_table8():
        skip=old.table.K.DYNAMIC_HVE_SKIP_WEEKS
        quiet,_,_=old.table.K._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(),np.arange(len(dates)))
        wave=np.asarray((dates>=old.table.WAVE["start_dt"])&(dates<=old.table.WAVE["end_dt"]))
        target=dates.get_loc(old.table.REPORT)+1-skip;anchor=old.table.K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE+1-skip
        risk=risk[skip:];dead=dead[skip:]
        observed=np.r_[0.,np.cumsum(-np.log1p(-dead/risk))]
        x=pars[(pars.enrollment==enroll)&(pars.group==old.GROUPS.index(group))&(pars.pseudo_dose==dose)].iloc[0]
        m=Model(int(risk[0]),quiet[skip:],wave[skip:],anchor,target,1.18)
        # At theta=0 use an explicit nearby interior reference, not a boundary CRLB.
        theta=max(.1,x.theta)
        p=m.reference(x.k,theta,observed)
        r=row(m,p,"Table8 fitted reference",enrollment=enroll,group=group,dose=dose,actual_theta=x.theta,actual_lo=x.theta_lo,actual_hi=x.theta_hi)
        rows.append(r)
    r=pd.DataFrame(rows);r.to_csv(HERE/"table8-fitted-reference.csv",index=False)
    print(r[["enrollment","group","dose","theta","theta_half95","actual_theta","actual_lo","actual_hi"]].round(3).to_string(index=False))
if __name__=="__main__":main()


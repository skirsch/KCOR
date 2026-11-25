"""Noiseless counterfactual input substitutions, not an operational estimator."""
import replay as r
import numpy as np
import pandas as pd
def run(mode):
    s=r.study
    if mode=="stress":s.POPULATION=50000;s.WINDOWS=((2,5),(35,52));s.SEED=20261126
    else:s.POPULATION=1000000;s.WINDOWS=((2,26),(35,52));s.SEED=20261124
    s.population.cache_clear();s.initialize()
    fit=r.candidate(s.model,"multistart")
    old=s.model.apply_nph_correction_post_inversion
    exact=r.make_correction(s.model)
    prob,_=s.population("nonlinear_wave");c=prob*s.POPULATION/4
    dead,alive=s.arrays(c)
    h=s.model.hazard_from_mr(dead/(alive+s.model.EPS));h[...,:s.skip]=0
    observed=np.cumsum(h,axis=-1)
    keep=(s.TIME<=s.WAVE[1])|(s.TIME==s.TIME[-1])
    weights=s.pooling_weights(c[None])
    rows=[];ratios=[]
    for k_true in (False,True):
        for theta_true in (False,True):
            for label,correction in (("approximate",old),("weekly_survival",exact)):
                growth=np.empty((2,2))
                for age,dose in np.ndindex(s.THETA.shape):
                    (kf,thf),d=fit(h[age,dose],s.TIME-s.skip,s.quiet,s.config["k_anchor_weeks"],s.config["gamma_per_week"],deaths_arr=dead[age,dose])
                    k=s.RATES[age] if k_true else kf;theta=s.THETA[age,dose] if theta_true else thf
                    H=observed[age,dose,keep];raw=s.model.invert_gamma_frailty(H,theta)
                    curve,diag=correction(raw,s.TIME[keep],s.DATES.to_numpy()[keep],theta,1.18,s.DATES[s.WAVE[0]],s.DATES[s.WAVE[1]],k,s.config["gamma_per_week"],observed_cumhaz=H)
                    growth[age,dose]=curve[-1]/curve[s.anchor]
                    rows.append(dict(mode=mode,method=label,k_true=k_true,theta_true=theta_true,age=s.STRATA[age],dose=dose,k_used=k,theta_used=theta,k_fitted=kf,theta_fitted=thf,true_k=s.RATES[age],true_theta=s.THETA[age,dose],wave_true=3*s.RATES[age]*(s.WAVE[1]-s.WAVE[0]+1),wave_estimate=diag.get("post_wave_exposure"),growth=growth[age,dose]))
                ratio=growth[:,1]/growth[:,0];pooled=np.exp(np.average(np.log(ratio),weights=weights))
                ratios.append(dict(mode=mode,method=label,k_true=k_true,theta_true=theta_true,pooled=pooled,age1940=ratio[0],age1950=ratio[1]))
    return rows,ratios
if __name__=="__main__":
    rows=[];ratios=[]
    for mode in ("main","stress"):
        a,b=run(mode);rows+=a;ratios+=b
    pd.DataFrame(rows).to_csv(r.HERE/"noiseless-parameters.csv",index=False)
    df=pd.DataFrame(ratios);df.to_csv(r.HERE/"noiseless-ratios.csv",index=False);print(df.to_string(index=False))

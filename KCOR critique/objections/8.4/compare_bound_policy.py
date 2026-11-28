"""Isolate a policy-only change: keep the fitted constrained theta, flag proximity to the bound."""
import os,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(exist_ok=True)
os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only')
sys.path.insert(0,str(ROOT/'code'))
import KCOR as K
BANDS=[1975,1965,1955,1945,1935,1925,-2]
def growth(H,theta,anchor):
    v=K.invert_gamma_frailty(H,theta)
    return float(v[-1]/v[anchor])
def exact():
    gamma=.085/52; t=np.arange(80,dtype=float)
    G=.001*np.expm1(gamma*np.arange(81))/gamma
    configs=K.get_theta_degenerate_fit_config();orig=K._DATASET_THETA_DEGENERATE_CFG_CACHE.copy()
    rows=[];profiles=[]
    try:
        for truth in [99.8,99.95,100.,120.]:
            H=np.log1p(truth*G)/truth; h=np.diff(H)
            def run(cap):
                K._DATASET_THETA_DEGENERATE_CFG_CACHE={**orig,'theta0_max':cap}
                (k,theta),diag=K.fit_theta0_gompertz(h,t,np.ones(len(t),bool),4,gamma)
                return float(k),float(theta),float(diag['theta0_raw']),diag
            k,applied,raw,diag=run(100.)
            k2,expanded,raw2,diag2=run(200.)
            base=G[-1]/G[4]
            v=dict(theta_true=truth,k_fit=k,raw_at100=raw,applied_at100=applied,bound_flag=bool(diag['theta_hit_bound']),
                rmse=float(diag['rmse_hazard']),current_kcor=growth(H[1:],applied,3)/base,
                retain_kcor=growth(H[1:],raw,3)/base,raw_at200=raw2,expanded_kcor=growth(H[1:],expanded,3)/base)
            rows.append(v)
            for theta in [0.,50.,90.,99.8,99.95,100.,100.05,110.,120.,140.,200.]:
                def loss(logk):
                    g=np.exp(logk)*np.expm1(gamma*np.arange(81))/gamma
                    pred=np.diff(np.log1p(theta*g)/theta if theta else g)
                    return float(np.mean(((h-pred)/np.mean(h))**2))
                fit=minimize_scalar(loss,bounds=(np.log(1e-8),np.log(.1)),method='bounded',options={'xatol':1e-14})
                profiles.append(dict(theta_true=truth,theta_profile=theta,k_profile=float(np.exp(fit.x)),scaled_mse=float(fit.fun)))
    finally: K._DATASET_THETA_DEGENERATE_CFG_CACHE=orig
    pd.DataFrame(rows).to_csv(OUT/'exact-tests.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'exact-profiles.csv',index=False)
    print('Exact policy tests',pd.DataFrame(rows).to_string(index=False),flush=True)
def empirical():
    src=ROOT/'data/Czech/KCOR_CMR.xlsx'
    raw=pd.read_excel(src,sheet_name='2021_24',usecols=['YearOfBirth','Dose','ISOweekDied','Alive','Dead'])
    raw['week']=raw.ISOweekDied.astype(str).str.replace('-','',regex=False).str[:6].astype(int)
    raw=raw[raw.Dose.isin([0,2])&raw.week.between(202124,202416)]
    groups=raw.groupby(['YearOfBirth','Dose','week'],as_index=False)[['Alive','Dead']].sum()
    weeks=np.sort(groups.week.unique());assert len(weeks)==149
    rng=np.random.default_rng(20261126);draws=999
    keys=[(yob,dose) for dose in (0,2) for yob in (1975,1980)]
    keys += [(int(yob),int(dose)) for (yob,dose),_ in groups.groupby(['YearOfBirth','Dose']) if (yob>0 or yob==-1) and (yob,dose) not in keys]
    observed=np.zeros((len(BANDS),2,2,len(weeks)),dtype=np.int64)
    sampled=np.zeros((len(BANDS),draws,2,2,len(weeks)),dtype=np.int64)
    for yob,dose in keys:
        x=groups[(groups.YearOfBirth==yob)&(groups.Dose==dose)].sort_values('week')
        assert np.array_equal(x.week,weeks)
        risk,dead=x.Alive.to_numpy(int),x.Dead.to_numpy(int)
        assert np.all(risk[:-1]-dead[:-1]==risk[1:])
        cells=np.r_[dead,risk[-1]-dead[-1]]
        counts=rng.multinomial(risk[0],cells/risk[0],size=draws) if risk[0] else np.zeros((draws,len(cells)),dtype=np.int64)
        rd=np.cumsum(counts[:,::-1],axis=1)[:,::-1][:,:-1]
        targets=[BANDS.index(-2)]
        if yob>0:
            band=(yob-5)//10*10+5
            if band in BANDS:targets.append(BANDS.index(band))
        j=(0,2).index(dose)
        for i in targets:
            observed[i,j]+=[risk,dead];sampled[i,:,j,0]+=rd;sampled[i,:,j,1]+=counts[:,:-1]
    saved=pd.read_csv(ROOT/'analysis/solver_scaling/s6-all-window-draws.csv')
    summary=json.loads((ROOT/'analysis/solver_scaling/s6-all-window.json').read_text(encoding='utf8'))
    skip=K.DYNAMIC_HVE_SKIP_WEEKS;anchor=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    def H(counts):
        risk,dead=counts
        return np.cumsum(np.where(np.arange(len(risk))>=skip,K.hazard_from_mr(dead/risk),0.))
    out=[];detail=[]
    for i,band in enumerate(BANDS):
        g=saved[saved.birth_band.eq(band)].sort_values('draw')
        assert len(g)==draws and np.array_equal(g.draw,np.arange(draws))
        current=[];retain=[]
        for n,row in enumerate(g.itertuples()):
            ha,hb=[H(c) for c in sampled[i,n]]
            ca,cb=row.theta0,row.theta2
            ra=row.theta0_raw if row.bound0 else ca;rb=row.theta2_raw if row.bound2 else cb
            point=growth(hb,cb,anchor)/growth(ha,ca,anchor)
            alt=growth(hb,rb,anchor)/growth(ha,ra,anchor)
            current.append(point);retain.append(alt)
            detail.append(dict(birth_band=band,draw=n,current=point,retain=alt,bound0=row.bound0,bound2=row.bound2))
        assert np.max(np.abs(np.array(current)-g.kcor.to_numpy()))<1e-7,(band,np.max(np.abs(np.array(current)-g.kcor.to_numpy())))
        ref=next(v for v in summary['ratios'] if v['birth_band']==band)
        p=[next(v for v in summary['params'] if v['birth_band']==band and v['dose']==d) for d in (0,2)]
        ha,hb=[H(c) for c in observed[i]]
        altpoint=growth(hb,p[1]['point_raw'],anchor)/growth(ha,p[0]['point_raw'],anchor)
        vals=dict(birth_band=band,point_current=ref['point'],point_retain=altpoint,
            current_interval=np.percentile(current,[2.5,50,97.5]).tolist(),retain_interval=np.percentile(retain,[2.5,50,97.5]).tolist(),
            bound_draws0=int(g.bound0.sum()),bound_draws2=int(g.bound2.sum()),changed_draws=int(np.sum(np.abs(np.array(retain)-current)>1e-10)))
        out.append(vals);print(vals,flush=True)
    pd.DataFrame(detail).to_csv(OUT/'matched-draws.csv',index=False)
    (OUT/'s6-policy-comparison.json').write_text(json.dumps(out,indent=2),encoding='utf8')
if __name__=='__main__':exact();empirical()

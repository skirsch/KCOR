"""Matched ceiling sensitivity. Existing 100-cap draws are the unchanged reference."""
import os,sys,json,argparse
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.stats import t as student_t
BANDS=[1975,1965,1955,1945,1935,1925,-2]
def init(root,weeks):
    global K,CFG,WEEKS,ORIG
    os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only')
    sys.path.insert(0,str(Path(root)/'code'))
    import KCOR as K
    CFG=K.get_gompertz_theta_fit_config();WEEKS=weeks;ORIG=K._DATASET_THETA_DEGENERATE_CFG_CACHE.copy()
def result(counts,cap,quiet=None,t=None,anchor=None):
    risk,dead=counts;skip=K.DYNAMIC_HVE_SKIP_WEEKS
    if t is None:t=np.arange(len(risk),dtype=float)
    if quiet is None:quiet=K._build_theta_quiet_mask(WEEKS,t)[0]
    if anchor is None:anchor=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    K._DATASET_THETA_DEGENERATE_CFG_CACHE={**ORIG,'theta0_max':cap}
    eff=np.where(t>=skip,K.hazard_from_mr(dead/risk),0.);H=np.cumsum(eff)
    (k,theta),d=K.fit_theta0_gompertz(eff,t-skip,quiet,CFG['k_anchor_weeks'],CFG['gamma_per_week'],deaths_arr=dead,risk_arr=risk)
    applied=theta if d.get('success') and np.isfinite(theta) and theta>=0 else 0.
    applied,_,_=K._apply_theta_degenerate_guard(applied,K.get_theta_degenerate_fit_config())
    raw=float(d.get('theta0_raw',theta));bound=bool(d.get('theta_hit_bound',False))
    retained=raw if bound and d.get('success') and np.isfinite(raw) else applied
    def grow(th):
        v=K.invert_gamma_frailty(H,th);return float(v[-1]/v[anchor])
    return dict(k=float(k),raw=raw,applied=float(applied),bound=bound,status=d['status'],growth=grow(applied),retain_growth=grow(retained))
def worker(task):
    band,draw,counts=task
    rows=[]
    for cap in [200,500]:
        a,b=[result(c,cap) for c in counts]
        rows.append(dict(birth_band=band,draw=draw,cap=cap,theta0_raw=a['raw'],theta2_raw=b['raw'],theta0=a['applied'],theta2=b['applied'],bound0=a['bound'],bound2=b['bound'],status0=a['status'],status2=b['status'],kcor=b['growth']/a['growth'],retain=b['retain_growth']/a['retain_growth']))
    return rows
def samples(root):
    raw=pd.read_excel(root/'data/Czech/KCOR_CMR.xlsx',sheet_name='2021_24',usecols=['YearOfBirth','Dose','ISOweekDied','Alive','Dead'])
    raw['week']=raw.ISOweekDied.astype(str).str.replace('-','',regex=False).str[:6].astype(int)
    raw=raw[raw.Dose.isin([0,2])&raw.week.between(202124,202416)]
    groups=raw.groupby(['YearOfBirth','Dose','week'],as_index=False)[['Alive','Dead']].sum()
    weeks=np.sort(groups.week.unique());assert len(weeks)==149
    rng=np.random.default_rng(20261126);draws=999
    keys=[(yob,dose) for dose in (0,2) for yob in (1975,1980)]
    keys += [(int(yob),int(dose)) for (yob,dose),_ in groups.groupby(['YearOfBirth','Dose']) if (yob>0 or yob==-1) and (yob,dose) not in keys]
    obs=np.zeros((len(BANDS),2,2,len(weeks)),dtype=np.int64)
    sample=np.zeros((len(BANDS),draws,2,2,len(weeks)),dtype=np.int64)
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
            obs[i,j]+=[risk,dead];sample[i,:,j,0]+=rd;sample[i,:,j,1]+=counts[:,:-1]
    return weeks,obs,sample
def exact(out):
    gamma=.085/52;t=np.arange(80,dtype=float);G=.001*np.expm1(gamma*np.arange(81))/gamma
    rows=[]
    for truth in [.5,50,99.95,120,250,450,550]:
        H=np.log1p(truth*G)/truth;h=np.diff(H)
        for cap in [100,200,500]:
            K._DATASET_THETA_DEGENERATE_CFG_CACHE={**ORIG,'theta0_max':cap}
            (k,theta),diag=K.fit_theta0_gompertz(h,t,np.ones(len(t),bool),4,gamma)
            raw=float(diag['theta0_raw']);base=G[-1]/G[4]
            def g(th):
                v=K.invert_gamma_frailty(H[1:],th);return float(v[-1]/v[3]/base)
            rows.append(dict(truth=truth,cap=cap,raw=raw,applied=float(theta),current_kcor=g(theta),retain_kcor=g(raw),bound=bool(diag['theta_hit_bound']),rmse=diag['rmse_hazard']))
    pd.DataFrame(rows).to_csv(out/'ceiling-exact.csv',index=False)
def ci(a):
    a=np.asarray(a,float);m=a.mean();h=student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a));return [float(m),float(m-h),float(m+h)]
def synthetic(root,out):
    sys.path.insert(0,str(root/'test/quiet_window_contamination'))
    import run_contamination_test as study
    gamma=.085/52;k=.00045616902672594684;skip=K.DYNAMIC_HVE_SKIP_WEEKS
    weeks,rebased,quiet=study.build_time_arrays(study.N_WEEKS,skip)
    rows=[]
    for size in (50000,500000,5000000):
        for rep in range(200):
            rng=np.random.default_rng(np.random.SeedSequence([10042,2,0,0,rep]))
            histories=[study.simulate_one_cohort(rng,size,study.N_WEEKS,k,gamma,theta,skip,None) for theta in (.5,4.)]
            for cap in [200,500]:
                a,b=[result(c,cap,quiet=quiet,t=weeks,anchor=skip+4) for c in histories]
                rows.append(dict(size=size,rep=rep,cap=cap,theta_a=a['raw'],theta_b=b['raw'],bound_a=a['bound'],bound_b=b['bound'],kcor=a['growth']/b['growth'],retain=a['retain_growth']/b['retain_growth']))
        print('SYNTHETIC completed',size,flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'ceiling-synthetic.csv',index=False)
    summary=[]
    for (size,cap),g in f.groupby(['size','cap']):
        summary.append(dict(size=int(size),cap=int(cap),kcor_mean_ci=ci(g.kcor),retain_mean_ci=ci(g.retain),theta_a_mean_ci=ci(g.theta_a),theta_b_mean_ci=ci(g.theta_b),bound_a=int(g.bound_a.sum()),bound_b=int(g.bound_b.sum())))
    (out/'ceiling-synthetic-summary.json').write_text(json.dumps(summary,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);p.add_argument('--workers',type=int,default=8);p.add_argument('--synthetic',action='store_true');a=p.parse_args()
    if a.synthetic:
        init(a.root,np.array([]));synthetic(a.root,a.out);return
    weeks,obs,sample=samples(a.root);init(a.root,weeks);exact(a.out)
    points=[]
    for i,band in enumerate(BANDS):points.extend(worker((band,-1,obs[i])))
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(a.root,weeks)) as pool:
        for n,chunk in enumerate(pool.map(worker,((band,j,sample[i,j]) for i,band in enumerate(BANDS) for j in range(999)),chunksize=16),1):
            rows.extend(chunk)
            if n%999==0:
                pd.DataFrame(rows).to_csv(a.out/'ceiling-s6-draws.csv',index=False)
                print('S6 completed',n,'of',6993,flush=True)
    f=pd.DataFrame(rows);p=pd.DataFrame(points);p.to_csv(a.out/'ceiling-s6-points.csv',index=False)
    result_rows=[]
    for (band,cap),g in f.groupby(['birth_band','cap']):
        v=p[(p.birth_band==band)&(p.cap==cap)].iloc[0]
        item=dict(birth_band=int(band),cap=int(cap),point=float(v.kcor),retain_point=float(v.retain),interval=np.percentile(g.kcor,[2.5,50,97.5]).tolist(),retain_interval=np.percentile(g.retain,[2.5,50,97.5]).tolist(),theta0_point=float(v.theta0_raw),theta2_point=float(v.theta2_raw),theta0_interval=np.percentile(g.theta0_raw,[2.5,50,97.5]).tolist(),theta2_interval=np.percentile(g.theta2_raw,[2.5,50,97.5]).tolist(),bound0=int(g.bound0.sum()),bound2=int(g.bound2.sum()),failed0=int((~g.status0.isin(['ok','insufficient_signal','weak_identifiability','baseline_rate_bound'])).sum()),failed2=int((~g.status2.isin(['ok','insufficient_signal','weak_identifiability','baseline_rate_bound'])).sum()))
        result_rows.append(item)
    (a.out/'ceiling-s6-summary.json').write_text(json.dumps(result_rows,indent=2));print('DONE',len(f),flush=True)
if __name__=='__main__':main()

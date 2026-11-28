"""Matched five-structure frailty-misspecification validation of current KCOR.

Known fixed-frailty null, identical baseline, anchored endpoint benchmark=1.
Independent person histories except explicit gamma-shared clusters.
"""
import os,sys,json,argparse,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.special import roots_hermitenorm
from scipy.stats import t as student_t,norm
SCENARIOS=['gamma','lognormal','two_point','bimodal','correlated_gamma']
N=100_000;W=120;K0=.002;GAMMA=.085/52;THETA=[1.,.3];CLUSTER=20;SKIP=2;ANCHOR=5;SEED=20261126
T=np.arange(W,dtype=float);G=K0*np.expm1(GAMMA*np.arange(W+1))/GAMMA;QUIET=(T>=20)&(T<=80)
def init(root):
    global K
    os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only')
    sys.path.insert(0,str(Path(root)/'code'))
    import KCOR as K
    assert K.get_theta_degenerate_fit_config()['theta0_max']==500
def survival(scenario,theta):
    if scenario in ('gamma','correlated_gamma'):return (1+theta*G)**(-1/theta)
    if scenario=='lognormal':
        x,w=roots_hermitenorm(192);v=np.log1p(theta);z=np.exp(-v/2+np.sqrt(v)*x)
        return (w/np.sqrt(2*np.pi))@np.exp(-z[:,None]*G[None,:])
    a=.2;eps=.02 if scenario=='bimodal' else 0.
    between=(theta-eps)/(1+eps);b=1+between/(1-a);p=(b-1)/(b-a)
    if eps:return p*(1+eps*a*G)**(-1/eps)+(1-p)*(1+eps*b*G)**(-1/eps)
    return p*np.exp(-a*G)+(1-p)*np.exp(-b*G)
def probabilities(scenario,theta):
    s=survival(scenario,theta);p=np.r_[-np.diff(s),s[-1]]
    assert np.min(p)>=0 and abs(p.sum()-1)<1e-10
    return p/p.sum()
def fit(cells,point=False):
    risk=np.cumsum(cells[::-1])[::-1][:-1];dead=cells[:-1]
    eff=np.where(T>=SKIP,K.hazard_from_mr(dead/risk),0.)
    (k,th),d=K.fit_theta0_gompertz(eff,T-SKIP,QUIET,4,GAMMA,deaths_arr=dead,risk_arr=risk if point else None)
    theta=th if d.get('success') and np.isfinite(th) and th>=0 else 0.
    theta,_,_=K._apply_theta_degenerate_guard(theta,K.get_theta_degenerate_fit_config())
    H=K.invert_gamma_frailty(np.cumsum(eff),theta)
    growth=float(H[-1]/H[ANCHOR]) if H[ANCHOR]>0 and np.isfinite(H[-1]) else np.nan
    return dict(growth=growth,theta=float(d.get('theta0_raw',th)),applied=float(theta),status=d['status'],failed=not d.get('success'),flag=d['status']!='ok',residual=bool(d.get('sampling_flag',False)))
def pair(a,b,point=False):
    x,y=fit(a,point),fit(b,point)
    v=x['growth']/y['growth'] if np.isfinite(y['growth']) and y['growth']>0 else np.nan
    return v,x,y
def cluster_cells(rng,theta):
    nc=N//CLUSTER;z=rng.gamma(1/theta,theta,size=nc)
    # Each cluster shares a latent Z. Conditional individual failures are independent.
    thresholds=rng.exponential(size=(nc,CLUSTER))/z[:,None]
    week=np.searchsorted(G[1:],thresholds,side='right')
    ids=np.arange(nc)[:,None]*(W+1)+week
    cells=np.bincount(ids.ravel(),minlength=nc*(W+1)).reshape(nc,W+1)
    assert cells.sum()==N
    return cells
def worker(task):
    scenario,rep,draws=task
    # Separate reproducible generators for independent datasets and resampling.
    idx=SCENARIOS.index(scenario)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,idx,rep,0]))
    rb=np.random.default_rng(np.random.SeedSequence([SEED,idx,rep,1]))
    if scenario=='correlated_gamma':
        clusters=[cluster_cells(rng,th) for th in THETA]
        orig=[v.sum(axis=0) for v in clusters]
        # Matched cluster and person resampling on the identical observed dataset.
        modes={'cluster':[],'person':[]}
        for v,c in zip(clusters,orig):
            weights=rb.multinomial(len(v),np.full(len(v),1/len(v)),size=draws)
            modes['cluster'].append((weights.astype(float)@v.astype(float)).astype(np.int64))
            modes['person'].append(rb.multinomial(N,c/N,size=draws))
    else:
        orig=[rng.multinomial(N,probabilities(scenario,th)) for th in THETA]
        modes={'person':[rb.multinomial(N,c/N,size=draws) for c in orig]}
    point,a,b=pair(*orig,point=True);out=[]
    for mode,samples in modes.items():
        vals=[];flags=0;failures=0
        for i in range(draws):
            v,x,y=pair(samples[0][i],samples[1][i]);vals.append(v);flags+=x['flag'] or y['flag'];failures+=x['failed'] or y['failed']
        vals=np.array(vals);valid=int(np.isfinite(vals).sum())
        lo,hi=np.percentile(vals,[2.5,97.5]) if valid==draws and np.isfinite(point) else (np.nan,np.nan)
        out.append(dict(scenario=scenario,rep=rep,bootstrap=mode,point=point,lo=lo,hi=hi,valid_draws=valid,draws=draws,coverage=bool(lo<=1<=hi),flag=bool(a['flag'] or b['flag']),residual=bool(a['residual'] or b['residual']),failed=bool(a['failed'] or b['failed']),flagged_draws=int(flags),failed_draws=int(failures),theta_a=a['theta'],theta_b=b['theta'],status_a=a['status'],status_b=b['status'],deaths_a=int(orig[0][:-1].sum()),deaths_b=int(orig[1][:-1].sum())))
    return out
def wilson(k,n):
    z=norm.ppf(.975);p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]
def mean_ci(x):
    x=np.asarray(x,float);m=x.mean();h=student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan
    return [float(m),float(m-h),float(m+h)]
def summarize(f):
    rows=[]
    for (scenario,mode),g in f.groupby(['scenario','bootstrap'],sort=False):
        n=len(g);v=g.point.dropna();available=g.lo.notna();covered=int(g.coverage.sum());flags=int(g.flag.sum());residual=int(g.residual.sum());failed=int(g.failed.sum())
        # If a full interval is unavailable count it as noncoverage in all-attempted rate.
        rows.append(dict(scenario=scenario,bootstrap=mode,datasets=n,point_available=len(v),intervals_available=int(available.sum()),bias_mean_ci=mean_ci(v-1),mean_kcor_ci=mean_ci(v),variance=float(v.var(ddof=1)),sd=float(v.std(ddof=1)),coverage=covered,coverage_ci=wilson(covered,n),fit_flag=flags,fit_flag_ci=wilson(flags,n),residual_flag=residual,residual_flag_ci=wilson(residual,n),failed=failed,failed_ci=wilson(failed,n),median_interval_width=float((g.hi-g.lo).median()),theta_a_mean_ci=mean_ci(g.theta_a.dropna()),theta_b_mean_ci=mean_ci(g.theta_b.dropna()),flagged_bootstrap_draws=int(g.flagged_draws.sum()),failed_bootstrap_draws=int(g.failed_draws.sum())))
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--datasets',type=int,default=100);p.add_argument('--bootstraps',type=int,default=199);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);init(a.root)
    exact=[]
    for scenario in SCENARIOS:
        vals=[N*probabilities(scenario,theta) for theta in THETA]
        est,x,y=pair(*vals,point=False);exact.append(dict(scenario=scenario,point=est,theta_a=x['theta'],theta_b=y['theta'],status_a=x['status'],status_b=y['status']))
    (a.output/'expected-counts.json').write_text(json.dumps(exact,indent=2))
    print('EXPECTED COUNTS',exact,flush=True)
    tasks=[(s,i,a.bootstraps) for i in range(a.datasets) for s in SCENARIOS]
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(a.root,)) as pool:
        for i,chunk in enumerate(pool.map(worker,tasks,chunksize=1),1):
            rows.extend(chunk)
            if i%5==0 or i==len(tasks):
                f=pd.DataFrame(rows);f.to_csv(a.output/'replicates.csv',index=False)
                print('COMPLETE',i,'of',len(tasks),'elapsed',round(time.monotonic()-start),'sec',flush=True)
    f=pd.DataFrame(rows);assert len(f)==a.datasets*6
    summary=summarize(f)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'design.json').write_text(json.dumps(dict(scenarios=SCENARIOS,people_per_cohort=N,weeks=W,baseline_weekly_k=K0,gamma=GAMMA,variances=THETA,skip=SKIP,anchor_index=ANCHOR,quiet_weeks=[20,80],cluster_size=CLUSTER,seed=SEED,datasets=a.datasets,bootstraps=a.bootstraps,truth=1),indent=2))
    print('DONE',json.dumps(summary),flush=True)
if __name__=='__main__':main()

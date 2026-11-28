"""Re-evaluate all forced-alpha diagnostics at the exact archived fitted parameters.

No optimization is repeated. The archived matched death histories are regenerated
from the original seed, and the current production Profile prediction and sampling
screen are evaluated at each fitted parameter pair. Aggregate original success
and insufficient-signal flags must be zero, and eligibility is independently
checked. A fresh-refit subset is compared separately.
"""
from pathlib import Path
import argparse,sys,os,json,time
from concurrent.futures import ProcessPoolExecutor
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/table8'))
import table8_bootstrap as T

def initialize(pop,params):
    global PARAMS;PARAMS=params
    os.environ['KCOR_NPH_MODE']='forced_alpha';T.initialize(pop,'revised')

def one(task):
    i,draw=task;dates,time,quiet,keep,c=T.DATA[i];counts=c.copy()
    if draw:counts=np.random.default_rng([T.SEED,i,draw]).multinomial(int(c.sum()),c.ravel()/c.sum()).reshape(c.shape)
    counts[2]+=counts[0]+counts[1];alive=np.cumsum(counts[:,:,::-1],axis=2)[:,:,::-1][:,:,:-1];dead=counts[:,:,:-1]
    K=T.K;skip=K.DYNAMIC_HVE_SKIP_WEEKS;t=time-skip;cfg=T.CONFIG;gamma=cfg['gamma_per_week'];anchor=cfg['k_anchor_weeks'];guard=T.GUARD
    iso=pd.DatetimeIndex(dates).strftime('%G%V').astype(int).to_numpy();a=T.WAVE['start_dt'].isocalendar();b=T.WAVE['end_dt'].isocalendar();active=(iso>=a.year*100+a.week)&(iso<=b.year*100+b.week)&(t>=0);assert active.any();wave=(float(t[active][0]),float(t[active][-1]));alpha=T.WAVE['apply_alpha'];Profile=K._survivor_module().Profile
    group=np.zeros((3,3,len(T.DIAGNOSTIC_CATEGORIES)),bool)
    for y,d in np.ndindex((3,3)):
        k,theta,raw=PARAMS[i,draw,3*y+d];assert np.isfinite([k,theta,raw]).all() and theta==raw
        h=K.hazard_from_mr(np.clip(dead[y,d]/(alive[y,d]+K.EPS),0,.999));h=np.where(time>=skip,h,0.);valid=np.isfinite(h)&(t>=0);marked=(t>=wave[0])&(t<=wave[1]);eligible=quiet&valid&~marked
        p=Profile(h,t,eligible,anchor,gamma,wave);assert p.fit.sum()>=3 and eligible.sum()>=3 and dead[y,d,eligible].sum()>=guard.get('min_quiet_deaths',30)
        pred,gaps,_,_,span=p.predict(k,theta,alpha,return_exposure_span=True);target=p.h[p.fit];rel=float(np.sqrt(np.mean(((target-pred)/(target+1e-12))**2)));mask=np.zeros_like(quiet);mask[p.idx]=p.fit
        diag=dict(success=True,insufficient_signal=False,insufficient_deaths=False,fit_mask_theta=mask,fitted_weekly_hazard=pred,theta_hit_bound=theta>=guard['theta0_max']*.999,k_hit_bound=k<=1e-8*1.001 or k>=.1*.999,identifiability_flag=bool(p.fit.sum()<6 or (np.isfinite(span) and span<1e-4)),legacy_relative_error_flag=rel>float(guard.get('bad_fit_relrmse_threshold',1e5)),delta_negative_clamped=any(x<=0 for x in gaps))
        K._sampling_diagnostics(dead[y,d],alive[y,d],t,diag);group[y,d]=T.fit_diagnostic_flags(diag)
    # All archived subgroup weights and anchor-period death counts are positive.
    pair=np.stack([group[:,[n,d],:].any(axis=(0,1)) for n,d in T.PAIRS])
    return i,draw,group,pair

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--workers',type=int,default=6);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    reference=ROOT/'analysis/ceiling500/table8/forced_alpha-draws.npz';a=np.load(reference);vals=a['points'];flags=a['flags'];params=a['parameters'];assert np.all(flags==0) and vals.shape==(2,1000,3)
    pop=[T.read_population(ROOT/'analysis/table8/input.xlsx',s) for s in T.SHEETS];group=np.zeros((2,1000,3,3,len(T.DIAGNOSTIC_CATEGORIES)),bool);pair=np.zeros((2,1000,3,len(T.DIAGNOSTIC_CATEGORIES)),bool);start=time.monotonic()
    tasks=[(i,j) for j in range(1000) for i in range(2)]
    with ProcessPoolExecutor(args.workers,initializer=initialize,initargs=(pop,params)) as pool:
        for n,(i,j,g,p) in enumerate(pool.map(one,tasks,chunksize=4),1):
            group[i,j]=g;pair[i,j]=p
            if n%200==0:print(n,'/',len(tasks),'elapsed',round(time.monotonic()-start),flush=True)
    report={}
    for i,sh in enumerate(T.SHEETS):
        lo,hi=np.quantile(vals[i,1:],[.025,.975],axis=0);counts={}
        for j,(n,d) in enumerate(T.PAIRS):counts[f'{n}/{d}']={name:dict(point_flag=bool(pair[i,0,j,k]),bootstrap_draws=int(pair[i,1:,j,k].sum())) for k,name in enumerate(T.DIAGNOSTIC_CATEGORIES)}
        report[sh]=dict(point=vals[i,0].tolist(),lower=lo.tolist(),upper=hi.tolist(),insufficient_signal_draws_any_group=0,unsuccessful_group_fits=0,diagnostics=counts)
    (out/'forced_alpha-table8.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'forced_alpha-draws.npz',points=vals,flags=flags,parameters=params,group_diagnostics=group,pair_diagnostics=pair,diagnostic_categories=T.DIAGNOSTIC_CATEGORIES)
    print('DONE archive diagnostics',flush=True)
if __name__=='__main__':main()

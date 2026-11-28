"""Current matched primary Table 8 gap-diagnostic replay; does not change fits."""
from pathlib import Path
import argparse,sys,os,json,io,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'analysis/table8'))
import table8_bootstrap as T
from bootstrap_reporting import gap_diagnostic
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'

def initialize(pop):
    os.environ['KCOR_NPH_MODE']='identified_only';T.initialize(pop,'revised')

def one(task):
    i,j=task;records=[];parameters=[];seed_status=[];native=T.compare.joint_fit
    def capture(*a,**kw):
        result,diag=native(*a,**kw);records.append(diag.get('gap_first_pass'));seed_status.append(diag.get('gap_seed_status','not_assessed'));parameters.append([float(result[0]),float(result[1]),float(diag.get('theta0_raw',result[1]))]);return result,diag
    T.compare.joint_fit=capture
    try:point,flags,failed,group_flags,pair_flags,members=T.estimate(i,j,diagnostics=True,return_composition=True)
    finally:T.compare.joint_fit=native
    assert len(records)==9
    return i,j,point,flags,failed,parameters,group_flags,pair_flags,seed_status,members,{(T.SHEETS[i],age,dose):records[3*k+dose] for k,age in enumerate([1940,1950,1960]) for dose in range(3)}

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--draws',type=int,default=999);p.add_argument('--workers',type=int,default=8);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    pop=[T.read_population(ROOT/'analysis/table8/input.xlsx',s) for s in T.SHEETS]
    ref=np.load(ROOT/'analysis/table8_diagnostics/current/identified_only-draws.npz')['points'];gaps={};draws=[{} for _ in range(a.draws)];maximum=0.;changed=[];vals=np.full((2,a.draws+1,3),np.nan);flags=np.full((2,a.draws+1,2),np.nan);params=np.full((2,a.draws+1,9,3),np.nan);start=time.monotonic();group_diag=np.zeros((2,a.draws+1,3,3,len(T.DIAGNOSTIC_CATEGORIES)),bool);pair_diag=np.zeros((2,a.draws+1,3,len(T.DIAGNOSTIC_CATEGORIES)),bool);seed=np.full((2,a.draws+1,9),"",dtype="U32")
    members=np.zeros((2,a.draws+1,3,3),bool)
    tasks=[(i,j) for j in range(a.draws+1) for i in range(2)]
    with ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(pop,)) as pool:
        for n,(i,j,v,f,e,p,g,pair,se,m,d) in enumerate(pool.map(one,tasks,chunksize=4),1):
            diff=float(np.max(np.abs(np.asarray(v)-ref[i,j])));maximum=max(maximum,diff)
            if diff>=1e-8:changed.append(dict(enrollment=T.SHEETS[i],draw=j,max_difference=diff))
            vals[i,j]=v;flags[i,j]=f,e;params[i,j]=p;group_diag[i,j]=g;pair_diag[i,j]=pair;seed[i,j]=se;members[i,j]=m
            (gaps if j==0 else draws[j-1]).update(d)
            if n%200==0 or n==len(tasks):print(n,'/',len(tasks),'max point difference',maximum,'elapsed',round(time.monotonic()-start),flush=True)
    report={};matched=vals.copy()
    for i,sheet in enumerate(T.SHEETS):
        matched[i],lo,hi,composition_changed,valid=T.composition_intervals(vals[i],members[i])
        diagnostic_counts={f"{n}/{d}":{name:dict(point_flag=bool(pair_diag[i,0,j,k]),bootstrap_draws=int(pair_diag[i,1:,j,k].sum())) for k,name in enumerate(T.DIAGNOSTIC_CATEGORIES)} for j,(n,d) in enumerate(T.PAIRS)}
        report[sheet]=dict(point=vals[i,0].tolist(),lower=lo.tolist(),upper=hi.tolist(),point_contributors=members[i,0].tolist(),composition_changed_draws=composition_changed.tolist(),valid_draws=valid.tolist(),insufficient_signal_draws_any_group=int((flags[i,1:,0]>0).sum()),unsuccessful_group_fits=int(flags[i,:,1].sum()),diagnostics=diagnostic_counts)
    np.savez_compressed(a.output/"identified_only-draws.npz",points=matched,raw_points=vals,contributors=members,flags=flags,parameters=params,group_diagnostics=group_diag,pair_diagnostics=pair_diag,diagnostic_categories=T.DIAGNOSTIC_CATEGORIES,seed_status=seed)
    (a.output/"identified_only-table8.json").write_text(json.dumps(report,indent=2))
    summary,status=gap_diagnostic(gaps,draws);pd.DataFrame(summary).to_csv(a.output/'gap-diagnostics.csv',index=False)
    counts=pd.DataFrame(summary).status.value_counts().to_dict();seed_values,seed_counts=np.unique(seed,return_counts=True);result=dict(seed_status_counts={str(z):int(c) for z,c in zip(seed_values,seed_counts)},draws=a.draws,comparisons=len(summary),status_counts=counts,max_point_difference=maximum,changed_draws=changed,cohort_status={str(k):v for k,v in status.items()});(a.output/'summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()

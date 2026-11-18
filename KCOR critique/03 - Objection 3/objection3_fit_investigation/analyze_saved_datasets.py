"""Post-run diagnostic: refit saved originals with joint k/theta alternatives.

This does not create bootstrap intervals or change the prespecified coverage study.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
CAL=HERE.parent/'objection3_corrected_validation'
sys.path.insert(0,str(CAL))
from production_adapter import K, arrays, configure

design=json.loads((CAL/'design.json').read_text(encoding='utf-8'))
scenario=design['common']|next(s for s in design['scenarios'] if s['id']=='unequal_large')
manifest=json.loads((CAL/'final_2026-11-10/manifest.json').read_text(encoding='utf-8'))
assert manifest['hashes'][str((CAL/'source_snapshot/code/KCOR.py').relative_to(CAL.parents[1]))] == hashlib.sha256((CAL/'source_snapshot/code/KCOR.py').read_bytes()).hexdigest()
cfg=configure(scenario['windows'])
weeks=int(scenario['weeks'])
t=np.arange(weeks,dtype=float)
skip=int(cfg['skip'])
shape=np.exp(cfg['gompertz']['gamma_per_week']*(t-skip))
fit_mask=(t>=skip)&(t<=40)

def approximate(k,theta):
    inc=np.where(t>=skip,k*shape,0.)
    before=np.r_[0.,np.cumsum(inc)[:-1]]
    return inc/(1+theta*before)

def exact(k,theta):
    H=np.cumsum(k*shape)
    before=np.r_[0.,H[:-1]]
    if theta<1e-8:
        return k*shape
    return (np.log1p(theta*H)-np.log1p(theta*before))/theta

def ratio(h,thetas):
    H=np.cumsum(np.where(t>=skip,h,0.),axis=1)
    adjusted=np.asarray([K.invert_gamma_frailty(H[d],thetas[d]) for d in (0,1)])
    q=np.divide(adjusted[1],adjusted[0],out=np.full(weeks,np.nan),where=adjusted[0]>K.EPS)
    return float(q[scenario['target']]/q[cfg['anchor']])

def fit(h,starting_k,starting_theta,model):
    x0=[max(.01,min(10.,starting_k*1000.)),max(1e-9,min(99.,starting_theta))]
    res=least_squares(lambda x:(model(x[0]/1000.,x[1])[fit_mask]-h[fit_mask])*1000.,
        x0=x0,bounds=([.01,0.],[10.,100.]),
        ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=500)
    return float(res.x[0]/1000.),float(res.x[1]),bool(res.success)

source=CAL/'final_2026-11-10/unequal_large'
rows=[]
for path in sorted(source.glob('[0-9][0-9][0-9][0-9].json')):
    record=json.loads(path.read_text(encoding='utf-8'))
    with np.load(path.with_suffix('.npz')) as saved:
        counts=saved['counts']
        assert hashlib.sha256(counts.tobytes()).hexdigest()==record['counts_sha256']
    dead,alive=arrays(counts)
    observed=np.asarray([K.hazard_from_mr(np.clip(dead[d]/(alive[d]+K.EPS),0,.999)) for d in (0,1)])
    fits={}
    for label,model in [('joint_approximate',approximate),('joint_exact_discrete',exact)]:
        params=[fit(observed[d],record['diagnostics'][d]['k'],record['diagnostics'][d]['theta'],model) for d in (0,1)]
        fits[label]=dict(params=params,point=ratio(observed,[x[1] for x in params]))
    rows.append(dict(repetition=record['repetition'],truth=record['truth'],production_point=record['point'],
                     joint_approximate_point=fits['joint_approximate']['point'],
                     joint_exact_discrete_point=fits['joint_exact_discrete']['point'],
                     production_theta_0=record['diagnostics'][0]['theta'],production_theta_1=record['diagnostics'][1]['theta'],
                     joint_approximate_k_0=fits['joint_approximate']['params'][0][0],
                     joint_approximate_k_1=fits['joint_approximate']['params'][1][0],
                     joint_approximate_theta_0=fits['joint_approximate']['params'][0][1],
                     joint_approximate_theta_1=fits['joint_approximate']['params'][1][1],
                     joint_approximate_both_success=all(x[2] for x in fits['joint_approximate']['params']),
                     joint_exact_discrete_both_success=all(x[2] for x in fits['joint_exact_discrete']['params'])))
assert len(rows)==600
with (HERE/'saved_dataset_refits.csv').open('x',newline='',encoding='utf-8') as output:
    writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

summary=dict(scope='Post-run original point estimates only; no new intervals or coverage',
    datasets=len(rows),source_manifest_sha256=hashlib.sha256((CAL/'final_2026-11-10/manifest.json').read_bytes()).hexdigest())
for label in ['production','joint_approximate','joint_exact_discrete']:
    point=np.asarray([r[f'{label}_point'] for r in rows])
    summary[label]=dict(mean=float(np.mean(point)),bias_from_model_truth=float(np.mean(point)-1),
        sd=float(np.std(point,ddof=1)),rmse=float(np.sqrt(np.mean((point-1)**2))),
        median=float(np.median(point)),quantiles=np.quantile(point,[.025,.25,.5,.75,.975]).tolist())
for label in ['joint_approximate','joint_exact_discrete']:
    summary[label]['successful_both_arms']=sum(r[f'{label}_both_success'] for r in rows)
summary['paired_joint_approximate_minus_production_mean']=float(np.mean([r['joint_approximate_point']-r['production_point'] for r in rows]))
summary['paired_joint_exact_minus_joint_approximate_mean']=float(np.mean([r['joint_exact_discrete_point']-r['joint_approximate_point'] for r in rows]))
summary['joint_approximate_theta_1_near_zero']=sum(r['joint_approximate_theta_1']<.01 for r in rows)
summary['production_theta_1_near_zero']=sum(r['production_theta_1']<.01 for r in rows)
with (HERE/'saved_dataset_refits_summary.json').open('x',encoding='utf-8') as output:
    json.dump(summary,output,indent=2)
print(json.dumps(summary,indent=2))

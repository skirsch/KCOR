"""Replay existing bootstrap seeds to inspect fitted parameter boundaries."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE=Path(__file__).resolve().parent
CAL=HERE.parent/'objection3_corrected_validation'
sys.path.insert(0,str(CAL))
from production_adapter import estimate
from simulation import bootstrap

design=json.loads((CAL/'design.json').read_text(encoding='utf-8'))
scenario=design['common']|next(s for s in design['scenarios'] if s['id']=='unequal_large')
run=CAL/'final_2026-11-10/unequal_large'
results=[]
for rep in (0,7,171):
    path=run/f'{rep:04d}.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    with np.load(path.with_suffix('.npz')) as saved:
        counts=saved['counts']
        points_saved=saved['draws']
    rng=np.random.default_rng(np.random.SeedSequence([design['final_seed_namespace'],2,rep,1]))
    estimates=[]
    theta=[]
    for b in range(len(points_saved)):
        result=estimate(bootstrap(counts,rng),scenario['windows'],scenario['target'])
        estimates.append(result['point'])
        theta.append([d['theta'] for d in result['diagnostics']])
    estimates=np.asarray(estimates)
    theta=np.asarray(theta)
    max_error=float(np.max(np.abs(estimates-points_saved)))
    assert max_error<1e-12
    results.append(dict(repetition=rep,original_point=record['point'],
        original_theta=[d['theta'] for d in record['diagnostics']],
        percentile_interval=record['intervals']['percentile'],
        percentile_covers_truth=bool(record['intervals']['percentile'][0]<=1<=record['intervals']['percentile'][1]),
        bootstrap_draws=len(estimates),max_replay_point_error=max_error,
        bootstrap_point_quantiles=np.quantile(estimates,[0,.025,.25,.5,.75,.975,1]).tolist(),
        bootstrap_point_mean=float(estimates.mean()),
        bootstrap_theta1_near_zero=int((theta[:,1]<.01).sum()),
        bootstrap_theta1_quantiles=np.quantile(theta[:,1],[0,.025,.25,.5,.75,.975,1]).tolist(),
        correlation_point_theta0=float(np.corrcoef(estimates,theta[:,0])[0,1]),
        correlation_point_theta1=float(np.corrcoef(estimates,theta[:,1])[0,1])))
with (HERE/'bootstrap_fit_replay.json').open('x',encoding='utf-8') as output:
    json.dump(results,output,indent=2)
print(json.dumps(results,indent=2))

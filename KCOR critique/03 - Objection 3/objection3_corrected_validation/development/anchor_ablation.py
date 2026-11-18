"""Exploratory deterministic ablation of the fitter's early k estimate."""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from production_adapter import K, arrays, configure
from simulation import population

design=json.loads((HERE/'design.json').read_text(encoding='utf-8'))
scenario=design['common']|design['scenarios'][2]
probs,truth=population(scenario)
counts=probs*np.asarray(scenario['arm_weights'])[:,None]*scenario['n']
cfg=configure(scenario['windows'])
dead,alive=arrays(counts)
t=np.arange(scenario['weeks'],dtype=float)
quiet=(t>=6)&(t<=40)
h=np.asarray([K.hazard_from_mr(np.clip(dead[d]/(alive[d]+K.EPS),0,.999)) for d in (0,1)])
H=np.cumsum(np.where(t>=cfg['skip'],np.nan_to_num(h),0.),axis=1)

source=inspect.getsource(K.fit_theta0_gompertz)
needle='    k_fixed = float(np.average(k_terms))'
assert source.count(needle)==1
ablation=source.replace(needle,f'    k_fixed = {scenario["k"]!r}  # deterministic generating value for diagnosis')
environment=dict(vars(K))
exec(compile(ablation,'<anchor-ablation>','exec'),environment)
fitter=environment['fit_theta0_gompertz']

def result(fit):
    params=[]
    for d in (0,1):
        (k,theta),diag=fit(np.where(t>=cfg['skip'],h[d],0.),t-cfg['skip'],quiet,
            cfg['gompertz']['k_anchor_weeks'],cfg['gompertz']['gamma_per_week'],deaths_arr=dead[d])
        params.append(dict(k=float(k),theta=float(theta),success=bool(diag['success']),status=str(diag['status'])))
    adjusted=np.asarray([K.invert_gamma_frailty(H[d],params[d]['theta']) for d in (0,1)])
    q=np.divide(adjusted[1],adjusted[0],out=np.full(adjusted.shape[1],np.nan),where=adjusted[0]>K.EPS)
    return dict(params=params,population_output=float(q[scenario['target']]/q[cfg['anchor']]))

out=dict(model_truth=truth,production=result(K.fit_theta0_gompertz),
         early_k_replaced_with_generating_k=result(fitter),
         ablation_note='For mechanistic diagnosis only; not a proposed estimator or coverage validation.')
(HERE/'development/anchor_ablation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))

"""Probe deterministic population fit and optimizer sensitivity without rerunning coverage."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from production_adapter import K, arrays, configure, estimate
from simulation import population

design=json.loads((HERE/'design.json').read_text(encoding='utf-8'))
scenario=design['common']|design['scenarios'][2]
probs,truth=population(scenario)
expected_counts=probs*np.asarray(scenario['arm_weights'])[:,None]*scenario['n']
result=estimate(expected_counts,scenario['windows'],scenario['target'],return_curves=True)
cfg=configure(scenario['windows'])
dead,alive=arrays(expected_counts)
t=np.arange(scenario['weeks'],dtype=float)
quiet=(t>=6)&(t<=40)
hazards=np.asarray([K.hazard_from_mr(np.clip(dead[d]/(alive[d]+K.EPS),0,.999)) for d in (0,1)])
H=np.cumsum(np.where(t>=cfg['skip'],np.nan_to_num(hazards),0.),axis=1)

def ratio(thetas):
    adjusted=np.asarray([K.invert_gamma_frailty(H[d],thetas[d]) for d in (0,1)])
    q=np.divide(adjusted[1],adjusted[0],out=np.full(adjusted.shape[1],np.nan),where=adjusted[0]>K.EPS)
    return float(q[scenario['target']]/q[cfg['anchor']])

multi=[]
for d in (0,1):
    arm=[]
    for initial in (0.,.1,1.,5.,20.):
        (k,theta),diag=K.fit_theta0_gompertz(
            np.where(t>=cfg['skip'],hazards[d],0.),t-cfg['skip'],quiet,
            cfg['gompertz']['k_anchor_weeks'],cfg['gompertz']['gamma_per_week'],
            theta0_init=initial,deaths_arr=dead[d])
        arm.append(dict(initial=initial,k=float(k),theta=float(theta),
            success=bool(diag['success']),status=str(diag['status']),
            relative_hazard_rmse=float(diag['relRMSE_hazard_fit'])))
    multi.append(arm)
fit=[d['theta'] for d in result['diagnostics']]
true=scenario['theta']
out=dict(model_truth=truth,generating_k=scenario['k'],generating_theta=true,
    production_population_output=result['point'],production_fit_theta=fit,
    production_fit_k=[d['k'] for d in result['diagnostics']],
    oracle_both_theta=ratio(true),
    oracle_control_theta=ratio([true[0],fit[1]]),
    oracle_treatment_theta=ratio([fit[0],true[1]]),
    optimizer_start_sensitivity=multi,
    max_theta_spread_across_starts=[max(r['theta'] for r in arm)-min(r['theta'] for r in arm) for arm in multi])
(HERE/'development/population_fit_probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))

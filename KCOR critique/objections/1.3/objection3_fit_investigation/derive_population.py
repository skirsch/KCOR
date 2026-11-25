"""Independent deterministic audit of the corrected KCOR quiet-window fit.

This file changes no production code and does not reselect the coverage target.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAL = HERE.parent/'objection3_corrected_validation'
sys.path.insert(0,str(CAL))
from production_adapter import K, arrays, configure, estimate
from simulation import population

design = json.loads((CAL/'design.json').read_text(encoding='utf-8'))
scenario = design['common'] | next(s for s in design['scenarios'] if s['id']=='unequal_large')
cfg = configure(scenario['windows'])
probs, truth = population(scenario)
expected_counts = probs*np.asarray(scenario['arm_weights'])[:,None]*scenario['n']
dead, alive = arrays(expected_counts)
t = np.arange(scenario['weeks'],dtype=float)
skip = int(cfg['skip'])
gamma = float(cfg['gompertz']['gamma_per_week'])
baseline_shape = np.exp(gamma*(t-skip))
fit_mask = (t>=skip)&(t<=40)
anchor_mask = (t>=skip)&(t<skip+int(cfg['gompertz']['k_anchor_weeks']))
observed = np.asarray([K.hazard_from_mr(np.clip(dead[d]/(alive[d]+K.EPS),0,.999)) for d in (0,1)])
production = estimate(expected_counts,scenario['windows'],scenario['target'])

def exact_week_hazard(k,theta):
    """Exact discrete gamma marginal log-hazard from enrollment, including skip weeks."""
    H=np.cumsum(k*baseline_shape)
    before=np.r_[0.,H[:-1]]
    if theta==0:
        return k*baseline_shape
    return (np.log1p(theta*H)-np.log1p(theta*before))/theta

def approx_week_hazard(k,theta):
    """Production's Gompertz hazard approximation for a contiguous quiet window."""
    increments=np.where(t>=skip,k*baseline_shape,0.)
    H_before=np.r_[0.,np.cumsum(increments)[:-1]]
    return np.divide(increments,1+theta*H_before)

def normalized_ratio(thetas):
    H=np.cumsum(np.where(t>=skip,observed,0.),axis=1)
    adjusted=np.asarray([K.invert_gamma_frailty(H[d],thetas[d]) for d in (0,1)])
    q=np.divide(adjusted[1],adjusted[0],out=np.full(adjusted.shape[1],np.nan),where=adjusted[0]>K.EPS)
    return float(q[scenario['target']]/q[cfg['anchor']])

arms=[]
joint_approx_theta=[]
joint_exact_theta=[]
for d in (0,1):
    h=observed[d]
    true_theta=float(scenario['theta'][d])
    k_anchor=float(np.mean(h[anchor_mask]/baseline_shape[anchor_mask]))
    production_theta=float(production['diagnostics'][d]['theta'])
    production_k=float(production['diagnostics'][d]['k'])
    assert abs(production_k-k_anchor)<1e-14
    exact_true=exact_week_hazard(scenario['k'],true_theta)
    exact_resid=float(np.max(np.abs(exact_true[fit_mask]-h[fit_mask])))
    assert exact_resid<1e-12
    scalar=minimize_scalar(lambda theta:float(np.sum((approx_week_hazard(k_anchor,theta)[fit_mask]-h[fit_mask])**2)),
                           bounds=(0.,10.),method='bounded',options={'xatol':1e-12})
    # A separate scalar optimizer checks the source's fixed-k theta optimum.
    # The source uses the same contiguous anchor-plus-quiet mask in this case.
    assert abs(float(scalar.x)-production_theta)<1e-5, (d,scalar.x,production_theta)
    def joint(model, x0):
        res=least_squares(lambda x:(model(x[0]/1000.,x[1])[fit_mask]-h[fit_mask])*1000.,
            x0=np.asarray(x0,float),bounds=([.01,0.],[10.,100.]),
            ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=5000)
        assert res.success
        return dict(k=float(res.x[0]/1000.),theta=float(res.x[1]),
                    max_abs_residual=float(np.max(np.abs(model(res.x[0]/1000.,res.x[1])[fit_mask]-h[fit_mask]))),
                    nfev=int(res.nfev))
    approx_joint=joint(approx_week_hazard,[scenario['k']*1000.,true_theta])
    exact_joint=joint(exact_week_hazard,[scenario['k']*1000.,true_theta])
    assert abs(exact_joint['k']-scenario['k'])<1e-9 and abs(exact_joint['theta']-true_theta)<1e-5
    joint_approx_theta.append(approx_joint['theta'])
    joint_exact_theta.append(exact_joint['theta'])
    arms.append(dict(arm=d,generating_k=scenario['k'],generating_theta=true_theta,
        early_k_from_marginal_hazards=k_anchor,
        production_theta=production_theta,
        independent_fixed_k_theta=float(scalar.x),
        exact_model_at_generating_parameters_max_hazard_error=exact_resid,
        approximate_model_at_generating_parameters_relative_rmse=float(np.sqrt(np.mean(((approx_week_hazard(scenario['k'],true_theta)[fit_mask]-h[fit_mask])/h[fit_mask])**2))),
        approximate_model_at_production_parameters_relative_rmse=float(np.sqrt(np.mean(((approx_week_hazard(k_anchor,production_theta)[fit_mask]-h[fit_mask])/h[fit_mask])**2))),
        joint_approximate_fit=approx_joint,
        joint_exact_discrete_fit=exact_joint))

out=dict(scenario='unequal_large',scope='exact expected counts; no sampling noise',
    source_sha256=hashlib.sha256((CAL/'source_snapshot/code/KCOR.py').read_bytes()).hexdigest(),
    design_sha256=hashlib.sha256((CAL/'design.json').read_bytes()).hexdigest(),
    model_truth=truth,production_population_output=production['point'],arms=arms,
    oracle_inversion_at_generating_theta=normalized_ratio(scenario['theta']),
    inversion_at_joint_approximate_theta=normalized_ratio(joint_approx_theta),
    inversion_at_joint_exact_discrete_theta=normalized_ratio(joint_exact_theta),
    note='Counterfactual fits diagnose approximation; they are not validated replacement estimators.')
with (HERE/'population_derivation.json').open('x',encoding='utf-8') as file:
    json.dump(out,file,indent=2)
print(json.dumps(out,indent=2))

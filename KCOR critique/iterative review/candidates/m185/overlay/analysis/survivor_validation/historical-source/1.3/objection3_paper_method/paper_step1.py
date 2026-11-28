"""Check the manuscript's explicit joint Step 1 on exact expected counts.

This is deliberately limited to a single quiet window. It is not an interval
validation or a substitute for the manuscript's unspecified block bootstrap.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CAL=HERE.parent/'objection3_corrected_validation'
sys.path.insert(0,str(CAL))
import numpy as np
from scipy.optimize import least_squares
from production_adapter import arrays,configure
from simulation import population

def fit_step1(h, t, gamma, theta_start):
    def residual(x):
        k=np.exp(x[0]);theta=x[1]
        Hgom=k*np.expm1(gamma*t)/gamma
        return h-k*np.exp(gamma*t)/(1+theta*Hgom)
    guess=float(np.mean(h[:4]/np.exp(gamma*t[:4])))
    r=least_squares(residual,[np.log(guess),theta_start],
        bounds=([np.log(1e-8),0.],[np.log(.1),100.]),
        xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=10000)
    return dict(k=float(np.exp(r.x[0])),theta=float(r.x[1]),
        objective=float(np.sum(r.fun**2)),success=bool(r.success),
        message=str(r.message))

def main():
    design=json.loads((CAL/'design.json').read_text())
    scenario=design['common']|next(s for s in design['scenarios'] if s['id']=='unequal_large')
    probabilities,truth=population(scenario)
    counts=probabilities*np.asarray(scenario['arm_weights'])[:,None]*scenario['n']
    dead,alive=arrays(counts)
    cfg=configure(scenario['windows']);gamma=cfg['gompertz']['gamma_per_week']
    start,end=scenario['windows'][0]
    t=np.arange(start,end+1)-cfg['skip']
    h=-np.log1p(-dead/alive)
    fits=[]
    for arm in (0,1):
        all_starts=[fit_step1(h[arm,start:end+1],t,gamma,s) for s in (.01,.5,4.,20.)]
        chosen=min(all_starts,key=lambda x:x['objective'])
        assert chosen['success']
        assert max(x['objective'] for x in all_starts)-chosen['objective']<1e-10
        fits.append(dict(chosen=chosen,all_starts=all_starts))
    H=np.cumsum(np.where(np.arange(h.shape[1])>=cfg['skip'],h,0.),axis=1)
    inverted=[]
    for arm in (0,1):
        theta=fits[arm]['chosen']['theta']
        inverted.append(np.expm1(theta*H[arm])/theta if theta>1e-8 else H[arm])
    inverted=np.asarray(inverted)
    ratio=inverted[1]/inverted[0]
    point=float(ratio[scenario['target']]/ratio[cfg['anchor']])
    report=dict(scope='manuscript joint Step 1, one quiet window, exact expected counts',
        scenario=scenario,truth=truth,point=point,fits=fits,
        formula='h_pred = k*exp(gamma*t)/(1+theta*k*expm1(gamma*t)/gamma)',
        source_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [HERE/'paper_step1.py',CAL/'simulation.py',CAL/'production_adapter.py',
                CAL/'design.json',HERE.parent/'objection3_resolution/source_snapshot/documentation/preprint/paper.md']})
    with (HERE/'exact_population.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(truth=truth,point=point,fits=[x['chosen'] for x in fits]),indent=2))

if __name__=='__main__':main()

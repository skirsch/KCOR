"""Inspect saved bootstrap convergence flags without changing pilot outcomes."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent/'objection3_joint_fit_candidate'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'objection3_corrected_validation'))
import numpy as np
from scipy.optimize import minimize
from conditional_fit import fit_conditional, _objective
from production_adapter import arrays, configure
from simulation import bootstrap


def main():
    manifest=json.loads((HERE/'pilot'/'manifest.json').read_text())
    scenario=manifest['scenario']
    cfg=configure(scenario['windows'])
    first,last=scenario['windows'][0]
    cases=[]
    for rep in (0,1,2):
        with np.load(HERE/'pilot'/f'{rep:04d}.npz') as saved:
            counts=saved['counts']; flags=saved['boot_success']
        rng=np.random.default_rng(np.random.SeedSequence([
            manifest['seed_namespace'],manifest['scenario_index'],rep,1]))
        for b in range(len(flags)):
            sample=bootstrap(counts,rng)
            if np.all(flags[b]):
                continue
            dead,alive=arrays(sample)
            for arm in np.where(~flags[b])[0]:
                d=dead[arm,first:last+1];a=alive[arm,first:last+1]
                fit=fit_conditional(d,a,gamma_per_week=cfg['gompertz']['gamma_per_week'])
                shape=np.exp(cfg['gompertz']['gamma_per_week']*np.arange(len(d)))
                starts=[]
                for th in (.01,.1,1,5,20):
                    k=float(np.mean(-np.log1p(-d[:4]/a[:4])/shape[:4]))
                    r=minimize(_objective,[np.log(k),th],args=(d,a,shape),method='L-BFGS-B',jac=True,
                        bounds=[(np.log(1e-8),np.log(1e-1)),(0.,100.)],
                        options=dict(maxiter=2000,ftol=1e-12,gtol=1e-7,maxls=100))
                    starts.append(dict(theta_start=th,theta=float(r.x[1]),objective=float(r.fun),
                        success=bool(r.success),message=str(r.message),gradient=float(np.max(np.abs(r.jac)))))
                cases.append(dict(repetition=rep,bootstrap=b,arm=int(arm),
                    recorded_fit=fit.__dict__,retry_starts=starts))
                if len(cases)>=5:
                    out=HERE/'pilot'/'failure_probe.json'
                    out.write_text(json.dumps(cases,indent=2),encoding='utf-8')
                    print(json.dumps(cases,indent=2))
                    return


if __name__=='__main__':
    main()

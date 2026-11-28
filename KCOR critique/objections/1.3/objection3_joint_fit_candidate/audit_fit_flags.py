"""Post-pilot audit of optimization status, leaving all saved draws unchanged."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'objection3_corrected_validation'))
import numpy as np
from scipy.optimize import minimize
from conditional_fit import fit_conditional,_objective
from production_adapter import arrays,configure
from simulation import bootstrap


def main():
    manifest=json.loads((HERE/'pilot'/'manifest.json').read_text())
    scenario=manifest['scenario']; ns=manifest['seed_namespace']; si=manifest['scenario_index']
    first,last=scenario['windows'][0]
    gamma=configure(scenario['windows'])['gompertz']['gamma_per_week']
    cases=[]
    for rep in range(manifest['repetitions']):
        with np.load(HERE/'pilot'/f'{rep:04d}.npz') as z:
            counts=z['counts']; flags=z['boot_success']; saved_theta=z['boot_theta']
        rng=np.random.default_rng(np.random.SeedSequence([ns,si,rep,1]))
        for b in range(manifest['draws']):
            sample=bootstrap(counts,rng)
            if np.all(flags[b]):
                continue
            dead,alive=arrays(sample)
            for arm in np.where(~flags[b])[0]:
                d=dead[arm,first:last+1];a=alive[arm,first:last+1]
                fit=fit_conditional(d,a,gamma_per_week=gamma)
                np.testing.assert_allclose(fit.theta,saved_theta[b,arm],atol=1e-12)
                assert not fit.success
                shape=np.exp(gamma*np.arange(len(d)))
                _,grad=_objective([np.log(fit.k_effective),fit.theta],d,a,shape)
                projected=grad.copy()
                if fit.theta<=1e-10 and projected[1]>0: projected[1]=0
                if fit.theta>=100-1e-10 and projected[1]<0: projected[1]=0
                case=dict(repetition=rep,bootstrap=b,arm=int(arm),theta=fit.theta,
                    objective=fit.objective,projected_gradient=float(np.max(np.abs(projected))))
                # Check whether a less stringent optimizer changes interior
                # flagged fits materially; preserve the pilot estimate.
                if fit.theta>.01:
                    retry=minimize(_objective,[np.log(fit.k_effective),fit.theta],
                        args=(d,a,shape),method='L-BFGS-B',jac=True,
                        bounds=[(np.log(1e-8),np.log(.1)),(0.,100.)],
                        options=dict(maxiter=2000,ftol=1e-12,gtol=1e-7,maxls=100))
                    case.update(retry_theta=float(retry.x[1]),
                        retry_objective=float(retry.fun),
                        retry_theta_change=float(abs(retry.x[1]-fit.theta)),
                        retry_objective_improvement=float(fit.objective-retry.fun))
                cases.append(case)
        if (rep+1)%10==0:
            print(f'audited {rep+1}/{manifest["repetitions"]}',flush=True)
    pg=np.array([c['projected_gradient'] for c in cases])
    interior=[c for c in cases if c['theta']>.01]
    report=dict(flagged_arm_fits=len(cases),at_zero=int(sum(c['theta']==0 for c in cases)),
        near_zero=int(sum(c['theta']<=.01 for c in cases)),
        projected_gradient_under_1e_3=int(sum(pg<1e-3)),
        projected_gradient_over_1e_2=int(sum(pg>.1)),
        interior_flagged=len(interior),
        interior_retry_theta_change_gt_01=int(sum(c['retry_theta_change']>.1 for c in interior)),
        interior_retry_objective_improvement_gt_1e_4=int(sum(c['retry_objective_improvement']>1e-4 for c in interior)),
        max_interior_retry_theta_change=max((c['retry_theta_change'] for c in interior),default=0),
        max_interior_retry_objective_improvement=max((c['retry_objective_improvement'] for c in interior),default=0))
    out=HERE/'pilot'/'assessment'
    (out/'fit_flags.json').write_text(json.dumps(dict(summary=report,cases=cases),indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

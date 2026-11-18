"""Check the three flagged original sample fits using projected gradients."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'objection3_corrected_validation'))
import numpy as np
from conditional_fit import _objective
from production_adapter import arrays,configure


def main():
    manifest=json.loads((HERE/'pilot'/'manifest.json').read_text())
    first,last=manifest['scenario']['windows'][0]
    gamma=configure(manifest['scenario']['windows'])['gompertz']['gamma_per_week']
    details=[]
    for rep in range(manifest['repetitions']):
        p=HERE/'pilot'/f'{rep:04d}'
        row=json.loads(p.with_suffix('.json').read_text())
        if row['accepted']: continue
        with np.load(p.with_suffix('.npz')) as z: dead,alive=arrays(z['counts'])
        for arm,fit in enumerate(row['diagnostics']):
            if fit['success']: continue
            d=dead[arm,first:last+1];a=alive[arm,first:last+1]
            shape=np.exp(gamma*np.arange(len(d)))
            _,grad=_objective([np.log(fit['k_effective']),fit['theta']],d,a,shape)
            projected=grad.copy()
            if fit['theta']<=1e-10 and projected[1]>0:projected[1]=0.
            details.append(dict(repetition=rep,arm=arm,theta=fit['theta'],
                raw_gradient=fit['gradient_norm'],
                projected_gradient=float(np.max(np.abs(projected)))))
    assert len(details)==3
    report=dict(flagged_original_arm_fits=len(details),all_projected_under_1e_3=
                all(d['projected_gradient']<1e-3 for d in details),details=details)
    (HERE/'pilot'/'assessment'/'original_flags.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

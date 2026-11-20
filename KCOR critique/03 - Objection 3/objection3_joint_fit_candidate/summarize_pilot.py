"""Audit every retained pilot record and summarize candidate performance."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'objection3_corrected_validation'))
import numpy as np
from production_adapter import intervals, arrays, configure
from simulation import sample_dataset,bootstrap
from candidate_estimator import estimate_candidate


def wilson(k,n):
    z=1.959963984540054
    p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half,center+half]


def main():
    run=HERE/'pilot'
    assert (run/'COMPLETE.json').exists()
    manifest=json.loads((run/'manifest.json').read_text())
    R,B=manifest['repetitions'],manifest['draws']
    scenario=manifest['scenario']
    cfg=configure(scenario['windows'])
    rows=[]
    hashes=set()
    replays=[]
    for rep in range(R):
        base=run/f'{rep:04d}'
        row=json.loads(base.with_suffix('.json').read_text())
        with np.load(base.with_suffix('.npz')) as saved:
            counts=saved['counts'];draws=saved['draws'];theta=saved['boot_theta']
            success=saved['boot_success'];near_zero=saved['boot_near_zero']
        assert draws.shape==(B,) and theta.shape==(B,2)
        digest=hashlib.sha256(counts.tobytes()).hexdigest()
        assert digest==row['counts_sha256'] and digest not in hashes
        hashes.add(digest)
        assert row['bootstrap_fit_failure_draws']==int(np.sum(~np.all(success,axis=1)))
        assert row['bootstrap_near_zero_draws']==int(np.sum(np.any(near_zero,axis=1)))
        # Independently reconstruct the unchanged gamma inversion and anchor
        # normalization without calling the candidate or KCOR's inversion.
        dead,alive=arrays(counts)
        hazards=-np.log1p(-dead/alive)
        hazards[:,:cfg['skip']]=0.
        H=np.cumsum(hazards,axis=1)
        direct=[]
        for arm in (0,1):
            th=row['diagnostics'][arm]['theta']
            direct.append(np.expm1(th*H[arm])/th if th>1e-8 else H[arm])
        direct=np.asarray(direct)
        ratio=np.divide(direct[1],direct[0],out=np.full_like(direct[0],np.nan),
                        where=direct[0]>0)
        direct_point=ratio[scenario['target']]/ratio[cfg['anchor']]
        np.testing.assert_allclose(direct_point,row['point'],rtol=1e-8)
        for kind in ('percentile','basic'):
            np.testing.assert_allclose(intervals(row['point'],draws)[kind],row['intervals'][kind],rtol=1e-12)
        if rep in (0,R-1):
            ns,si=manifest['seed_namespace'],manifest['scenario_index']
            regen,truth=sample_dataset(scenario,np.random.default_rng(np.random.SeedSequence([ns,si,rep,0])))
            np.testing.assert_array_equal(regen,counts)
            assert truth==row['truth']
            point=estimate_candidate(counts,scenario['windows'],scenario['target'])
            np.testing.assert_allclose(point['point'],row['point'],rtol=1e-11)
            rng=np.random.default_rng(np.random.SeedSequence([ns,si,rep,1]))
            for b in range(3):
                result=estimate_candidate(bootstrap(counts,rng),scenario['windows'],scenario['target'])
                np.testing.assert_allclose(result['point'],draws[b],rtol=1e-11)
            replays.append(dict(repetition=rep,checked_bootstrap_draws=3))
        rows.append(row)
    point=np.array([r['point'] for r in rows])
    truth=np.array([r['truth'] for r in rows])
    error=point-truth
    summary=dict(repetitions=R,bootstrap_draws=B,true_value=float(truth[0]),
        mean_point=float(point.mean()),mean_error=float(error.mean()),
        point_sd=float(point.std(ddof=1)),rmse=float(np.sqrt(np.mean(error**2))),
        accepted=int(sum(r['accepted'] for r in rows)),
        original_fit_failures=int(sum(any(not d['success'] for d in r['diagnostics']) for r in rows)),
        bootstrap_fit_failure_draws=int(sum(r['bootstrap_fit_failure_draws'] for r in rows)),
        bootstrap_near_zero_draws=int(sum(r['bootstrap_near_zero_draws'] for r in rows)),
        original_near_zero_arms=int(sum(d['near_zero'] for r in rows for d in r['diagnostics'])),
        intervals={})
    for kind in ('percentile','basic'):
        ci=np.array([r['intervals'][kind] for r in rows])
        valid=np.isfinite(ci).all(axis=1)&(ci[:,0]<=ci[:,1])
        covered=valid&(ci[:,0]<=truth)&(truth<=ci[:,1])
        summary['intervals'][kind]=dict(covered=int(covered.sum()),coverage=float(covered.mean()),
            wilson_95=wilson(int(covered.sum()),R),invalid=int((~valid).sum()),
            mean_width=float(np.mean((ci[:,1]-ci[:,0])[valid])),
            below_truth=int(np.sum(valid&(ci[:,1]<truth))),
            above_truth=int(np.sum(valid&(ci[:,0]>truth))))
    assessment=run/'assessment'
    assessment.mkdir(exist_ok=False)
    (assessment/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (assessment/'replicates.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['repetition','point','truth','accepted',
            'percentile_low','percentile_high','basic_low','basic_high',
            'theta_arm0','theta_arm1','bootstrap_fit_failure_draws','bootstrap_near_zero_draws'])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(repetition=r['repetition'],point=r['point'],truth=r['truth'],
                accepted=r['accepted'],percentile_low=r['intervals']['percentile'][0],
                percentile_high=r['intervals']['percentile'][1],
                basic_low=r['intervals']['basic'][0],basic_high=r['intervals']['basic'][1],
                theta_arm0=r['diagnostics'][0]['theta'],theta_arm1=r['diagnostics'][1]['theta'],
                bootstrap_fit_failure_draws=r['bootstrap_fit_failure_draws'],
                bootstrap_near_zero_draws=r['bootstrap_near_zero_draws']))
    (assessment/'audit.json').write_text(json.dumps(dict(passed=True,independent_datasets=len(hashes),
        retained_bootstrap_draws=R*B,reconstructed_intervals=True,seed_replays=replays),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

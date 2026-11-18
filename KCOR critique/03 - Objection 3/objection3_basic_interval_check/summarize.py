"""Audit retained draws and calculate interval coverage without changing outcomes."""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'objection3_corrected_validation'))
import numpy as np
from production_adapter import estimate,intervals
from simulation import sample_dataset,bootstrap
from run import hashes

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [c-h,c+h]

def main():
    out=HERE/'pilot'
    assert (out/'COMPLETE.json').exists()
    manifest=json.loads((out/'manifest.json').read_text())
    assert hashes()==manifest['hashes']
    d=manifest['design'];ns=d['seed_namespace'];R=d['outer_repetitions_per_size'];B=d['bootstrap_draws']
    summaries=[];replayed=[];unique=set()
    for si,n in enumerate(d['cohort_sizes']):
        case=dict(manifest['scenario'],n=n);rows=[]
        for rep in range(R):
            path=out/str(n)/f'{rep:04d}'
            row=json.loads(path.with_suffix('.json').read_text())
            with np.load(path.with_suffix('.npz')) as z:counts=z['counts'];draws=z['draws']
            assert draws.shape==(B,) and counts.sum()==n
            digest=hashlib.sha256(counts.tobytes()).hexdigest()
            assert digest==row['counts_sha256'] and digest not in unique
            unique.add(digest)
            for kind in ('percentile','basic'):
                np.testing.assert_allclose(intervals(row['point'],draws)[kind],row['intervals'][kind],rtol=1e-12)
            if rep in (0,R-1):
                generated,truth=sample_dataset(case,np.random.default_rng(np.random.SeedSequence([ns,si,rep,0])))
                np.testing.assert_array_equal(generated,counts)
                assert truth==row['truth']
                point=estimate(counts,case['windows'],case['target'])
                np.testing.assert_allclose(point['point'],row['point'],rtol=1e-12)
                brng=np.random.default_rng(np.random.SeedSequence([ns,si,rep,1]))
                for b in range(3):
                    fit=estimate(bootstrap(counts,brng),case['windows'],case['target'])
                    np.testing.assert_allclose(fit['point'],draws[b],rtol=1e-12)
                replayed.append(dict(n=n,repetition=rep,draws=3))
            rows.append(row)
        for kind in ('percentile','basic'):
            ci=np.asarray([r['intervals'][kind] for r in rows]);truth=np.asarray([r['truth'] for r in rows])
            valid=np.isfinite(ci).all(axis=1)&(ci[:,0]<=ci[:,1])
            covered=valid&(ci[:,0]<=truth)&(truth<=ci[:,1]);k=int(covered.sum())
            e=np.asarray([r['point']-r['truth'] for r in rows])
            summaries.append(dict(n=n,interval=kind,covered=k,total=R,coverage=k/R,
                wilson_95=wilson(k,R),accepted=sum(r['accepted'] for r in rows),
                invalid=int((~valid).sum()),bootstrap_fit_failure_draws=sum(r['bootstrap_fit_failures'] for r in rows),
                mean_width=float(np.mean((ci[:,1]-ci[:,0])[valid])) if valid.any() else None,
                mean_error=float(e.mean()),point_sd=float(np.std(e,ddof=1)),
                entirely_above=int(np.sum(valid&(ci[:,0]>truth))),
                entirely_below=int(np.sum(valid&(ci[:,1]<truth)))))
    assessment=out/'assessment';assessment.mkdir(exist_ok=False)
    (assessment/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    (assessment/'audit.json').write_text(json.dumps(dict(passed=True,datasets=len(unique),
        retained_draws=len(unique)*B,reconstructed_intervals=True,seed_replays=replayed),indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()

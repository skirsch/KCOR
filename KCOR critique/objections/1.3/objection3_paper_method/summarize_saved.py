"""Audit and summarize the retained paired manuscript Step 1 point estimates."""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CAL=HERE.parent/'objection3_corrected_validation'
sys.path.insert(0,str(CAL))
import numpy as np
from production_adapter import arrays,configure
from paper_step1 import fit_step1

def main():
    out=HERE/'saved_600';assert (out/'COMPLETE.json').exists()
    manifest=json.loads((out/'manifest.json').read_text())
    for path,digest in manifest['hashes'].items():
        source=(HERE.parent/path).read_bytes()
        if Path(path).name=='production_adapter.py':
            # The retained run predates the folder move. Restore the original
            # repository-root expression in memory for the historical hash.
            assert source.count(b'parents[4]')==1
            source=source.replace(b'parents[4]',b'parents[2]')
        assert hashlib.sha256(source).hexdigest()==digest
    cfg=configure(manifest['scenario']['windows']);first,last=manifest['scenario']['windows'][0]
    t=np.arange(first,last+1)-cfg['skip'];gamma=cfg['gompertz']['gamma_per_week']
    paper=[];old=[];theta=[];keys=set();replayed=[]
    prior=CAL/'final_2026-11-10/unequal_large'
    for i in range(600):
        row=json.loads((out/f'{i:04d}.json').read_text())
        saved=json.loads((prior/f'{i:04d}.json').read_text())
        assert row['repetition']==i and row['truth']==saved['truth']==1.
        assert row['tested_code_point']==saved['point']
        assert row['counts_sha256']==saved['counts_sha256'] and row['counts_sha256'] not in keys
        keys.add(row['counts_sha256'])
        assert all(f['chosen']['success'] for f in row['fits'])
        paper.append(row['paper_joint_point']);old.append(row['tested_code_point'])
        theta.append([f['chosen']['theta'] for f in row['fits']])
        if i in (0,299,599):
            with np.load((prior/f'{i:04d}.npz')) as z:counts=z['counts']
            assert hashlib.sha256(counts.tobytes()).hexdigest()==row['counts_sha256']
            dead,alive=arrays(counts);h=-np.log1p(-dead/alive)
            for arm in (0,1):
                check=fit_step1(h[arm,first:last+1],t,gamma,4.)
                assert abs(check['theta']-row['fits'][arm]['chosen']['theta'])<1e-5
            replayed.append(i)
    paper=np.asarray(paper);old=np.asarray(old);theta=np.asarray(theta)
    def metrics(x):return dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),
        mean_error=float(np.mean(x-1)),rmse=float(np.sqrt(np.mean((x-1)**2))),
        minimum=float(np.min(x)),maximum=float(np.max(x)))
    summary=dict(n=600,truth=1.,paper_joint_step1=metrics(paper),
        tested_fixed_rate_code=metrics(old),
        paper_fit_successes=600,theta_near_zero_by_arm=np.sum(theta<.01,axis=0).tolist(),
        scope='point estimates, one quiet window; no manuscript bootstrap or full diagnostic gate')
    audit=dict(passed=True,datasets=len(keys),original_points_paired=True,
        source_hashes_verified=True,replayed_datasets=replayed)
    with (out/'summary.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
    with (out/'audit.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

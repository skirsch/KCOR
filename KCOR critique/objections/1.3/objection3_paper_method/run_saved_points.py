"""Apply the manuscript's one-window joint Step 1 to 600 retained datasets."""
from __future__ import annotations
import hashlib,json,os,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent
CAL=HERE.parent/'objection3_corrected_validation'
sys.path.insert(0,str(CAL))
import numpy as np
from production_adapter import arrays,configure
from paper_step1 import fit_step1

def main():
    prior=CAL/'final_2026-11-10/unequal_large'
    source=[HERE/'paper_step1.py',HERE/'run_saved_points.py',CAL/'design.json',
        CAL/'production_adapter.py',CAL/'simulation.py',
        HERE.parent/'objection3_resolution/source_snapshot/documentation/preprint/paper.md']
    hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    design=json.loads((CAL/'design.json').read_text())
    scenario=design['common']|next(s for s in design['scenarios'] if s['id']=='unequal_large')
    cfg=configure(scenario['windows']);gamma=cfg['gompertz']['gamma_per_week']
    first,last=scenario['windows'][0];t=np.arange(first,last+1)-cfg['skip']
    out=HERE/'saved_600';out.mkdir(exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(dict(created=datetime.now(timezone.utc).isoformat(),
        source=prior.as_posix(),scenario=scenario,hashes=hashes,
        python=platform.python_version(),numpy=np.__version__,workers=1),indent=2),encoding='utf-8')
    try:
        if os.name=='nt':
            try:
                import psutil
                psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            except (ImportError,OSError):pass
        started=time.perf_counter()
        for rep in range(600):
            base=prior/f'{rep:04d}'
            old=json.loads(base.with_suffix('.json').read_text())
            with np.load(base.with_suffix('.npz')) as z:counts=z['counts']
            assert hashlib.sha256(counts.tobytes()).hexdigest()==old['counts_sha256']
            dead,alive=arrays(counts)
            h=-np.log1p(-dead/alive)
            H=np.cumsum(np.where(np.arange(h.shape[1])>=cfg['skip'],h,0.),axis=1)
            fits=[];curves=[]
            for arm in (0,1):
                candidates=[fit_step1(h[arm,first:last+1],t,gamma,start)
                    for start in (.01,.5,4.,20.)]
                fit=min(candidates,key=lambda f:f['objective'])
                fits.append(dict(chosen=fit,all_starts=candidates))
                theta=fit['theta']
                curves.append(np.expm1(theta*H[arm])/theta if theta>1e-8 else H[arm])
            curves=np.asarray(curves)
            ratio=np.divide(curves[1],curves[0],out=np.full(curves.shape[1],np.nan),where=curves[0]>0)
            point=float(ratio[scenario['target']]/ratio[cfg['anchor']])
            record=dict(repetition=rep,truth=old['truth'],paper_joint_point=point,
                tested_code_point=old['point'],fits=fits,counts_sha256=old['counts_sha256'])
            with (out/f'{rep:04d}.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
            if (rep+1)%50==0:
                elapsed=time.perf_counter()-started
                print(f'{rep+1}/600 complete; {elapsed:.1f}s elapsed; ~{elapsed/(rep+1)*(599-rep):.1f}s remaining',flush=True)
        assert all(hashlib.sha256((HERE.parent/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
        with (out/'COMPLETE.json').open('x',encoding='utf-8') as f:
            json.dump(dict(datasets=600,completed=datetime.now(timezone.utc).isoformat(),
                seconds=time.perf_counter()-started),f,indent=2)
    except BaseException:
        print('Run stopped; retained records remain under saved_600',flush=True)
        raise

if __name__=='__main__':main()

"""Fresh bounded one-worker candidate pilot. Resumable without overwriting records."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'objection3_corrected_validation'))
import numpy as np
import scipy
from production_adapter import intervals
from simulation import bootstrap, sample_dataset
from candidate_estimator import estimate_candidate

R, B, NAMESPACE = 60, 199, 672913


def hashes():
    paths = sorted(HERE.glob('*.py')) + [HERE/'DESIGN.md']
    paths += [HERE.parent/'objection3_corrected_validation'/p for p in
              ('simulation.py', 'production_adapter.py', 'design.json',
               'source_snapshot/code/KCOR.py', 'source_snapshot/data/Czech/Czech.yaml')]
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    gate = json.loads((HERE/'exact_population'/'verification.json').read_text())
    assert gate['passed']
    design = json.loads((HERE.parent/'objection3_corrected_validation'/'design.json').read_text())
    scenario = design['common'] | next(s for s in design['scenarios'] if s['id']=='unequal_large')
    si = next(i for i,s in enumerate(design['scenarios']) if s['id']=='unequal_large')
    out = HERE/'pilot'
    manifest_path = out/'manifest.json'
    current_hashes = hashes()
    if out.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['hashes']==current_hashes and manifest['scenario']==scenario
        assert manifest['repetitions']==R and manifest['draws']==B and manifest['seed_namespace']==NAMESPACE
    else:
        out.mkdir()
        manifest = dict(created=datetime.now(timezone.utc).isoformat(), scenario=scenario,
            scenario_index=si, repetitions=R, draws=B, seed_namespace=NAMESPACE,
            hashes=current_hashes, python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, workers=1)
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Give the foreground user work priority on Windows while the one-worker study runs.
    if os.name=='nt':
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except (ImportError, OSError):
            pass
    started=time.perf_counter()
    for rep in range(R):
        jp=out/f'{rep:04d}.json'
        npz=out/f'{rep:04d}.npz'
        if jp.exists():
            assert npz.exists()
            continue
        assert not npz.exists(), f'Partial result for {rep}; inspect before resuming'
        data_rng=np.random.default_rng(np.random.SeedSequence([NAMESPACE,si,rep,0]))
        boot_rng=np.random.default_rng(np.random.SeedSequence([NAMESPACE,si,rep,1]))
        counts,truth=sample_dataset(scenario,data_rng)
        original=estimate_candidate(counts,scenario['windows'],scenario['target'])
        draws=np.empty(B)
        boot_theta=np.empty((B,2))
        boot_success=np.empty((B,2),bool)
        boot_near_zero=np.empty((B,2),bool)
        for b in range(B):
            result=estimate_candidate(bootstrap(counts,boot_rng),scenario['windows'],scenario['target'])
            draws[b]=result['point']
            boot_theta[b]=[d['theta'] for d in result['diagnostics']]
            boot_success[b]=[d['success'] for d in result['diagnostics']]
            boot_near_zero[b]=[d['near_zero'] for d in result['diagnostics']]
        row=dict(repetition=rep,truth=truth,point=original['point'],
            accepted=original['accepted'],diagnostics=original['diagnostics'],
            intervals=intervals(original['point'],draws),
            bootstrap_fit_failure_draws=int(np.sum(~np.all(boot_success,axis=1))),
            bootstrap_near_zero_draws=int(np.sum(np.any(boot_near_zero,axis=1))),
            counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())
        with npz.open('xb') as f:
            np.savez_compressed(f,counts=counts,draws=draws,boot_theta=boot_theta,
                boot_success=boot_success,boot_near_zero=boot_near_zero)
        with jp.open('x',encoding='utf-8') as f:
            json.dump(row,f,indent=2)
        if (rep+1)%5==0 or rep==0:
            elapsed=time.perf_counter()-started
            print(f'{rep+1}/{R} complete; elapsed {elapsed:.1f}s; {elapsed/(rep+1)*(R-rep-1):.1f}s estimated remaining',flush=True)
    assert hashes()==current_hashes, 'Source files changed during pilot'
    complete=out/'COMPLETE.json'
    if not complete.exists():
        with complete.open('x',encoding='utf-8') as f:
            json.dump(dict(completed=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.perf_counter()-started, jobs=R,draws=R*B),f,indent=2)
    print('Pilot complete',flush=True)


if __name__=='__main__':
    main()

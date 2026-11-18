"""Run or resume the frozen, single-worker size extension."""
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
import scipy
from production_adapter import estimate,intervals
from simulation import sample_dataset,bootstrap,population

def hashes():
    paths=[HERE/'run.py',HERE/'summarize.py',HERE/'design.json',HERE/'DESIGN.md',
        CAL/'simulation.py',CAL/'production_adapter.py',CAL/'design.json',
        CAL/'source_snapshot/code/KCOR.py',CAL/'source_snapshot/data/Czech/Czech.yaml']
    return {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def main():
    design=json.loads((HERE/'design.json').read_text())
    old=json.loads((CAL/'design.json').read_text())
    scenario=old['common'] | next(s for s in old['scenarios'] if s['id']=='unequal_large')
    nrep=design['outer_repetitions_per_size'];B=design['bootstrap_draws'];ns=design['seed_namespace']
    out=HERE/'pilot';source_hashes=hashes()
    if out.exists():
        manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['hashes']==source_hashes and manifest['design']==design
    else:
        out.mkdir()
        manifest=dict(created=datetime.now(timezone.utc).isoformat(),design=design,
                      scenario=scenario,hashes=source_hashes,python=platform.python_version(),
                      numpy=np.__version__,scipy=scipy.__version__)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if os.name=='nt':
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except (ImportError,OSError):pass
    total=nrep*len(design['cohort_sizes']);done=0;start=time.perf_counter()
    for size_index,n in enumerate(design['cohort_sizes']):
        case=dict(scenario,n=n)
        p,truth=population(case)
        expected=p*np.asarray(case['arm_weights'])[:,None]*n
        exact=estimate(expected,case['windows'],case['target'])
        assert abs(truth-1.)<1e-12
        sub=out/str(n);sub.mkdir(exist_ok=True)
        exact_path=sub/'exact_population.json'
        if exact_path.exists():
            assert json.loads(exact_path.read_text())['point']==exact['point']
        else:
            exact_path.write_text(json.dumps(dict(truth=truth,point=exact['point'],diagnostics=exact['diagnostics']),indent=2),encoding='utf-8')
        for rep in range(nrep):
            jp=sub/f'{rep:04d}.json';npz=jp.with_suffix('.npz')
            if jp.exists():
                assert npz.exists();done+=1;continue
            assert not npz.exists(),f'Partial pair {npz}'
            rng=np.random.default_rng(np.random.SeedSequence([ns,size_index,rep,0]))
            brng=np.random.default_rng(np.random.SeedSequence([ns,size_index,rep,1]))
            counts,truth=sample_dataset(case,rng)
            original=estimate(counts,case['windows'],case['target'])
            draws=np.empty(B)
            failed=0
            for b in range(B):
                fit=estimate(bootstrap(counts,brng),case['windows'],case['target'])
                draws[b]=fit['point']
                failed+=int(any(not d['success'] for d in fit['diagnostics']))
            row=dict(n=n,repetition=rep,truth=truth,point=original['point'],
                     accepted=original['accepted'],diagnostics=original['diagnostics'],
                     bootstrap_fit_failures=failed,intervals=intervals(original['point'],draws),
                     counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())
            with npz.open('xb') as f:np.savez_compressed(f,counts=counts,draws=draws)
            with jp.open('x',encoding='utf-8') as f:json.dump(row,f,indent=2)
            done+=1
            if done==1 or done%5==0:
                elapsed=time.perf_counter()-start
                print(f'{done}/{total} complete; {elapsed:.1f}s elapsed; ~{elapsed/done*(total-done):.1f}s remaining',flush=True)
    assert hashes()==source_hashes
    complete=out/'COMPLETE.json'
    if not complete.exists():
        with complete.open('x',encoding='utf-8') as f:
            json.dump(dict(completed=datetime.now(timezone.utc).isoformat(),datasets=total,
                           draws=total*B,elapsed_seconds=time.perf_counter()-start),f,indent=2)
    print('Complete',flush=True)

if __name__=='__main__':main()

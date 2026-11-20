"""Compare current and candidate point estimates on the same pilot datasets."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'objection3_corrected_validation'))
import numpy as np
from production_adapter import estimate


def metrics(v,truth):
    error=v-truth
    return dict(mean_error=float(error.mean()),sd=float(v.std(ddof=1)),
                rmse=float(np.sqrt(np.mean(error**2))))


def main():
    manifest=json.loads((HERE/'pilot'/'manifest.json').read_text())
    scenario=manifest['scenario'];R=manifest['repetitions']
    old=[];new=[];rows=[]
    for rep in range(R):
        path=HERE/'pilot'/f'{rep:04d}'
        with np.load(path.with_suffix('.npz')) as z: counts=z['counts']
        candidate=json.loads(path.with_suffix('.json').read_text())
        original=estimate(counts,scenario['windows'],scenario['target'])
        old.append(original['point']);new.append(candidate['point'])
        rows.append(dict(repetition=rep,current=original['point'],candidate=candidate['point'],
                         truth=candidate['truth'],current_accepted=original['accepted'],
                         candidate_raw_accepted=candidate['accepted']))
    old=np.asarray(old);new=np.asarray(new);truth=rows[0]['truth']
    report=dict(same_datasets=R,truth=truth,current=metrics(old,truth),
        candidate=metrics(new,truth),current_accepted=sum(r['current_accepted'] for r in rows),
        candidate_raw_accepted=sum(r['candidate_raw_accepted'] for r in rows),
        rows=rows)
    out=HERE/'pilot'/'assessment'/'paired_points.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__': main()

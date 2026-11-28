"""Replay the draws determining the archived 2.5th and 97.5th percentiles."""
import os,sys,io,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="forced_alpha",KCOR_BOOTSTRAP_DRAWS="0")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"analysis/table8"))
import table8_bootstrap as T
def init(pop):T.initialize(pop,"revised")
def run(job):
    i,j=job;v,f,e=T.estimate(i,j);return i,j,v,f,e
def main():
    pop=[T.read_population(io.BytesIO((ROOT/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
    old=np.load(ROOT/"analysis/table8/forced_alpha-draws.npz")["points"]
    assert old.shape==(2,1000,3)
    jobs=set()
    for i in range(2):
        for j in range(3):
            order=np.argsort(old[i,1:,j])+1
            for q in (.025,.975):
                rank=q*(len(order)-1)
                for k in [int(np.floor(rank)),int(np.ceil(rank))]:
                    jobs.add((i,int(order[k])))
    rows=[]
    with ProcessPoolExecutor(6,initializer=init,initargs=(pop,)) as pool:
        for i,j,v,f,e in pool.map(run,sorted(jobs),chunksize=1):
            delta=np.asarray(v)-old[i,j]
            rows.append(dict(enrollment=T.SHEETS[i],draw=j,old=old[i,j].tolist(),new=v,difference=delta.tolist(),insufficient=f,failed=e))
            print(T.SHEETS[i],j,float(np.max(abs(delta))),flush=True)
    result=dict(selected_draws=len(rows),maximum_absolute_change=max(max(abs(v) for v in row['difference']) for row in rows),rows=rows)
    (Path(__file__).resolve().parent/"percentile-draw-check.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print({k:v for k,v in result.items() if k!='rows'},flush=True)
if __name__=="__main__":main()


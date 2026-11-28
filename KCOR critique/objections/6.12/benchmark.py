"""Matched point and timing comparison on the actual archived negative control."""
import os,sys,io,json,time
from pathlib import Path
import numpy as np
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="forced_alpha",KCOR_BOOTSTRAP_DRAWS="0")
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(root/"analysis/table8"))
import table8_bootstrap as T
def main():
    pop=[T.read_population(io.BytesIO((root/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
    T.initialize(pop,"revised")
    assert T.WAVE["apply_correction"]
    values=[]
    for draw in (0,1,42):
        for i in range(2):
            start=time.perf_counter()
            v,flags,failed=T.estimate(i,draw)
            values.append(dict(draw=draw,enrollment=T.SHEETS[i],point=v,insufficient=flags,failed=failed,seconds=time.perf_counter()-start))
            print(values[-1],flush=True)
    out.write_text(json.dumps(values,indent=2),encoding="utf8")
if __name__=="__main__":main()

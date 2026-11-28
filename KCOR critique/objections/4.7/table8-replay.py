from pathlib import Path
import os,sys,json,io
from concurrent.futures import ProcessPoolExecutor
import numpy as np
A=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m100"
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
sys.path.insert(0,str(D/"analysis/table8"))
import table8_bootstrap as T
def init(pop,mode):
 os.environ["KCOR_NPH_MODE"]=mode
 T.initialize(pop,"revised")
 assert T.WAVE["apply_correction"]==(mode=="forced_alpha")
def run(job):
 i,j=job;return i,j,T.estimate(i,j)
def main():
 mode=sys.argv[1]
 pop=[T.read_population(io.BytesIO((D/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
 init(pop,mode)
 data=np.empty((2,1000,3));flags=np.empty((2,1000,2))
 for i in range(2):
  v,f,e=T.estimate(i,0);data[i,0]=v;flags[i,0]=f,e
 print(mode,"points",data[:,0].tolist(),flush=True)
 with ProcessPoolExecutor(5,initializer=init,initargs=(pop,mode)) as p:
  for step,(i,j,(v,f,e)) in enumerate(p.map(run,[(i,j) for j in range(1,1000) for i in range(2)],chunksize=4),1):
   data[i,j]=v;flags[i,j]=f,e
   if step%200==0:print(mode,step,flush=True)
 res={}
 for i,e in enumerate(T.SHEETS):
  res[e]=dict(point=data[i,0].tolist(),lower=np.quantile(data[i,1:],[.025,.975],axis=0)[0].tolist(),upper=np.quantile(data[i,1:],[.025,.975],axis=0)[1].tolist(),flagged_draws=int((flags[i,1:,0]>0).sum()),failed=int(flags[i,:,1].sum()))
 (A/(mode+"-table8.json")).write_text(json.dumps(res,indent=2),encoding="utf8")
 np.savez_compressed(A/(mode+"-draws.npz"),points=data,flags=flags)
 print(res,flush=True)
if __name__=="__main__":main()
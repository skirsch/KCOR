from pathlib import Path
import os,sys,json,io
from concurrent.futures import ProcessPoolExecutor
import numpy as np
A=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m94"
sys.path.insert(0,str(D/"analysis/table8"))
import table8_bootstrap as T
def init(pop):
 T.initialize(pop,"revised")
 T.WAVE=dict(T.WAVE,apply_alpha=1.,apply_correction=False)
 T.K.get_covid_correction_config=lambda: T.WAVE
def run(job):
 i,j=job;return i,j,T.estimate(i,j)
def main():
 pop=[T.read_population(io.BytesIO((D/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
 init(pop)
 data=np.empty((2,1000,3));flags=np.empty((2,1000,2))
 for i in range(2):
  value=T.estimate(i,0);data[i,0]=value[0];flags[i,0]=value[1:]
 print("module-off points",data[:,0].tolist(),flush=True)
 with ProcessPoolExecutor(5,initializer=init,initargs=(pop,)) as p:
  for step,(i,j,value) in enumerate(p.map(run,[(i,j) for i in range(2) for j in range(1,1000)],chunksize=8),1):
   data[i,j]=value[0];flags[i,j]=value[1:]
   if step%400==0:print(step,flush=True)
 res={}
 for i,e in enumerate(T.SHEETS):
  res[e]=dict(point=data[i,0].tolist(),lower=np.quantile(data[i,1:],[.025,.975],axis=0)[0].tolist(),upper=np.quantile(data[i,1:],[.025,.975],axis=0)[1].tolist(),flagged_draws=int((flags[i,1:,0]>0).sum()),failed=int(flags[i,:,1].sum()))
 (A/"module-off-table8.json").write_text(json.dumps(res,indent=2),encoding="utf8")
 np.savez_compressed(A/"module-off-draws.npz",points=data,flags=flags)
 print(res)
if __name__=="__main__":main()
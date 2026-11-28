from pathlib import Path
import os,sys,io,json
import numpy as np
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m104"
os.environ["KCOR_NPH_MODE"]=sys.argv[1]
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
sys.path.insert(0,str(D/"analysis/table8"))
import table8_bootstrap as T
pop=[T.read_population(io.BytesIO((D/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
T.initialize(pop,"revised")
old=np.load(D/"analysis/table8"/(sys.argv[1]+"-draws.npz"))
diff=[]; flags=[]; overlap=[]
for i,(dates,time,quiet,keep,counts) in enumerate(T.DATA):
 marked=(dates>=np.datetime64(T.WAVE["start_dt"]))&(dates<=np.datetime64(T.WAVE["end_dt"]))&(time>=T.K.DYNAMIC_HVE_SKIP_WEEKS)
 overlap.append(int((quiet&marked).sum()))
 for j in (0,1,2,21,50,100,500,999):
  p,f,e=T.estimate(i,j)
  diff.append(float(np.max(np.abs(np.array(p)-old["points"][i,j]))))
  flags.append([int(f),int(e)])
res={"mode":sys.argv[1],"eligible_quiet_wave_overlap":overlap,"draws_checked":[0,1,2,21,50,100,500,999],"max_point_difference":max(diff),"flags":flags}
assert max(diff)<1e-9
assert not any(any(x) for x in flags)
Path("KCOR critique/objections/5.2",sys.argv[1]+"-table8-check.json").write_text(json.dumps(res,indent=2))
print(res)

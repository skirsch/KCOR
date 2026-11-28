from pathlib import Path
import os,sys,json,io
import numpy as np
R=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m95"
os.environ["DATASET"]="Czech"
sys.path.insert(0,str(D/"code"))
import KCOR as K
t=np.arange(24,dtype=float);g=.085/52
H=.002*np.expm1(g*np.arange(25))/g
h=np.diff(np.log1p(2*H)/2)
rows=[]
for n in range(4):
 quiet=t<12+n
 v,d=K.fit_theta0_gompertz(h,t,quiet,4,g,deaths_arr=np.full(24,200),wave=(0,11),alpha=1.18)
 assert d["n_fit"]==n and int(d["fit_mask_theta"].sum())==n
 assert d["success"]==(n>=3)
 if n<3:assert d["status"]=="insufficient_data" and np.isnan(v).all()
 rows.append(dict(eligible=n,reported=d["n_fit"],success=bool(d["success"]),status=d["status"]))
sys.path.insert(0,str(D/"analysis/table8"))
import table8_bootstrap as T
pop=[T.read_population(io.BytesIO((D/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
T.initialize(pop,"revised")
old=np.load(D/"analysis/survivor_validation/table8-draws.npz")["points"]
for i in range(2):
 for j in (0,1):
  v,flags,failed=T.estimate(i,j)
  np.testing.assert_allclose(v,old[i,j],rtol=1e-12)
  assert flags==failed==0
out=dict(eligibility=rows,table8_points_and_first_draw="identical to frozen m94",full_999_draws_rerun=False)
(R/"verification.json").write_text(json.dumps(out,indent=2))
print(out)
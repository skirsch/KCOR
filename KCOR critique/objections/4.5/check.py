from pathlib import Path
import os,sys,json,io
import numpy as np
R=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m98"
os.environ["DATASET"]="Czech"
sys.path.insert(0,str(D/"code"))
import KCOR as K
t=np.arange(60,dtype=float);g=.085/52;k=.002;theta=2
B=k*np.expm1(g*np.arange(61))/g
h=np.diff(np.log1p(theta*B)/theta)
rows=[]
for label,q in [("two first,31 later",(t<2)|((t>=20)&(t<51))),("all segments short",(t%4)<2),("ordinary first window",(t<20)|((t>=30)&(t<51))),("insufficient total",t<2)]:
 v,d=K.fit_theta0_gompertz(h,t,q,4,g,deaths_arr=np.full(60,200))
 print(label,v,d["status"],d["n_fit"],flush=True)
 if q.sum()>=3:
  assert d["success"] and abs(v[0]-k)<1e-7 and abs(v[1]-theta)<1e-5
 else:assert not d["success"]
 rows.append(dict(case=label,quiet=int(q.sum()),k=float(v[0]),theta=float(v[1]),status=d["status"],n_fit=d["n_fit"]))
sys.path.insert(0,str(D/"analysis/table8"))
import table8_bootstrap as T
pop=[T.read_population(io.BytesIO((D/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
T.initialize(pop,"revised")
old=json.loads((R.parent/"4.4/verification.json").read_text())["table8"]
for i,sh in enumerate(T.SHEETS):
 for j in (0,1):
  v,f,e=T.estimate(i,j);np.testing.assert_allclose(v,old[sh][j][0],rtol=1e-12)
  assert f==e==0
(R/"verification.json").write_text(json.dumps(dict(cases=rows,table8_points_and_first_draw="identical to m97",full_999_draws_rerun=False),indent=2))
print("Table 8 matched.",flush=True)
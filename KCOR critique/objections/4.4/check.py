from pathlib import Path
import os,sys,json,io
import numpy as np
R=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m97"
os.environ["DATASET"]="Czech"
sys.path.insert(0,str(D/"code"))
import KCOR as K
t=np.arange(60,dtype=float);g=.0016
B=.002*np.expm1(g*np.arange(61))/g;W=.006*np.minimum(np.arange(61),4)
h=np.diff(np.log1p(2*(B+W))/2);q=t>=10
rows=[]
for a in (1.,1.0000001,1.000001,1.18):
 v,d=K.fit_theta0_gompertz(h,t,q,4,g,deaths_arr=np.full(60,200),wave=(0,3),alpha=a)
 assert d["success"] and d["n_fit"]==50
 rows.append(dict(alpha=a,k=v[0],theta=v[1],n_fit=d["n_fit"]))
assert abs(rows[0]["theta"]-2)<1e-5 and abs(rows[1]["theta"]-2)<1e-5
# Execute actual profiled gap-loop from source with a controlled raw offset sequence.
import textwrap
src=(D/"code/KCOR.py").read_text(encoding="utf8")
a=src.index("                prev = 0.\n                for start in reentry_indices:")
b=src.index("            pred = weekly_gamma_hazard",a)
code=textwrap.dedent(src[a:b])
ns=dict(reentry_indices=[0,1,2],inv=np.array([0.,-.001,0.]),k=0.,unit_start=np.zeros(3),offset=np.zeros(3))
exec(code,ns)
assert np.allclose(ns["offset"],0)
# 4.1 regression at both proportional and nonlinear exponent.
g=.085/52;t=np.arange(24,dtype=float);H=.002*np.expm1(g*np.arange(25))/g;h=np.diff(np.log1p(2*H)/2)
for alpha in (1.,1.18):
 for n in range(4):
  v,d=K.fit_theta0_gompertz(h,t,t<12+n,4,g,deaths_arr=np.full(24,200),wave=(0,11),alpha=alpha)
  assert d["n_fit"]==n and d["success"]==(n>=3)
sys.path.insert(0,str(D/"analysis/table8"))
import table8_bootstrap as T
pop=[T.read_population(io.BytesIO((D/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
T.initialize(pop,"revised")
res={}
for i,sh in enumerate(T.SHEETS):
 res[sh]=[T.estimate(i,j) for j in (0,1)]
out=dict(alpha_limit=rows,gap_example=ns["offset"].tolist(),table8=res)
(R/"verification.json").write_text(json.dumps(out,indent=2))
print(out,flush=True)
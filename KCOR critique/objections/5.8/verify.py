from pathlib import Path
import os,sys,json
import numpy as np
R=Path(__file__).resolve().parent;D=Path(os.environ["TEMP"])/"kcor-s4-approved"/sys.argv[1]
os.environ["DATASET"]="Czech";sys.path.insert(0,str(D/"code"))
import KCOR as K
t=np.arange(24,dtype=float);g=.085/52;k=.001;theta=2.
H=k*np.expm1(g*np.arange(25))/g
h=np.diff(np.log1p(theta*H)/theta)
h[:4]*=1.15;h[8:12]*=3.
quiet=(t<8)|(t>=12)
v,d=K.fit_theta0_gompertz(h,t,quiet,4,g,deaths_arr=np.full(24,200))
raw=float(np.expm1(v[1]*h[:12].sum())/v[1]-v[0]*np.expm1(g*12)/g)
result={"k":float(v[0]),"theta":float(v[1]),"reported_raw":d["delta_raw_values"],"reported_applied":d["delta_applied_values"],"expected_raw":raw,"status":d["status"]}
if sys.argv[1]=="m110":
 assert abs(result["reported_raw"][0]-raw)<1e-10
 assert abs(result["reported_applied"][0]-max(raw,0))<1e-10
 # Exact Gompertz curve with an arbitrary persistent offset in a gap;
 # compare each quiet interval against its own opening offset.
 t=np.arange(60,dtype=float);skip=2;k=.001;g=.002
 E=k*np.expm1(g*np.maximum(t+1-skip,0))/g
 E[t>=20]+=.05
 q=((t>=2)&(t<15))|((t>=25)&(t<55))
 z=K._compute_fit_diagnostics(E,t,q,k,g,skip)
 assert len(z["shape_windows"])==2 and z["max_abs_dev"]<1e-12
 # Deliberate curvature change in second quiet window must be detected.
 E2=E.copy();E2[t>=25]+=.001*(t[t>=25]-24)
 z2=K._compute_fit_diagnostics(E2,t,q,k,g,skip)
 assert z2["shape_windows"][0]["max_abs_dev"]<1e-12 and z2["shape_windows"][1]["max_abs_dev"]>.02
 result["exact_shape"]=z;result["perturbed_shape"]=z2
 # Nonlinear diagnostics must report the final survivor-profile gaps, not seed gamma gaps.
 v,d=K.fit_theta0_gompertz(h,t[:24],quiet,4,.085/52,deaths_arr=np.full(24,200),wave=(8,9),alpha=1.18)
 assert d["delta_raw_scale"]=="observed_log_survival"
 assert d["delta_raw_values"]==d["nonlinear_gap_observed_deficit"] and d["delta_applied_values"]==d["nonlinear_gap_exposure"]
 result["nonlinear"]={"raw":d["delta_raw_values"],"applied":d["delta_applied_values"],"raw_scale":d["delta_raw_scale"]}
(R/(sys.argv[1]+"-diagnostic-check.json")).write_text(json.dumps(result,indent=2))
print(result)

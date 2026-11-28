from pathlib import Path
import os,sys,numpy as np,json
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m111"; os.environ["DATASET"]="Czech";sys.path.insert(0,str(D/"code"))
import KCOR as K
t=np.arange(60.);g=.085/52;B=.001*np.expm1(g*np.arange(61))/g;p=-np.expm1(-np.diff(np.log1p(2*B)/2))
rng=np.random.default_rng(0);a=[];d=[];n=1500
for x in p:
 a.append(n);j=rng.binomial(n,x);d.append(j);n-=j
a=np.array(a);d=np.array(d);h=-np.log1p(-d/a)
v,diag=K.fit_theta0_gompertz(h,t,np.ones(60,bool),4,g,deaths_arr=d,risk_arr=a)
assert abs(v[1]-diag["theta0_raw"])<1e-10 and diag["legacy_relative_error_flag"] and not diag["sampling_flag"] and diag["sampling_n"]==60
print(v,diag["sampling_min_tail"],diag["sampling_threshold"])
# check exact tail for a large deviation and zero count.
f={"fit_mask_theta":np.array([True,True]),"fitted_weekly_hazard":-np.log1p(-np.array([.001,.02]))}
K._sampling_diagnostics([0,0],[1000,1000],[0,1],f)
assert not f["sampling_weeks"][0]["flag"] and f["sampling_weeks"][1]["flag"]
res={"retained_theta":float(v[1]),"status":diag["status"],"legacy_flag":diag["legacy_relative_error_flag"],"sampling_flag":diag["sampling_flag"],"minimum_tail":diag["sampling_min_tail"],"threshold":diag["sampling_threshold"],"examples":f["sampling_weeks"]}
Path("KCOR critique/objections/5.8/sampling-verification.json").write_text(json.dumps(res,indent=2))
print(res)

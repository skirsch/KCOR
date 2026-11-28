from pathlib import Path
import os,sys,json
import numpy as np
D=Path(os.environ["TEMP"])/"kcor-s4-approved"/sys.argv[1]
os.environ["DATASET"]="Czech"
sys.path.insert(0,str(D/"code"))
import KCOR as K
t=np.arange(40,dtype=float); g=.002; q=t>=9
out=[]
for case,death,wave,alpha in [
 ("zero-anchor",np.where(t<2,0,10),None,None),
 ("zero-anchor-wave",np.where(t<2,0,10),(5,8),1.18),
 ("positive-anchor",np.full(40,10),None,None),
 ("all-zero",np.zeros(40),None,None),
 ("below-minimum",np.where((t>=9)&(t<20),1,0),None,None)]:
 risk=15000-np.r_[0,np.cumsum(death[:-1])]
 h=-np.log1p(-death/risk)
 v,d=K.fit_theta0_gompertz(h,t,q,2,g,deaths_arr=death,wave=wave,alpha=alpha)
 out.append(dict(case=case,k=float(v[0]),theta=float(v[1]),status=d["status"],message=d["message"],n_fit=d["n_fit"],success=bool(d["success"]),quiet_deaths=float(d["total_quiet_deaths"]),raw_theta=float(d["theta0_raw"])))
if sys.argv[1]=="m104":
 assert out[0]["n_fit"]==33 and out[0]["quiet_deaths"]==310 and np.isfinite(out[0]["k"]) and out[0]["message"]!="k_anchor_too_small"
 assert out[1]["quiet_deaths"]==310 and np.isfinite(out[1]["k"]) and out[1]["message"]!="k_anchor_too_small"
 assert not out[3]["success"]
 assert out[4]["status"]=="insufficient_deaths"
Path("KCOR critique/objections/5.2",sys.argv[1]+"-regression.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out))

from pathlib import Path
import os,sys,json
import numpy as np
D=Path(os.environ["TEMP"])/"kcor-s4-approved"/sys.argv[1]
os.environ["DATASET"]="Czech"
sys.path.insert(0,str(D/"code"))
import KCOR as K
def check(name,h,t,q,a,w,alpha,deaths,g=.002):
 v,d=K.fit_theta0_gompertz(h,t,q,a,g,deaths_arr=deaths,wave=w,alpha=alpha)
 return dict(name=name,alpha=alpha,status=d["status"],success=bool(d["success"]),n_fit=d["n_fit"],n_quiet=d["n_quiet_bins"],deaths=float(d["total_quiet_deaths"]),weak=bool(d["identifiability_flag"]),insufficient_deaths=bool(d["insufficient_deaths"]),theta=float(v[1]))
t=np.arange(40,dtype=float); w=(5,18); marked=(t>=5)&(t<=18)
out=[]
for alpha in (1.,1.18):
 for extra in (0,4,20):
  deaths=np.where(marked,4,1).astype(float)
  deaths[-1]+=extra
  risk=15000-np.r_[0,np.cumsum(deaths[:-1])]
  out.append(check(f"eligible-{26+extra}",-np.log1p(-deaths/risk),t,np.ones(40,bool),2,w,alpha,deaths))
t=np.arange(30,dtype=float)
out.append(check("anchor-only",.001*np.exp(.0015*t),t,(t>=8)&(t<=15),4,(8,15),1.18,np.full(30,10)))
t=np.arange(30,dtype=float)
out.append(check("five-eligible",.001*np.exp(.0015*t),t,(t>=8)&(t<=18),2,(8,15),1.18,np.full(30,20)))
t=np.arange(24,dtype=float)
out.append(check("zero-rows",.001*np.exp(.0015*t),t,t<12,4,(0,11),1.18,np.full(24,200)))
t=np.arange(60,dtype=float);g=.0016; B=.002*np.expm1(g*np.arange(61))/g; W=.006*np.minimum(np.arange(61),4)
h=np.diff(np.log1p(2*(B+W))/2)
for alpha in (1.,1.0000001):
 out.append(check("continuity",h,t,t>=10,4,(0,3),alpha,np.full(60,200),g))
if sys.argv[1]=="m103":
 for x in out:
  if x["name"]=="eligible-26": assert x["status"]=="insufficient_deaths" and x["deaths"]==26 and x["n_quiet"]==26
  if x["name"] in ("eligible-30","eligible-46"): assert not x["insufficient_deaths"] and x["deaths"] in (30,46)
  if x["name"] in ("anchor-only","zero-rows"): assert not x["success"] and x["n_quiet"]==0 and x["deaths"]==0
  if x["name"]=="five-eligible": assert x["n_fit"]==5 and x["n_quiet"]==3 and x["deaths"]==60 and x["weak"]
  if x["name"]=="continuity": assert x["n_fit"]==50 and abs(x["theta"]-2)<1e-3
p=Path("KCOR critique/objections/5.1")/(sys.argv[1]+"-regression.json")
p.write_text(json.dumps(out,indent=2))
print(json.dumps(out))

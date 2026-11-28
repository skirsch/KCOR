"""Adjudicate specific counterexamples against series 4 frozen candidate."""
from pathlib import Path
import os,sys,json
import numpy as np
from scipy.stats import binom
A=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m94"
os.environ["DATASET"]="Czech"
sys.path.insert(0,str(D/"code"))
import KCOR as K
import kcor_survivor as S

t=np.arange(24,dtype=float);g=.085/52;k=.002;theta=2.
H=k*np.expm1(g*np.arange(25))/g
h=np.diff(np.log1p(theta*H)/theta);quiet=t<12
value,diag=K.fit_theta0_gompertz(h,t,quiet,4,g,deaths_arr=np.full(24,200),wave=(0,11),alpha=1.18)
mask=(quiet|(t<4))&~((t>=0)&(t<=11))
no_rows=dict(eligible=int(mask.sum()),reported_n_fit=diag["n_fit"],success=bool(diag["success"]),status=diag["status"],frailty=float(value[1]))
assert no_rows["eligible"]==0 and no_rows["status"]=="ok"
# Enough total quiet weeks, but the first segment has fewer than three rows.
t2=np.arange(60,dtype=float);B=.002*np.expm1(g*np.arange(61))/g
h2=np.diff(np.log1p(2*B)/2);q2=(t2<2)|((t2>=20)&(t2<51))
v2,d2=K.fit_theta0_gompertz(h2,t2,q2,4,g,deaths_arr=np.full(60,200))
short_seed=dict(total_quiet=int(q2.sum()),first_window=2,status=d2["status"],success=bool(d2["success"]),message=d2["message"])
assert not short_seed["success"]
# Continuity of the fitted estimator at alpha=1 with a marked wave in the fitting anchors.
t3=np.arange(60,dtype=float);g3=.0016;k3=.002;th3=2.
B3=k3*np.expm1(g3*np.arange(61))/g3;W3=.006*np.minimum(np.arange(61),4)
h3=np.diff(np.log1p(th3*(B3+W3))/th3);q3=t3>=10
limit=[]
for alpha in (1.,1.0000001,1.000001,1.18):
 v3,d3=K.fit_theta0_gompertz(h3,t3,q3,4,g3,deaths_arr=np.full(60,200),wave=(0,3),alpha=alpha)
 limit.append(dict(alpha=alpha,k=float(v3[0]),theta=float(v3[1]),n_fit=int(d3["n_fit"]),status=d3["status"]))
# Exact expectation over finite binomial death counts for a marked week with no true wave.
def expectation(N,b,alpha):
 p=b/(1+b);ks=np.arange(int(binom.ppf(1-1e-13,N,p))+1);weights=binom.pmf(ks,N,p)
 vals=[]
 for j in ks:
  q=-np.log1p(-j/N)
  if q>S.cumulative_hazard(b,0,1.,alpha):
   W=S.solve_increase(lambda x:S.cumulative_hazard(b,x,1.,alpha),q,.001);B=b
  else:B=np.expm1(q);W=0.
  vals.append((B,W,B+W,np.expm1(q)))
 m=np.array(vals).T@weights
 return dict(N=N,theta=1.,alpha=alpha,true_B=b,true_W=0.,prob=float(weights.sum()),mean_B=float(m[0]),mean_W=float(m[1]),mean_corrected=float(m[2]),mean_proportional=float(m[3]))
waves=[expectation(N,.001,a) for N in (5000,50000,500000) for a in (1.18,2.)]
# Reproduce the raw-difference clamp identity in the proportional gap objective.
raw=np.array([0.,-.001,0.]);increments=np.maximum(np.diff(raw,prepend=0.),0.)
report=dict(no_nonlinear_fit_rows=no_rows,short_first_window=short_seed,alpha_limit=limit,marked_zero_wave=waves,
            gap_offset_example=dict(raw=raw.tolist(),applied=np.cumsum(increments).tolist()))
(A/"counterexamples.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))
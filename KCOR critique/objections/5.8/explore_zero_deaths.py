from pathlib import Path
import os,sys,json
import numpy as np
from scipy.stats import binom
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m110";os.environ["DATASET"]="Czech";sys.path.insert(0,str(D/"code"))
import KCOR as K
t=np.arange(60,dtype=float);g=.085/52;k=.001;th=2.
B=k*np.expm1(g*np.arange(61))/g
q=np.diff(np.log1p(th*B)/th);p=-np.expm1(-q)
for seed in range(100):
 rng=np.random.default_rng(seed);a=np.empty(60,int);d=np.empty(60,int);n=1500
 for j in range(60):
  a[j]=n;d[j]=rng.binomial(n,p[j]);n-=d[j]
 obs=-np.log1p(-d/a)
 v,diag=K.fit_theta0_gompertz(obs,t,np.ones(60,bool),4,g,deaths_arr=d)
 if diag.get("bad_fit") and not diag.get("theta_hit_bound") and np.isfinite(diag.get("theta0_raw",np.nan)):break
else:raise RuntimeError("No isolated zero-event flag")
pred=K.weekly_gamma_hazard(v[0]*np.expm1(g*t)/g,v[0]*np.exp(g*t)*K._gompertz_week_factor(g),diag["theta0_raw"])
ph=-np.expm1(-pred)
zero=d==0
assert zero.any()
z=(d-a*ph)/np.sqrt(a*ph*(1-ph))
result={"population":1500,"weeks":60,"seed":seed,"deaths":int(d.sum()),"zero_death_weeks":int(zero.sum()),"status":diag["status"],"relRMSE":float(diag["relRMSE_hazard_fit"]),"expected_deaths_zero_weeks_true":[float((a*p)[zero].min()),float((a*p)[zero].max())],"zero_probability_true":[float(binom.pmf(0,a[zero],p[zero]).min()),float(binom.pmf(0,a[zero],p[zero]).max())],"zero_week_pearson_residual_fitted":[float(z[zero].min()),float(z[zero].max())],"max_abs_pearson_residual":float(np.max(np.abs(z))),"pearson_over_df":float(np.sum(z*z)/(len(z)-2)),"fit_theta_raw":float(diag["theta0_raw"]),"fit_theta_applied":float(v[1])}
# Original 310-death counterexample: approximate zero-death probability under fitted rate.
result["earlier_310_death_example"]={"fitted_expected_deaths_first_week":float(15000*(-np.expm1(-.0006085237574349148*K._gompertz_week_factor(.002)))),"probability_zero_first_week":float(np.exp(-15000*.0006085237574349148*K._gompertz_week_factor(.002)))}
Path("KCOR critique/objections/5.8/zero-death-exploration.json").write_text(json.dumps(result,indent=2))
print(result)

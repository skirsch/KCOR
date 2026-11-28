from pathlib import Path
import os,sys,json,io
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m99"
OUT=Path(os.environ["TEMP"])/"kcor-ordinary-bootstrap-check"
os.environ["DATASET"]="positive_control";os.environ["SA_DOSE_PAIRS"]="1,0"
sys.path.insert(0,str(D/"code"))
import KCOR as K
import bootstrap_reporting as B
cfg=K.get_gompertz_theta_fit_config();guard=K.get_theta_degenerate_fit_config()
rows={}
for scenario in ("benefit","harm"):
 f=pd.read_excel(OUT/(scenario+".xlsx"));h=B.histories(f,"2021_24","disjoint")
 def direct(g):
  a=g.groupby(["Dose","DateDied"])[["Alive","Dead"]].sum()
  dates=pd.DatetimeIndex(a.loc[0].index);iso=dates.strftime("%G%V").astype(int).to_numpy()
  weeks=np.arange(len(dates));quiet=K._build_theta_quiet_mask(iso,weeks)[0]
  curves=[]
  for dose in (0,1):
   z=a.loc[dose];haz=K.hazard_from_mr(z.Dead.to_numpy()/(z.Alive.to_numpy()+K.EPS))
   eff=np.where(weeks>=K.DYNAMIC_HVE_SKIP_WEEKS,haz,0.)
   (k,th),d=K.fit_theta0_gompertz(eff,weeks-K.DYNAMIC_HVE_SKIP_WEEKS,quiet,cfg["k_anchor_weeks"],cfg["gamma_per_week"],deaths_arr=z.Dead.to_numpy(),iso_week=iso)
   th=th if d["success"] and np.isfinite(th) and th>=0 else 0.
   th,_,_=K._apply_theta_degenerate_guard(th,guard)
   y=K.invert_gamma_frailty(eff.cumsum(),th)
   curves.append(y[80]/y[K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE])
  return curves[1]/curves[0]
 point=direct(f)
 vals=np.array([direct(B.draw_history(h,np.random.default_rng([33080,j,0]))) for j in range(1,100)])
 expected=np.r_[point,np.quantile(vals,[.025,.975])]
 wb=pd.read_excel(OUT/(scenario+"-verified.xlsx"),sheet_name="dose_pairs")
 w=wb[(wb.Date==pd.Timestamp("2022-12-26"))&(wb.YearOfBirth==1950)&(wb.Dose_num==1)&(wb.Dose_den==0)].iloc[0]
 actual=w[["KCOR","CI_lower","CI_upper"]].to_numpy(float)
 np.testing.assert_allclose(actual,expected,rtol=1e-10)
 assert w.CI_method=="person_bootstrap_percentile" and w.CI_valid_draws==99 and w.CI_flagged_draws==0
 rows[scenario]=dict(point=float(actual[0]),bootstrap=actual[1:].tolist(),analytic=[float(w.CI_analytic_lower),float(w.CI_analytic_upper)],draws=99,flags=int(w.CI_flagged_draws),independent_max_abs_difference=float(np.max(abs(actual-expected))))
 print(scenario,rows[scenario],flush=True)
(R/"verification.json").write_text(json.dumps(rows,indent=2))
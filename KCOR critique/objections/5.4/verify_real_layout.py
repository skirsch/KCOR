from pathlib import Path
import os,sys,json,pandas as pd,numpy as np
r=Path(__file__).resolve().parent
d=Path(os.environ["TEMP"])/"kcor-s4-approved/m106";sys.path.insert(0,str(d/"code"))
import bootstrap_reporting as B
res={}
for data,sh in [("Czech","2021_24"),("Czech","2022_06"),("FOIA","2022_47")]:
 f=pd.read_excel(d/"data"/data/"KCOR_CMR.xlsx",sheet_name=sh)
 h=B.histories(f,sh,"disjoint")
 x=B.draw_history(h,np.random.default_rng(33080))
 base=x[~x.YearOfBirth.isin([-2,-60])]
 for age in h[2].YearOfBirth.unique():
  cols=["Alive","Dead"]+(["Dead_COVID"] if "Dead_COVID" in base else [])
  a=x[x.YearOfBirth==age].groupby(["Dose","DateDied"])[cols].sum().sort_index()
  b=(base if age==-2 else base[base.YearOfBirth>=1960]).groupby(["Dose","DateDied"])[cols].sum().sort_index()
  pd.testing.assert_frame_equal(a,b,check_dtype=False)
 people=int(base[base.DateDied==base.DateDied.min()].Alive.sum())
 assert people==sum(int(c.sum()) for _,c in h[1])
 res[data+" "+sh]={"cohorts":len(h[1]),"people":people,"aggregate_rows":len(h[2]),"draw_checked":True}
 print(data,sh,res[data+" "+sh],flush=True)
(r/"real-history-verification.json").write_text(json.dumps(res,indent=2))

from pathlib import Path
import os,sys,json,importlib.util
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved"
sys.path.insert(0,str(D/"m106/code"))
import bootstrap_reporting as B
spec=importlib.util.spec_from_file_location("old_bootstrap",D/"m105/code/bootstrap_reporting.py"); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def make_sheet(sh):
 dates=pd.date_range(pd.Timestamp.fromisocalendar(*map(int,sh.split("_")),1),periods=100,freq="7D")
 rows=[];rng=np.random.default_rng(52)
 for age,dose,sex,dcci in [(a,d,s,c) for a in (1950,1960) for d in (0,1) for s in ("F","M","O") for c in (0,2)]:
  n=9000; alive=n; previous=0
  for date in dates:
   dead=int(rng.binomial(alive,.0015*(1+.1*dose+.06*dcci+.04*(age==1950))))
   covid=int(rng.binomial(dead,.1))
   rows.append(dict(ISOweekDied=date.strftime("%G-%V"),YearOfBirth=age,Sex=sex,DCCI=dcci,Dose=dose,DateDied=date,Dead=dead,Dead_COVID=covid,cumDead_prev=previous,Alive=alive))
   alive-=dead;previous+=dead
 base=pd.DataFrame(rows)
 allrows=[base]
 for age in (-2,-60):
  x=base if age==-2 else base[base.YearOfBirth>=1960]
  q=x.groupby(["ISOweekDied","DateDied","Dose"],as_index=False)[["Alive","Dead","Dead_COVID","cumDead_prev"]].sum()
  q["YearOfBirth"]=age;q["Sex"]="O";q["DCCI"]=-1
  allrows.append(q[base.columns])
 return pd.concat(allrows,ignore_index=True)
def check_aggregate(x):
 base=x[x.YearOfBirth>0]; cols=["Alive","Dead","Dead_COVID","cumDead_prev"]
 for age in (-2,-60):
  a=x[x.YearOfBirth==age].set_index(["Dose","DateDied"])[cols].sort_index()
  b=(base if age==-2 else base[base.YearOfBirth>=1960]).groupby(["Dose","DateDied"])[cols].sum().sort_index()
  pd.testing.assert_frame_equal(a,b,check_dtype=False)
  assert (x.Dead_COVID<=x.Dead).all()
 for _,g in base.groupby(["YearOfBirth","Dose","Sex","DCCI"]):
  g=g.sort_values("DateDied");assert (g.Alive.iloc[:-1].to_numpy()-g.Dead.iloc[:-1].to_numpy()==g.Alive.iloc[1:].to_numpy()).all()
frame=make_sheet("2021_24");check_aggregate(frame)
h=B.histories(frame,"2021_24","disjoint")
assert len(h[1])==4 and len(h[2])==400
expect=frame[(frame.YearOfBirth>0)&(frame.DateDied==frame.DateDied.min())].Alive.sum()
assert sum(int(c.sum()) for _,c in h[1])==expect
for i in range(10):
 x=B.draw_history(h,np.random.default_rng(100+i));check_aggregate(x)
 assert x[(x.YearOfBirth>0)&(x.DateDied==x.DateDied.min())].Alive.sum()==expect
try: old.histories(frame,"2021_24","disjoint")
except ValueError as e: old_error=str(e)
else: raise AssertionError("old accepted converter layout")
broken=make_sheet("2021_20")
broken.loc[broken[broken.YearOfBirth==-2].index[0],"Alive"]+=1
try: B.histories(broken,"2021_20","disjoint")
except ValueError as e: bad_error=str(e)
else: raise AssertionError("mismatched aggregate accepted")
# Existing simple layout draws must be bit-for-bit unchanged.
simple=frame[(frame.YearOfBirth>0)&(frame.Sex!="O")].groupby(["DateDied","ISOweekDied","YearOfBirth","Dose","Sex"],as_index=False)[["Alive","Dead"]].sum()
a=old.draw_history(old.histories(simple,"2021_24","disjoint"),np.random.default_rng(123))
b=B.draw_history(B.histories(simple,"2021_24","disjoint"),np.random.default_rng(123))
pd.testing.assert_frame_equal(a,b)
# Stored Table 8 layout remains on its separate path.
t=pd.read_excel(D/"m106/analysis/table8/input.xlsx",sheet_name="2021_24")
a=old.draw_history(old.histories(t,"2021_24","table8"),np.random.default_rng(123))
b=B.draw_history(B.histories(t,"2021_24","table8"),np.random.default_rng(123))
pd.testing.assert_frame_equal(a,b)
with pd.ExcelWriter(R/"converter-layout-test.xlsx") as w:
 broken.to_excel(w,sheet_name="2021_20",index=False)
 frame.to_excel(w,sheet_name="2021_24",index=False)
res=dict(old_error=old_error,bad_error=bad_error,disjoint_people=int(expect),draws_checked=10,aggregate_identity=True,old_simple_draw_identical=True,table8_draw_identical=True)
(R/"history-verification.json").write_text(json.dumps(res,indent=2))
print(res)


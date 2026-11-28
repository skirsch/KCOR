from pathlib import Path
import os,sys,json,pandas as pd,numpy as np
d=Path(os.environ["TEMP"])/"kcor-s4-approved/m107";sys.path.insert(0,str(d/"code"))
import bootstrap_reporting as B
r=Path("KCOR critique/objections/5.5")
f=pd.read_excel("KCOR critique/objections/5.4/converter-layout-test.xlsx",sheet_name="2021_24")
h=B.histories(f,"2021_24","disjoint")
def groups(x):
 base=x[x.YearOfBirth>0]; first=base[base.DateDied==base.DateDied.min()]
 return first.groupby(["YearOfBirth","Dose"]).Alive.sum().sort_index()
base=groups(f)
for seed in range(10):
 x=B.draw_history(h,np.random.default_rng(seed))
 pd.testing.assert_series_equal(groups(x),base,check_dtype=False)
# Explicit exit-bearing history: 100 alive, 3 deaths + 2 exits, etc.
dates=pd.date_range("2021-06-14",periods=3,freq="7D")
g=pd.DataFrame(dict(DateDied=dates,ISOweekDied=dates.strftime("%G-%V"),YearOfBirth=1950,Dose=0,Sex="F",Alive=[100,95,91],Dead=[3,2,1]))
z=B.histories(g,"2021_24","disjoint")
assert z[1][0][1].tolist()==[3,2,1,2,2,90]
for seed in range(10):
 draw=B.draw_history(z,np.random.default_rng(seed))
 exit_counts=draw.Alive.to_numpy()[:-1]-draw.Dead.to_numpy()[:-1]-draw.Alive.to_numpy()[1:]
 assert (exit_counts>=0).all() and draw.Alive.iloc[0]==100
# Pooled overlapping Table 8 components vary in size but total stays fixed.
t=pd.read_excel(d/"analysis/table8/input.xlsx",sheet_name="2021_24")
h=B.histories(t,"2021_24","table8")
original=h[2].sum(axis=2)
draws=[]
for seed in range(10):
 sampled=B.draw_history(h,np.random.default_rng(seed))
 a=sampled[sampled.DateDied==sampled.DateDied.min()].pivot(index="YearOfBirth",columns="Dose",values="Alive").to_numpy(copy=True)
 a[2]-=a[0]+a[1]
 assert a.sum()==original.sum()
 draws.append(a.tolist())
assert len({str(x) for x in draws})>1
res={"ordinary_fixed_group_size":True,"ordinary_draws_tested":10,"exit_cells":z[1][0][1].tolist(),"table8_fixed_total":int(original.sum()),"table8_component_sizes_vary":True,"table8_draws_tested":10}
(r/"verification.json").write_text(json.dumps(res,indent=2))
print(res)


from pathlib import Path
import os,sys,numpy as np,pandas as pd
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m109"
os.environ["DATASET"]="Czech";sys.path.insert(0,str(D/"code"))
import KCOR as K
from kcor_survivor import Survival
dates=pd.date_range("2021-06-14",periods=142,freq="7D")
t=np.arange(len(dates)); skip=K.DYNAMIC_HVE_SKIP_WEEKS
g=.085/52; time=np.maximum(np.arange(len(dates)+1)-skip,0)
wave=(dates>=pd.Timestamp("2021-10-11"))&(dates<=pd.Timestamp("2022-03-21"))
W=np.r_[0.,np.cumsum(np.where(wave,.004,0.))]
B=.002*np.expm1(g*time)/g
# Before the stabilization skip, positive observed mortality is present but excluded from modeling.
prior=.002*np.minimum(np.arange(len(dates)+1),skip)
rows=[]
for dose,theta in ((0,.5),(1,4.)):
 surv=Survival(theta,1.18)
 H=surv.hazard(B,W)+prior
 n=5000000; alive=np.rint(n*np.exp(-H)).astype(int); dead=alive[:-1]-alive[1:]
 for j,date in enumerate(dates):
  rows.append(dict(DateDied=date,ISOweekDied=date.strftime("%G-%V"),YearOfBirth=1950,Sex="F",Dose=dose,Alive=int(alive[j]),Dead=int(dead[j])))
base=pd.DataFrame(rows)
agg=base.copy();agg.YearOfBirth=-2;agg.Sex="O"
f=pd.concat([base,agg],ignore_index=True)
out=Path("KCOR critique/objections/5.7/wave-reporting-input.xlsx")
f.to_excel(out,sheet_name="2021_24",index=False)
print("generated",len(f),"rows", "skip",skip)

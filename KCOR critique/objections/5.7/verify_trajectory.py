from pathlib import Path
import os,sys,tempfile,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;D=Path(os.environ["TEMP"])/"kcor-s4-approved"/sys.argv[1]
os.environ["DATASET"]="Czech";os.environ["KCOR_NPH_MODE"]="forced_alpha";os.environ["KCOR_BOOTSTRAP_DRAWS"]="0";os.environ["SA_DOSE_PAIRS"]="1,0";os.environ["MONTE_CARLO"]="0"
sys.path.insert(0,str(D/"code"))
import KCOR as K
K.OVERRIDE_DOSE_PAIRS=[(1,0)]
saved={}
original=K.build_kcor_rows
def capture(df,sheet_name,*args,**kwargs):
 saved.update(df=df.copy(),sheet=sheet_name,params=args[2].copy() if len(args)>2 and args[2] is not None else {})
 return original(df,sheet_name,*args,**kwargs)
K.build_kcor_rows=capture
with tempfile.TemporaryDirectory(prefix="kcor-allage-mc-") as tmp:
 out=K.process_workbook(str(R/"wave-reporting-input.xlsx"),str(Path(tmp)/"unused.xlsx"),str(Path(tmp)/"run.log"),return_rows=True)
K.build_kcor_rows=original
base=saved["df"]; sheet=saved["sheet"]; params=saved["params"]
# Check per-dose corrected trajectory, not just the final ratio.
def check_report(out,frame):
 out=out[out.YearOfBirth==-2].sort_values(["mc_id","Date"],na_position="first")
 errs=[]
 for _,g in out.groupby("mc_id",dropna=False):
  mc=g.mc_id.iloc[0]
  sub=frame[frame.YearOfBirth==-2]
  if "mc_id" in sub: sub=sub[sub.mc_id==mc]
  for dose,label in [(1,"num"),(0,"den")]:
   ref=sub[sub.Dose==dose].sort_values("DateDied")
   assert len(ref)==len(g)
   errs.append(float(np.max(np.abs(g["adj_cum_hazard_"+label].to_numpy()-ref.CH.to_numpy()))))
 return max(errs)
ordinary=check_report(out,base)
# Seed two separate Monte Carlo reporting histories from actual normalized data.
# Perturb the second corrected trajectory while preserving each row's draw ID.
K.MONTE_CARLO_MODE=True
K.MC_INCLUDE_AGGREGATES=True
K.MC_YOB_START=1940;K.MC_YOB_END=1960
frames=[];m={};slopes={}
for draw in (11,22):
 z=base.copy();z["mc_id"]=draw
 if draw==22:
  z["CH"]=z["CH"]*(1+.002*z["t"]*(z.Dose+1))
  z["hazard_adj"]=z.groupby(["YearOfBirth","Dose"])["CH"].diff().fillna(z["CH"])
  z["hazard_eff"]=z["hazard_adj"]
 for key,p in params.items():
  if len(key)==3 and key[0]==sheet:
   m[(sheet,draw,key[1],key[2])]=p.copy()
   if key[1]==-2: slopes[(sheet,draw,key[1],key[2])]={"abnormal_fit":draw==22}
 frames.append(z)
f=pd.concat(frames,ignore_index=True)
rows=original(f,sheet,slope6_params_map=slopes,kcor6_params_map=m)
mc=check_report(rows,f)
result={"ordinary_max_trajectory_error":ordinary,"mc_max_trajectory_error":mc,"mc_ids":sorted(rows[rows.YearOfBirth==-2].mc_id.unique().tolist())}
if sys.argv[1]=="m109":
 assert ordinary<1e-12 and mc<1e-12
 for draw in (11,22):
  z=rows[(rows.YearOfBirth==-2)&(rows.mc_id==draw)]
  assert z.abnormal_fit.eq(True if draw==22 else "").all()
(R/(sys.argv[1]+"-trajectory-check.json")).write_text(json.dumps(result,indent=2))
print(result)

from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd
d=Path(__file__).resolve().parent; D=Path(os.environ["TEMP"])/"kcor-s4-approved/m105"
os.environ["DATASET"]="positive_control";sys.path.insert(0,str(D/"code"))
import KCOR as K
def test_ref():
 a=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
 def frame(times,numer,denom):
  return pd.DataFrame(dict(t_num=times,t_den=times,DateDied=pd.to_datetime("2021-06-14")+pd.to_timedelta(np.asarray(times)*7,unit="D"),CH_num=numer,CH_den=denom))
 for times,numer,denom,expected in [
  ([a-1,a,a+1],[1,2,6],[1,2,3],"ok"),
  ([a-1,a,a+1],[0,0,6],[1,2,3],"undefined_reference_ratio"),
  ([a-1,a,a+1],[1,2,6],[0,0,3],"undefined_reference_ratio"),
  ([a-1,a+1],[1,6],[1,3],"missing_reference_week")]:
  f=frame(times,numer,denom);K._set_kcor_reference(f)
  assert f.anchor_status.iloc[-1]==expected and f.KCOR_unanchored.iloc[-1]==2
  assert bool(np.isfinite(f.KCOR.iloc[-1]))==(expected=="ok")
test_ref()
def last(name,y):
 f=pd.read_excel(d/(name+".xlsx"),sheet_name="dose_pairs")
 z=f[(f.YearOfBirth==y)&(f.Dose_num==1)&(f.Dose_den==0)].iloc[-1]
 keys=["KCOR","KCOR_anchored","KCOR_unanchored","anchor_status","CI_lower","CI_upper","CI_undefined_anchor_draws","CI_missing_draws","CI_unanchored_lower","CI_unanchored_upper","CI_unanchored_valid_draws"]
 return {k:(None if pd.isna(z[k]) else z[k].item() if hasattr(z[k],"item") else z[k]) for k in keys}
rows={"valid":last("valid-out",1950),"invalid":last("invalid-final",1950),"mixed":last("mixed-final",1950),"aggregate":last("aggregate-out",-2),"invalid_pooled":last("invalid-final",0),"mixed_pooled":last("mixed-final",0)}
assert rows["valid"]["KCOR"]==rows["valid"]["KCOR_anchored"] and rows["valid"]["CI_lower"] is not None
assert rows["invalid"]["KCOR"] is None and rows["invalid"]["CI_undefined_anchor_draws"]==3 and rows["invalid"]["CI_unanchored_valid_draws"]==3
assert rows["mixed"]["KCOR"] is not None and rows["mixed"]["CI_lower"] is None and 0<rows["mixed"]["CI_undefined_anchor_draws"]<12 and rows["mixed"]["CI_unanchored_valid_draws"]==12
assert rows["aggregate"]["KCOR"] is None and rows["aggregate"]["KCOR_unanchored"] is not None
assert rows["invalid_pooled"]["anchor_status"]=="pooled_no_valid_reference" and rows["invalid_pooled"]["CI_undefined_anchor_draws"]==3
assert rows["mixed_pooled"]["CI_undefined_anchor_draws"]==rows["mixed"]["CI_undefined_anchor_draws"] and rows["mixed_pooled"]["CI_lower"] is None
p=Path("KCOR critique/objections/5.3/verification.json"); p.write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))

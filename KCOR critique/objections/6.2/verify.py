import argparse,os,sys,tempfile,numpy as np,pandas as pd
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument("candidate",type=Path);args=parser.parse_args()
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="identified_only",KCOR_BOOTSTRAP_DRAWS="0",SA_DOSE_PAIRS="1,0",MONTE_CARLO="0")
sys.path.insert(0,str(args.candidate/"code"));import KCOR as K
from bootstrap_reporting import flagged
K.OVERRIDE_DOSE_PAIRS=[(1,0)];K.YEAR_RANGE=None
f=pd.DataFrame({"YearOfBirth":[1930,1970,-60,-2,0,-3],"abnormal_fit":[True,False,False,False,True,False]})
assert flagged(f).tolist()==[True,False,False,False,True,False]
orig=K.build_kcor_rows
def capture(df,sheet,*args,**kwargs):
    params=args[2] if len(args)>2 else kwargs["kcor6_params_map"]
    for key,v in params.items():
        if key[-2]==1930:v["theta0_status"]="INSUFFICIENT_SIGNAL"
    return orig(df,sheet,*args,**kwargs)
K.build_kcor_rows=capture
src=pd.read_excel(Path(__file__).resolve().parent.parent/"5.7/wave-reporting-input.xlsx")
base=src[src.YearOfBirth==1950].copy()
older=base.copy();older["YearOfBirth"]=1930
younger=base.copy();younger["YearOfBirth"]=1970
post=younger.copy();post["YearOfBirth"]=-60
total=base.copy();total["YearOfBirth"]=-2;total["Alive"]*=2;total["Dead"]*=2
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    pd.concat([older,younger,post,total]).to_excel(p/"input.xlsx",sheet_name="2021_24",index=False)
    out=K.process_workbook(str(p/"input.xlsx"),str(p/"out.xlsx"),str(p/"run.log"),return_rows=True)
end=out[(out.Dose_num==1)&(out.Dose_den==0)&(out.Date==out.Date.max())].copy()
end["flag"]=flagged(end)
print(end[["YearOfBirth","KCOR","flag"]].to_string(index=False))
by_age=end.set_index("YearOfBirth").flag
assert bool(by_age.loc[1930]) and not bool(by_age.loc[1970])
assert not bool(by_age.loc[-2]) and not bool(by_age.loc[-60])
assert bool(by_age.loc[0])
if -3 in by_age.index:
    assert bool(by_age.loc[-3])

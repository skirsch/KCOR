import argparse,os,sys,tempfile,numpy as np,pandas as pd
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("candidate",type=Path)
args=parser.parse_args()
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="identified_only",KCOR_BOOTSTRAP_DRAWS="0",SA_DOSE_PAIRS="1,0",MONTE_CARLO="0")
sys.path.insert(0,str(args.candidate/"code"));import KCOR as K
K.OVERRIDE_DOSE_PAIRS=[(1,0)]
K.YEAR_RANGE=None
saved={};orig=K.build_kcor_rows
def capture(df,sheet,*args,**kwargs):
    saved["frame"]=df.copy()
    saved["params"]=args[2].copy() if len(args)>2 and args[2] is not None else kwargs.get("kcor6_params_map",{})
    return orig(df,sheet,*args,**kwargs)
K.build_kcor_rows=capture
df=pd.read_excel(Path(__file__).resolve().parent.parent/"5.7/wave-reporting-input.xlsx")
known=df[df.YearOfBirth==1950].copy()
unknown=known.copy()
unknown["YearOfBirth"]=-1
unknown["Alive"]=(known.Alive*.1).astype(int)
unknown["Dead"]=(known.Dead*.05).astype(int)
total=pd.concat([known,unknown]).groupby(["DateDied","ISOweekDied","Dose"],as_index=False)[["Alive","Dead"]].sum()
total["YearOfBirth"]=-2
total["Sex"]="O"
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    for name,frame in [("supplied",pd.concat([known,unknown,total])),("reconstructed",pd.concat([known,unknown])),("aggregate-only",total),("restricted-supplied",pd.concat([known,unknown,total]))]:
        if name=="restricted-supplied":K.YEAR_RANGE=(1920,2009)
        frame.to_excel(p/"input.xlsx",sheet_name="2021_24",index=False)
        out=K.process_workbook(str(p/"input.xlsx"),str(p/"out.xlsx"),str(p/"run.log"),return_rows=True)
        f=saved["frame"];a=f[f.YearOfBirth==-2].sort_values(["Dose","DateDied"])
        assert len(a)==len(total) and np.array_equal(a.Alive.to_numpy(),total.sort_values(["Dose","DateDied"]).Alive.to_numpy()),name
        params=saved["params"]; print("CHECK",name,[(key,v.get("theta_hat")) for key,v in params.items() if key[-2]==-2])
        assert len([key for key in params if key[-2]==-2])==2
        if name=="supplied": ref={key[-1]:v["theta_hat"] for key,v in params.items() if key[-2]==-2}
        else: assert all(abs(v["theta_hat"]-ref[key[-1]])<1e-7 for key,v in params.items() if key[-2]==-2)

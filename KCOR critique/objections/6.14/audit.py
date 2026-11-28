"""Aggregate audit of the Czech full-record product eligibility rule."""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1].parent
sys.path.insert(0,str(ROOT/"code"))
from mfg_codes import MFG_DICT,PFIZER,MODERNA
DOSES=["Prvni","Druha","Treti","Ctvrta"]
DATES=[f"Datum_{x}_davka" for x in DOSES]
CODES=[f"OckovaciLatkaKod_{x}_davka" for x in DOSES]
COLS=["Infekce","RokNarozeni","DatumUmrtiLPZ"]+DATES+CODES
ALLOWED={k for k,v in MFG_DICT.items() if v in (PFIZER,MODERNA)}
start=pd.Timestamp("2021-06-14")
def week(s,death=False):
    if death:s=s.str.replace(r"[^0-9-]","",regex=True)
    return pd.to_datetime(s+"-1",format="%G-%V-%u",errors="coerce")
def audit(path):
    totals=dict(otherwise_eligible=0,excluded_any=0,excluded_future_only=0,future_only_no_prior_first_four_dose=0,excluded_undated_only=0,future_only_before_date_consistency_check=0,no_prior_before_date_consistency_check=0)
    for f in pd.read_csv(path,usecols=COLS,dtype=str,chunksize=400000):
        eligible=pd.to_numeric(f.Infekce,errors="coerce").fillna(0).astype(int)<=1
        yob=pd.to_numeric(f.RokNarozeni.str.extract(r"(\d{4})")[0],errors="coerce")
        eligible &= yob.between(1920,2005)|yob.isna()
        death=week(f.DatumUmrtiLPZ,True)
        eligible &= death.isna() | (death>=start)
        dates=pd.DataFrame({i:week(f[col]) for i,col in enumerate(DATES)})
        codes=pd.DataFrame({i:f[col] for i,col in enumerate(CODES)})
        pre_date_check = eligible.copy()
        eligible &= ~(death.notna() & (dates[0]>death))
        bad=codes.notna() & codes.ne("") & ~codes.isin(ALLOWED)
        past=dates.lt(start)
        future=dates.ge(start)
        anybad=bad.any(axis=1)
        futureonly=(bad&future).any(axis=1)&~(bad&past).any(axis=1)
        undatedonly=anybad&~(bad&future).any(axis=1)&~(bad&past).any(axis=1)
        masks=[eligible,eligible&anybad,eligible&futureonly,eligible&futureonly&~past.any(axis=1),eligible&undatedonly,pre_date_check&futureonly,pre_date_check&futureonly&~past.any(axis=1)]
        for key,m in zip(totals,masks):totals[key]+=int(m.sum())
    return totals
def main():
    inputs=[ROOT/"data/Czech/records_100k.csv",ROOT/"data/Czech/Otevrena-data-NR-26-30-COVID-19-prehled-populace-2024-01.csv.gz"]
    result={}
    for p in inputs:
        result[p.name]=audit(p)
        print(p.name,result[p.name],flush=True)
    (Path(__file__).resolve().parent/"counts.json").write_text(json.dumps(result,indent=2),encoding="utf8")
if __name__=="__main__":main()

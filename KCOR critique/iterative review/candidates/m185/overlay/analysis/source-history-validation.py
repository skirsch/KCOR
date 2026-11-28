"""Validate the exact Czech inputs used for Table 8 and the S6 figures."""
from pathlib import Path
import pandas as pd,sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from fixed_cohort_history import invalid_components
def main():
    out={}
    for source,sheets,layout in [('analysis/table8/input.xlsx',['2021_24','2022_06'],'table8'),('data/Czech/KCOR_CMR.xlsx',['2021_24','2022_26'],'disjoint')]:
        for sh in sheets:
            x=pd.read_excel(ROOT/source,sheet_name=sh);x['DateDied']=pd.to_datetime(x.DateDied)
            start=pd.Timestamp.fromisocalendar(*map(int,sh.split('_')),1);x=x[x.DateDied>=start]
            bad=invalid_components(x,start,layout=layout)
            out[source+':'+sh]=dict(rows=len(x),declared_layout=layout,invalid_groups={str(k):str(v) for k,v in bad.items()})
            print(source,sh,len(x),'invalid groups',len(bad),flush=True)
    (ROOT/'analysis/source-history-validation.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    assert all(not v['invalid_groups'] for v in out.values())
if __name__=='__main__':main()

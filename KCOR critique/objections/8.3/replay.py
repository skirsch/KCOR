"""Regenerate the current Czech all-ages parameter sensitivity sweep and figure."""
import os,sys,subprocess,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def summarize():
    x=pd.read_csv(HERE/'current/KCOR_SA_cells.csv')
    rows=[]
    for (en,n,d),g in x.groupby(['enrollment','dose_num','dose_den'],sort=True):
        v=g.kcor.dropna()
        assert len(g)==49
        rows.append(dict(enrollment=en,dose_num=int(n),dose_den=int(d),cells=len(g),valid=len(v),minimum=float(v.min()) if len(v) else None,maximum=float(v.max()) if len(v) else None,crosses_one=bool(len(v) and v.min()<1<v.max()),report_date=g.report_date.iloc[0]))
    result=dict(comparisons=len(rows),valid_cells=int(x.kcor.notna().sum()),total_cells=len(x),crossing_comparisons=sum(a['crosses_one'] for a in rows),comparisons_with_unavailable=sum(a['valid']!=49 for a in rows),minimum=float(x.kcor.min()),maximum=float(x.kcor.max()),rows=rows)
    (HERE/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    pd.DataFrame(rows).to_csv(HERE/'ranges.csv',index=False)
    return result
def main():
    env=os.environ.copy()
    for k in ('SA_COHORTS','SA_DOSE_PAIRS','SA_YOB','SA_BASELINE_WEEKS','SA_QUIET_START_OFFSETS'):env.pop(k,None)
    env.update(DATASET='Czech',SENSITIVITY_ANALYSIS='1',KCOR_NPH_MODE='identified_only',KCOR_BOOTSTRAP_DRAWS='0',KCOR_BOOTSTRAP_LAYOUT='disjoint',MONTE_CARLO='0')
    output=HERE/'current';output.mkdir(exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'code/KCOR.py'),str(ROOT/'data/Czech/KCOR_CMR.xlsx'),str(output/'KCOR_SA.xlsx'),'Sensitivity Analysis','SA.log'],env=env,cwd=ROOT,check=True)
    result=summarize()
    assert result['comparisons']==27 and result['total_cells']==27*49
    subprocess.run([sys.executable,str(ROOT/'code/generate_sensitivity_figure.py'),'--input',str(output/'KCOR_SA.xlsx'),'--output',str(ROOT/'documentation/preprint/figures/fig_sensitivity_overview.png'),'--cohort','2021_24','--style','heatmap'],cwd=ROOT,check=True)
if __name__=='__main__': main()

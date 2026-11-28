"""Verify published pair counts and the independent forced-alpha refit subset."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'analysis/table8_diagnostics'
for mode,folder,reference in [('identified_only','current',ROOT/'analysis/gap_diagnostics/table8/identified_only-draws.npz'),('forced_alpha','archive-forced',ROOT/'analysis/ceiling500/table8/forced_alpha-draws.npz')]:
    a=np.load(D/folder/(mode+'-draws.npz'));ref=np.load(reference);np.testing.assert_array_equal(a['points'],ref['points']);report=json.loads((D/folder/(mode+'-table8.json')).read_text());g=a['group_diagnostics'];p=a['pair_diagnostics'];cats=a['diagnostic_categories']
    for i,sheet in enumerate(('2021_24','2022_06')):
        lo,hi=np.quantile(a['points'][i,1:],[.025,.975],axis=0);np.testing.assert_array_equal(lo,report[sheet]['lower']);np.testing.assert_array_equal(hi,report[sheet]['upper'])
        for j,(n,d) in enumerate(((1,0),(2,0),(2,1))):
            pair=np.take(g[i],[n,d],axis=2).any(axis=(1,2));np.testing.assert_array_equal(pair,p[i,:,j]);q=report[sheet]['diagnostics'][f'{n}/{d}']
            for k,cat in enumerate(cats):assert q[cat]['point_flag']==bool(pair[0,k]) and q[cat]['bootstrap_draws']==int(pair[1:,k].sum())
    print(mode,'all pair counts and percentile limits verified')
a=np.load(D/'forced-fresh-refit-subset.npz');b=np.load(D/'archive-forced/forced_alpha-draws.npz');mask=a['done'];assert mask.sum()>=300
for k in ('points','parameters','group_diagnostics','pair_diagnostics'):np.testing.assert_array_equal(a[k][mask],b[k][mask])
print('exact match of',int(mask.sum()),'fresh forced-alpha refits')

import importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd
root=Path(sys.argv[1])
here=root/'analysis/s6'
spec=importlib.util.spec_from_file_location('all_age',here/'all-age-precision.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
raw=pd.read_excel(root/'data/Czech/KCOR_CMR.xlsx',sheet_name='2021_24',usecols=['YearOfBirth','Dose','ISOweekDied','Alive','Dead'])
raw['week']=raw.ISOweekDied.astype(str).str.replace('-','',regex=False).str[:6].astype(int)
raw=raw[raw.Dose.isin([0,2])&raw.week.between(202124,202416)]
weeks=np.sort(raw.week.unique());m.initialize(root,weeks)
K=m.precision.K;cfg=m.precision.CFG
windows=[(202319,202340),(202222,202225),(202126,202136),(202411,202414)]
saved=json.loads((root/'analysis/gap_assessment/s6/s6-all-window.json').read_text(encoding='utf8'))
rows=[];summary=[]
for band in m.BANDS:
    vals=[]
    for omit in [None,*windows]:
        pair=[]
        for dose in (0,2):
            x=raw[(raw.Dose==dose)&((raw.YearOfBirth==-2) if band==-2 else ((raw.YearOfBirth>=band)&(raw.YearOfBirth<band+10)))].groupby('week')[['Alive','Dead']].sum().reindex(weeks)
            risk,dead=x.to_numpy().T; t=np.arange(len(risk),dtype=float);skip=K.DYNAMIC_HVE_SKIP_WEEKS
            eff=np.where(t>=skip,K.hazard_from_mr(dead/risk),0.)
            quiet,_,_=K._build_theta_quiet_mask(weeks,t)
            if omit: quiet=quiet & ~((weeks>=omit[0])&(weeks<=omit[1]))
            (k,th),diag=K.fit_theta0_gompertz(eff,t-skip,quiet,cfg['k_anchor_weeks'],cfg['gamma_per_week'],deaths_arr=dead,risk_arr=risk)
            theta=th if diag.get('success') and np.isfinite(th) and th>=0 else 0.
            theta,_,_=K._apply_theta_degenerate_guard(theta,K.get_theta_degenerate_fit_config())
            inv=K.invert_gamma_frailty(np.cumsum(eff),theta)
            pair.append((float(diag.get('theta0_raw',th)),float(inv[-1]/inv[K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE]),diag['status'],bool(diag.get('sampling_flag',False))))
        a,b=pair;v=dict(birth_band=band,omitted=str(omit),theta0=a[0],theta2=b[0],kcor=b[1]/a[1],status0=a[2],status2=b[2],residual0=a[3],residual2=b[3])
        rows.append(v);vals.append(v)
    ref=next(x for x in saved['ratios'] if x['birth_band']==band and x['mode']=='all')
    assert abs(vals[0]['kcor']-ref['point'])<1e-7,(band,vals[0],ref)
    summary.append(dict(birth_band=band,full=ref['point'],ci_low=ref['interval'][0],ci_high=ref['interval'][2],omit_min=min(x['kcor'] for x in vals[1:]),omit_max=max(x['kcor'] for x in vals[1:]),theta2_full=vals[0]['theta2'],theta2_min=min(x['theta2'] for x in vals[1:]),theta2_max=max(x['theta2'] for x in vals[1:])))
pd.DataFrame(rows).to_csv(here/'window-omission-detail.csv',index=False)
df=pd.DataFrame(summary);df.to_csv(here/'window-omission.csv',index=False);print(df.to_string(index=False))

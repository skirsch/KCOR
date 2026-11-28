import os,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from datetime import date,timedelta
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'code'))
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="identified_only")
import KCOR as K
from mfg_codes import MFG_DICT,PFIZER,MODERNA
allowed={k for k,v in MFG_DICT.items() if v in [PFIZER,MODERNA]}
start=date.fromisocalendar(2021,24,1);dates=[(start+timedelta(weeks=j)).strftime('%G-%V') for j in range(149)]
deathidx={v:i for i,v in enumerate(dates)};iso=np.array([int(x.replace('-','')) for x in dates])
bands=[-2,1975,1965,1955,1945,1935,1925]
counts=np.zeros((2,len(bands),2,150),dtype=np.int64)
Ds=['Prvni','Druha','Treti','Ctvrta']; dcols=[f'Datum_{d}_davka' for d in Ds];ccols=[f'OckovaciLatkaKod_{d}_davka' for d in Ds]
raw=root/'data/Czech/Otevrena-data-NR-26-30-COVID-19-prehled-populace-2024-01.csv.gz'
for f in pd.read_csv(raw,usecols=['Infekce','RokNarozeni','DatumUmrtiLPZ']+dcols+ccols,dtype=str,keep_default_na=False,chunksize=400000):
    by=pd.to_numeric(f.RokNarozeni.str[:4],errors='coerce').fillna(-1).to_numpy(dtype=int)
    bgroup=(by-5)//10*10+5
    death=f.DatumUmrtiLPZ.str.replace(r'[^0-9-]','',regex=True)
    eligible=(pd.to_numeric(f.Infekce,errors='coerce').fillna(0).to_numpy()<=1)&(((by>=1920)&(by<=2005))|(by==-1))&((death=='')|(death>=dates[0]))&~((death!='')&(f[dcols[0]]>death))
    wk=death.map(deathidx).fillna(149).to_numpy(dtype=int)
    bad_now=np.zeros(len(f),bool);bad_ever=bad_now.copy();dose=np.zeros(len(f),int)
    for j,(dc,cc) in enumerate(zip(dcols,ccols),1):
        prior=(f[dc].ne('')&f[dc].lt(dates[0])).to_numpy()
        bad=(f[cc].ne('')&~f[cc].isin(allowed)).to_numpy()
        bad_ever|=bad;bad_now|=bad&prior;dose[prior]=j
    for m,reject in enumerate((bad_ever,bad_now)):
        for j,d in enumerate([0,2]):
            mask=eligible&~reject&(dose==d)
            for a,b in enumerate(bands):
                ix=mask if b==-2 else (mask&(bgroup==b))
                counts[m,a,j]+=np.bincount(wk[ix],minlength=150)

t=np.arange(149);skip=K.DYNAMIC_HVE_SKIP_WEEKS;q,_,_=K._build_theta_quiet_mask(iso,t);cfg=K.get_gompertz_theta_fit_config()
saved=json.loads((root/'analysis/gap_assessment/s6/s6-all-window.json').read_text())
stats=[]
for m in range(2):
 for i,b in enumerate(bands):
  for annual in [.065,.085,.105]:
   r=[]
   for j,d in enumerate([0,2]):
    cells=counts[m,i,j];risks=cells[::-1].cumsum()[::-1][:-1];deaths=cells[:-1]
    if m==0:
      arch=next(v for v in saved['points'] if v['dose']==d and v['birth_band']==b)
      assert (risks[0],deaths.sum())==(arch['enrollment_people'],arch['total_deaths']),(b,d,risks[0],deaths.sum(),arch['enrollment_people'],arch['total_deaths'])
    eff=np.where(t>=skip,K.hazard_from_mr(deaths/risks),0)
    (k,th),di=K.fit_theta0_gompertz(eff,t-skip,q,cfg['k_anchor_weeks'],annual/52,deaths_arr=deaths,risk_arr=risks)
    inv=K.invert_gamma_frailty(eff.cumsum(),th)
    r.append((risks[0],deaths.sum(),th,inv[-1]/inv[K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE],di['sampling_flag']))
   ratio=r[1][3]/r[0][3]
   if m==0 and annual==.085:
    target=next(v['point'] for v in saved['ratios'] if v['birth_band']==b)
    assert abs(ratio-target)<1e-8,(ratio,target)
   stats.append([m,b,annual,ratio,*r[0][:3],*r[1][:3],r[0][-1],r[1][-1]])
out=pd.DataFrame(stats,columns=['enrollment_only','band','annual','kcor','n0','deaths0','theta0','n2','deaths2','theta2','flag0','flag2'])
out.to_csv(Path(__file__).with_name('eligibility-refits.csv'),index=False)
print(out[out.annual.eq(.085)].to_string(index=False))


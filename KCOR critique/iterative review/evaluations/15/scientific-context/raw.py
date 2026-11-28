import sys,os,json,time
from pathlib import Path
import pandas as pd,numpy as np
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'code'))
os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only')
import KCOR as K
from mfg_codes import MFG_DICT,PFIZER,MODERNA
allowed={k for k,v in MFG_DICT.items() if v in (PFIZER,MODERNA)}
path=next((root/'data/Czech').glob('*prehled*.gz'))
dose_names=['Prvni','Druha','Treti','Ctvrta','Pata','Sesta']
dates=[f'Datum_{i}_davka' for i in dose_names]
products=[f'OckovaciLatkaKod_{i}_davka' for i in dose_names[:4]]
cols=['Infekce','RokNarozeni','DatumUmrtiLPZ']+dates+products
available=pd.read_csv(path,nrows=0).columns
assert set(cols)<=set(available),set(cols)-set(available)
weeks=pd.date_range('2021-06-14',periods=149,freq='7D');sdates=weeks.strftime('%G-%V');weekmap=dict(zip(sdates,range(len(weeks))))
def norm(s):return s.str.replace(r'[^0-9-]','',regex=True)
bands=np.array([1975,1965,1955,1945,1935,1925,-2])
deads=np.zeros((2,len(bands),2,149),int); ns=np.zeros((2,len(bands),2),int)
for i,f in enumerate(pd.read_csv(path,usecols=cols,dtype=str,chunksize=400000)):
    eligible=pd.to_numeric(f.Infekce,errors='coerce').fillna(0).astype(int)<=1
    yob=pd.to_numeric(f.RokNarozeni.str.extract(r'(\d{4})')[0],errors='coerce')
    eligible &= yob.between(1920,2005)|yob.isna()
    death=norm(f.DatumUmrtiLPZ)
    eligible &= death.isna() | death.ge(sdates[0])
    eligible &= ~(death.notna() & f[dates[0]].gt(death))
    dop=np.zeros(len(f),int); pastbad=np.zeros(len(f),bool); anybad=np.zeros(len(f),bool)
    for j,col in enumerate(dates):
        past=f[col].notna() & f[col].lt(sdates[0]); dop[past]=j+1
        if j<len(products):
            b=(f[products[j]].notna() & ~f[products[j]].isin(allowed))
            pastbad |= (past & b).to_numpy(dtype=bool,na_value=False)
            anybad |= b.to_numpy(dtype=bool,na_value=False)
    observed=death.map(weekmap).fillna(-1).to_numpy(int); birth=(yob.fillna(-1).to_numpy(int)-5)//10*10+5
    for p,exclude in enumerate([anybad,pastbad]):
        ok=eligible.to_numpy(dtype=bool,na_value=False)&~exclude
        for a,band in enumerate(bands):
            eligible_band=ok if band==-2 else ok&(birth==band)
            for j,d in enumerate((0,2)):
                sel=eligible_band&(dop==d); ns[p,a,j]+=sel.sum()
                deads[p,a,j]+=np.bincount(observed[sel & (observed>=0)],minlength=149)
    if i%8==0:print('chunk',i,flush=True)
t=np.arange(149);skip=K.DYNAMIC_HVE_SKIP_WEEKS
quiet=K._build_theta_quiet_mask(weeks.strftime('%G%V').astype(int).to_numpy(),t)[0]
cfg=K.get_gompertz_theta_fit_config()
rows=[]
for p in range(2):
 for a,band in enumerate(bands):
  for gam in [.065,.085,.105]:
    growth=[];th=[];flag=[]
    for j in range(2):
        d=deads[p,a,j];risk=ns[p,a,j]-np.r_[0,np.cumsum(d)[:-1]]
        eff=np.where(t>=skip,K.hazard_from_mr(d/risk),0.)
        (k,theta),diag=K.fit_theta0_gompertz(eff,t-skip,quiet,cfg['k_anchor_weeks'],gam/52,deaths_arr=d,risk_arr=risk)
        theta=K._apply_theta_degenerate_guard(theta,K.get_theta_degenerate_fit_config())[0]
        h=K.invert_gamma_frailty(np.cumsum(eff),theta)
        growth.append(h[-1]/h[K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE]);th.append(theta);flag.append(diag['status'])
    rows.append(dict(p=p,band=int(band),gamma=gam,kcor=growth[1]/growth[0],th0=th[0],th2=th[1],n0=ns[p,a,0],n2=ns[p,a,1],flags=';'.join(flag)))
out=pd.DataFrame(rows);out.to_csv(Path(__file__).with_name('results.csv'),index=False); print(out.to_string(index=False))

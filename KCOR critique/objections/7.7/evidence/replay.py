"""Replay four supplementary figures with current all-window fitter, pointwise person bootstrap."""
import os,sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
os.environ.update(DATASET='Czech',KCOR_NPH_MODE='identified_only')
sys.path.insert(0,str(ROOT/'code'))
import KCOR as K
BANDS=[1930,1940,1950,-2]; DOSES=[0,2,3]; DRAWS=999
def fit(task):
    risk,dead,weeks=task
    t=np.arange(len(risk),dtype=float);skip=K.DYNAMIC_HVE_SKIP_WEEKS
    eff=np.where(t>=skip,K.hazard_from_mr(dead/risk),0.)
    quiet,_,_=K._build_theta_quiet_mask(weeks,t);cfg=K.get_gompertz_theta_fit_config()
    (k,th),d=K.fit_theta0_gompertz(eff,t-skip,quiet,cfg['k_anchor_weeks'],cfg['gamma_per_week'],deaths_arr=dead,risk_arr=risk)
    theta=th if d.get('success') and np.isfinite(th) and th>=0 else 0.
    theta,_,_=K._apply_theta_degenerate_guard(theta,K.get_theta_degenerate_fit_config())
    inv=K.invert_gamma_frailty(np.cumsum(eff),theta);anchor=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    assert inv[anchor]>0
    return inv[anchor:]/inv[anchor],dict(theta=float(theta),raw=float(d.get('theta0_raw',th)),status=d['status'],residual=bool(d.get('sampling_flag',False)))
def main():
    src=ROOT/'data/Czech/KCOR_CMR.xlsx';raw=pd.read_excel(src,sheet_name='2022_26',usecols=['DateDied','ISOweekDied','YearOfBirth','Dose','Alive','Dead'])
    raw['week']=raw.ISOweekDied.astype(str).str.replace('-','',regex=False).str[:6].astype(int)
    raw=raw[raw.Dose.isin(DOSES)&raw.week.between(202226,202441)]
    g=raw.groupby(['YearOfBirth','Dose','week'])[['Alive','Dead']].sum();weeks=np.sort(raw.week.unique())
    dates=pd.to_datetime([f'{w//100}-W{w%100:02}-1' for w in weeks],format='%G-W%V-%u')
    rng=np.random.default_rng(20261126);obs=np.zeros((4,3,2,len(weeks)),dtype=np.int64);sample=np.zeros((4,3,DRAWS,2,len(weeks)),dtype=np.int64)
    for band,dose in g.index.droplevel('week').unique():
        if band<=0 and band!=-1:continue
        x=g.loc[band,dose].reindex(weeks);risk,dead=x.to_numpy(dtype=np.int64).T
        assert np.all(risk[:-1]-dead[:-1]==risk[1:]),(band,dose)
        cells=np.r_[dead,risk[-1]-dead[-1]];assert cells.sum()==risk[0]
        counts=rng.multinomial(risk[0],cells/risk[0],size=DRAWS) if risk[0] else np.zeros((DRAWS,len(cells)),int)
        risk_draw=np.cumsum(counts[:,::-1],axis=1)[:,::-1][:,:-1]
        for i in [3]+([BANDS.index(band)] if band in BANDS else []):
            j=DOSES.index(dose);obs[i,j]+=[risk,dead];sample[i,j,:,0]+=risk_draw;sample[i,j,:,1]+=counts[:,:-1]
    for j,dose in enumerate(DOSES):assert np.array_equal(obs[3,j],g.loc[-2,dose].reindex(weeks).to_numpy().T)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=[];anchor=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,band in enumerate(BANDS):
            points=[];draws=[];diag=[]
            for j,dose in enumerate(DOSES):
                point,d=fit((*obs[i,j],weeks));points.append(point);diag.append(d)
                print('POINT',band,dose,d,flush=True)
                result=list(pool.map(fit,((s[0],s[1],weeks) for s in sample[i,j]),chunksize=12))
                draws.append(np.array([x[0] for x in result]));print('DONE',band,dose,flush=True)
            fig,ax=plt.subplots(figsize=(8.2,4.2));ax.axhline(1,color='gray',ls='--',lw=.8)
            lines=[]
            for j,color in [(1,'#1f77b4'),(2,'#ff7f0e')]:
                p=points[j]/points[0];b=draws[j]/draws[0];lo,hi=np.percentile(b,[2.5,97.5],axis=0)
                ax.plot(dates[anchor:],p,color=color,label=f'Dose {DOSES[j]} / Dose 0');ax.fill_between(dates[anchor:],lo,hi,color=color,alpha=.18)
                for date,v,l,h in zip(dates[anchor:],p,lo,hi):lines.append(dict(date=str(date.date()),dose=DOSES[j],kcor=v,low=l,high=h))
                end23=np.flatnonzero(dates[anchor:].year==2023)[-1]
                summary.append(dict(band=band,dose=DOSES[j],end2023=[float(p[end23]),float(lo[end23]),float(hi[end23])],end=[float(p[-1]),float(lo[-1]),float(hi[-1])],diag0=diag[0],diag_num=diag[j]))
            ax.set(xlabel='Calendar date',ylabel='Anchored KCOR');ax.legend(loc='best');fig.autofmt_xdate();fig.tight_layout()
            dest=ROOT/'documentation/preprint/figures';name=f'kcor_realdata_yob{band}_enroll2022w26_eval2023.png' if band!=-2 else 'kcor_realdata_allages_enroll2022w26_eval2023.png'
            if band!=-2:dest/= 'supplement'
            fig.savefig(dest/name,dpi=180);plt.close(fig)
            pd.DataFrame(lines).to_csv(HERE/f'trajectory-{band}.csv',index=False)
            np.savez_compressed(HERE/f'draws-{band}.npz',draws=draws,points=points,dates=np.asarray(dates[anchor:].astype(str),dtype="U10"))
            (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()

"""Pointwise matched person bootstrap for the 2021 age-shift figures."""
import os,sys,json,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout,redirect_stderr
import io,tempfile
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[key]="1"
os.environ.update(DATASET="Czech",KCOR_NPH_MODE="identified_only",KCOR_BOOTSTRAP_DRAWS="0",MONTE_CARLO="0")
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"code"))
import KCOR as K
import bootstrap_reporting as B
SHEET="2021_24"
SEED=20261121
DRAWS=999
def init(dates,cells):
    global DATES,CELLS,T,QUIET,CFG,GUARD
    DATES,CELLS=dates,cells
    T=np.arange(len(dates),dtype=float)
    QUIET,_,_=K._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(),T)
    CFG=K.get_gompertz_theta_fit_config()
    GUARD=K.get_theta_degenerate_fit_config()
    assert not K.get_covid_correction_config()["apply_correction"]
def estimate(draw):
    x=CELLS.copy()
    if draw:
        rng=np.random.default_rng([SEED,0,draw])
        x=rng.multinomial(int(x.sum()),x.ravel()/x.sum()).reshape(x.shape)
    x[2]+=x[0]+x[1]
    risk=np.cumsum(x[2,:,::-1],axis=1)[:,::-1][:,:-1]
    dead=x[2,:,:-1]
    norm=[]; diagrows=[]
    skip=K.DYNAMIC_HVE_SKIP_WEEKS
    anchor=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    for a,d in zip(risk,dead):
        h=K.hazard_from_mr(d/(a+K.EPS))
        eff=np.where(T>=skip,h,0.)
        (k,theta),diag=K.fit_theta0_gompertz(eff,T-skip,QUIET,CFG["k_anchor_weeks"],CFG["gamma_per_week"],deaths_arr=d,risk_arr=a,iso_week=DATES.strftime("%G%V").astype(int).to_numpy())
        raw=float(theta)
        theta=theta if diag.get("success") and np.isfinite(theta) and theta>=0 else 0.
        theta,_,_=K._apply_theta_degenerate_guard(theta,GUARD)
        H=K.invert_gamma_frailty(np.cumsum(eff),theta)
        norm.append(H/H[anchor])
        diagrows.append(dict(k=float(k),theta=float(theta),theta_raw=raw,status=diag.get("status"),success=bool(diag.get("success")),insufficient_signal=bool(diag.get("insufficient_signal")),sampling_flag=bool(diag.get("sampling_flag"))))
    norm=np.array(norm)
    with np.errstate(invalid="ignore",divide="ignore"):
        ratios=norm[1:]/norm[0]
    return ratios,diagrows
def verify(draw, source, dates, cells):
    rng=np.random.default_rng([SEED,0,draw])
    frame=B.draw_history(("table8",dates,cells),rng) if draw else pd.read_excel(source,sheet_name=SHEET)
    K.ENROLLMENT_DATES=[SHEET]
    K.OVERRIDE_DOSE_PAIRS=[(1,0),(2,0)]
    K.OVERRIDE_YOBS=[1960]
    with tempfile.TemporaryDirectory() as t, redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
        rows=K.process_workbook(str(source),str(Path(t)/"unused.xlsx"),bootstrap_frames={SHEET:frame},return_rows=True)
    actual=estimate(draw)[0]
    for j in (1,2):
        v=rows[(rows.YearOfBirth==1960)&(rows.Dose_num==j)&(rows.Dose_den==0)].set_index("Date")["KCOR"].reindex(dates).to_numpy(float)
        ix=np.arange(len(dates))>=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
        assert np.isfinite(v[ix]).all(), (draw,j,len(v),v[ix])
        np.testing.assert_allclose(actual[j-1,ix],v[ix],rtol=1e-8,atol=1e-10)
    print("Ordinary workbook check passed",draw,flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--draws",type=int,default=DRAWS);parser.add_argument("--workers",type=int,default=5);args=parser.parse_args()
    source=ROOT/"analysis/table8/input.xlsx"
    _,dates,cells=B.histories(pd.read_excel(source,sheet_name=SHEET),SHEET,"table8")
    init(dates,cells)
    print("Dates",dates[0],dates[-1],"anchor",K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE,dates[K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE],flush=True)
    for draw in (0,1,42):verify(draw,source,dates,cells)
    point,pdiag=estimate(0)
    print("Point diagnostics",pdiag,flush=True)
    vals=[];diags=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(dates,cells)) as pool:
        for i,(v,d) in enumerate(pool.map(estimate,range(1,args.draws+1),chunksize=8),1):
            vals.append(v);diags.append(d)
            if i%100==0:print("Draws",i,flush=True)
    vals=np.asarray(vals)
    anchor=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
    assert np.isfinite(vals[:,:,anchor:]).all()
    lo,hi=np.quantile(vals,[.025,.975],axis=0)
    rows=[]
    for j in (1,2):
        for t,date in enumerate(dates):
            if t>=anchor:
                rows.append(dict(pair=f"{j}/0",date=date.date(),week_since_enrollment=t,point=point[j-1,t],lower=lo[j-1,t],upper=hi[j-1,t]))
    pd.DataFrame(rows).to_csv(HERE/"curves.csv",index=False)
    np.savez_compressed(HERE/"draws.npz",dates=dates.to_numpy(),point=point,draws=vals)
    summary=dict(enrollment=SHEET,seed=SEED,draws=args.draws,anchor=int(anchor),anchor_date=str(dates[anchor].date()),end_date=str(dates[-1].date()),people=cells.sum(axis=(0,2)).tolist(),point_diagnostics=pdiag,draw_diagnostics=diags)
    (HERE/"diagnostics.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(pd.DataFrame(rows).groupby("pair").tail(1).to_string(index=False),flush=True)
if __name__=="__main__":main()

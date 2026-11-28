"""9.8: preserve available-strata points; refuse mixed-composition intervals."""
from pathlib import Path
import os,sys,io,tempfile
from contextlib import redirect_stdout,redirect_stderr
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'test'));sys.path.insert(0,str(ROOT/'code'))
from test_source_component_history import fixture
from test_round7_workbook import run,K

def sparse(n_anchor):
    f=fixture();mask=f.YearOfBirth.eq(1950)&f.Dose.eq(1)
    for sex in ['F','M']:
        idx=f[mask&f.Sex.eq(sex)].sort_values('DateDied').index
        d=f.loc[idx,'Dead'].to_numpy().copy();d[:6]=0
        if sex=='M':d[3]=n_anchor
        f.loc[idx,'Dead']=d;f.loc[idx,'Alive']=30000-np.r_[0,np.cumsum(d[:-1])]
    return f

def pair(out,age):return out[out.YearOfBirth.eq(age)&out.Dose_num.eq(1)&out.Dose_den.eq(0)]

def written(f,draws):
    os.environ.update(KCOR_BOOTSTRAP_DRAWS=str(draws),KCOR_BOOTSTRAP_WORKERS='4',KCOR_NPH_MODE='identified_only',KCOR_BOOTSTRAP_LAYOUT='disjoint')
    K.ENROLLMENT_DATES=['2021_24'];K.OVERRIDE_DOSE_PAIRS=[(1,0)];K.OVERRIDE_YOBS=None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        src=Path(t)/'input.xlsx';dst=Path(t)/'out.xlsx';f.to_excel(src,sheet_name='2021_24',index=False)
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):K.process_workbook(str(src),str(dst))
        return pd.read_excel(dst,sheet_name='dose_pairs')

def main():
    f=sparse(0);out=run('2021_24',f);p=pair(out,0);a=pair(out,1940);end=p.iloc[-1]
    assert end.pool_contributors=='1940' and end.pool_expected_ages=='1940,1950' and end.pool_composition_status=='partial'
    assert 0<end.pool_retained_weight_fraction<1 and abs(end.KCOR-a.KCOR.iloc[-1])<1e-10
    assert pair(out,1950).KCOR.isna().all()
    print('partial point',end.KCOR,end.pool_contributors,end.pool_retained_weight_fraction,flush=True)
    # One anchor-period death is lost in some person-bootstrap draws.
    f=sparse(1);out=written(f,19);p=pair(out,0).iloc[-1]
    print('sparse pool',p[['KCOR','pool_contributors','pool_expected_ages','pool_retained_weight_fraction','pool_composition_status','CI_pool_composition_changed_draws','CI_valid_draws','CI_lower','CI_upper']].to_dict(),flush=True)
    assert p.pool_contributors=='1940,1950' and p.pool_composition_status=='complete'
    assert 0<p.CI_pool_composition_changed_draws<19 and p.CI_valid_draws==19-p.CI_pool_composition_changed_draws
    assert pd.isna(p.CI_lower) and pd.isna(p.CI_upper)
    a=pair(out,1940).iloc[-1];assert a.CI_pool_composition_changed_draws==0 and pd.notna(a.CI_lower)
    # A well-supported pool retains the usual refit interval.
    out=written(fixture(),19);p=pair(out,0).iloc[-1]
    print('complete pool',p[['KCOR','pool_contributors','CI_pool_composition_changed_draws','CI_valid_draws','CI_lower','CI_upper']].to_dict(),flush=True)
    assert p.CI_pool_composition_changed_draws==0 and p.CI_valid_draws==19 and pd.notna(p.CI_lower) and pd.notna(p.CI_upper)
if __name__=='__main__':main()

"""18.1-18.4: validated MC source populations, raw comparator, and coverage."""
from test_round16_mc_pooling import K, S, input_frame, pd, np
from test_round17_mc_totals import supplied_totals
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import io,os,tempfile


def run(f, scope="exhaustive", write=False, low=1940, high=1950, include_aggregates=True):
    os.environ.update(MC_ENROLLMENT_DATE=S,MONTE_CARLO="1",KCOR_BOOTSTRAP_DRAWS="0",KCOR_BOOTSTRAP_LAYOUT="disjoint",KCOR_AGGREGATE_SCOPE=scope,KCOR_NPH_MODE="identified_only")
    K.MONTE_CARLO_MODE=True; K.MC_INCLUDE_AGGREGATES=include_aggregates
    K.MC_YOB_START=low; K.MC_YOB_END=high; K.ENROLLMENT_DATES=[S]
    K.OVERRIDE_DOSE_PAIRS=[(1,0)]; K.OVERRIDE_YOBS=None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        src=Path(t)/"in.xlsx"; dst=Path(t)/"out.xlsx"
        f.to_excel(src,sheet_name=S,index=False)
        with redirect_stdout(io.StringIO()) as log,redirect_stderr(io.StringIO()):
            out=K.process_workbook(str(src),str(dst),return_rows=not write)
        if write:
            out=pd.read_excel(dst,sheet_name=S,dtype={
                c: str for c in ("population_age_groups_num","population_age_groups_den",
                                 "population_zero_enrollment_groups_num","population_zero_enrollment_groups_den")})
            for name in ("KCOR","KCOR_unanchored","KCOR_ns"):
                out[name]=pd.to_numeric(out[name],errors="coerce")
        weights={int(i):K._asmr_weights_for(S,int(i)).copy() for i in f.mc_id.unique()}
        return out,weights,log.getvalue()


def select(out,i,age=-2):
    return out[out.mc_id.eq(i)&out.YearOfBirth.eq(age)].sort_values("Date").reset_index(drop=True)


def extended(null=False):
    f=input_frame(S)
    a=f[f.Dose.eq(0)&f.YearOfBirth.eq(1940)].copy(); a["YearOfBirth"]=1970
    dead=np.r_[np.repeat(40,8),np.repeat(180,12)]
    a["Dead"]=dead; a["Alive"]=40000-np.r_[0,np.cumsum(dead[:-1])]
    b=a.copy(); b["Dose"]=1
    if not null:
        d=dead.copy(); d[8:]*=2
        b["Dead"]=d; b["Alive"]=40000-np.r_[0,np.cumsum(d[:-1])]
    if null:
        f=f[f.Dose.eq(0)]; f=pd.concat([f,f.assign(Dose=1)],ignore_index=True)
    return pd.concat([f,a,b],ignore_index=True)


def arithmetic(f):
    # Independent raw all-ages calculation from disjoint counts, or supplied
    # totals if they are the only population rows in this fixture.
    cols=["Dose","DateDied"]
    total=f.groupby(cols)[["Alive","Dead"]].sum().reset_index().sort_values(cols)
    h={}
    for d,part in total.groupby("Dose"):
        weekly=-np.log1p(-part.Dead.to_numpy(float)/part.Alive.to_numpy(float))
        weekly[:int(K.DYNAMIC_HVE_SKIP_WEEKS)]=0.
        h[d]=np.cumsum(weekly)
    ratio=np.divide(h[1],h[0],out=np.full_like(h[0],np.nan),where=h[0]>0)
    return ratio/ratio[int(K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE)]


def matrix():
    # All four redundant-total layouts, both exactly identical doses and
    # nontrivial full-population trajectories. A 1970 component is not shown.
    frames=[]; sources={}
    for i,(null,doses) in enumerate([(n,d) for n in (True,False) for d in ([],[0],[1],[0,1])],1):
        f=extended(null).assign(mc_id=i); sources[i]=f
        frames.append(pd.concat([f,supplied_totals(f,doses)],ignore_index=True) if doses else f)
    batch=pd.concat(frames,ignore_index=True)
    out,weights,_=run(batch,write=True)
    for i in range(1,9):
        direct=select(out,i); chosen=select(out,1 if i<=4 else 5)
        assert len(direct)==20 and direct.normalization_status.eq("ok").all()
        assert direct.population_label.eq("All Ages").all()
        np.testing.assert_allclose(direct.KCOR,chosen.KCOR,rtol=1e-13,equal_nan=True)
        np.testing.assert_allclose(direct.KCOR_unanchored,chosen.KCOR_unanchored,rtol=1e-13,equal_nan=True)
        np.testing.assert_allclose(direct.KCOR_ns,arithmetic(sources[i]),rtol=1e-13,equal_nan=True)
        assert not len(select(out,i,1970))
        if i<=4: np.testing.assert_allclose(direct.KCOR.dropna(),1.,rtol=1e-13)
        # ASMR still represents configured individual-age target.
        assert set(weights[i])=={1940,1950}
    # The default MC reader range still omits 1970. Every layout must
    # continue to select the full raw population at that range.
    default,_,_=run(batch,low=1930,high=1960)
    for i in range(1,9):
        np.testing.assert_allclose(select(out,i).KCOR,select(default,i).KCOR,rtol=1e-13,equal_nan=True)
        np.testing.assert_allclose(select(out,i).KCOR_ns,select(default,i).KCOR_ns,rtol=1e-13,equal_nan=True)
    # A broader display range retains the same direct full-population result,
    # even though the individual-age and ASMR targets are allowed to change.
    broad,_,_=run(sources[5],high=1970)
    np.testing.assert_allclose(select(out,5).KCOR,select(broad,5).KCOR,rtol=1e-13,equal_nan=True)
    print("PASS all four layouts, null/non-null, hidden raw age, full-population direct/refit and independent raw comparator; written workbook")


def raw_only_and_malformed():
    a=extended().assign(mc_id=1); b=input_frame(S).assign(mc_id=2)
    direct_a=supplied_totals(a); direct_b=supplied_totals(b)
    totals=pd.concat([direct_a,direct_b],ignore_index=True)
    out,_,_=run(totals,write=True)
    for i,f in [(1,a),(2,b)]:
        np.testing.assert_allclose(select(out,i).KCOR_ns,arithmetic(f),rtol=1e-13,equal_nan=True)
    assert abs(select(out,1).KCOR_ns.iloc[-1]-select(out,2).KCOR_ns.iloc[-1])>.1
    print("PASS totals-only written raw comparator, distinct replicate trajectories")
    bad=direct_a.copy(); idx=bad[bad.Dose.eq(1)].sort_values("DateDied").index
    bad.loc[idx[8:],"Dead"]*=3
    bad.loc[idx,"Alive"]=bad.loc[idx[0],"Alive"]-np.r_[0,np.cumsum(bad.loc[idx,"Dead"].to_numpy()[:-1])]
    invalid_total=pd.concat([a,bad],ignore_index=True)
    hidden=a.assign(mc_id=3).copy(); hidden.loc[hidden.YearOfBirth.eq(1970)&hidden.Dose.eq(1)&hidden.DateDied.eq(hidden.DateDied.unique()[10]),"Alive"]+=50
    displayed=a.assign(mc_id=4).copy()
    displayed.loc[displayed.YearOfBirth.eq(1940)&displayed.Dose.eq(1)&displayed.DateDied.eq(displayed.DateDied.unique()[10]),"Alive"]+=50
    malformed=pd.concat([invalid_total,b,hidden,displayed],ignore_index=True)
    result,_,_=run(malformed,write=True)
    for i in (1,3):
        affected=select(result,i)
        assert len(affected)==20 and affected.KCOR.isna().all() and affected.KCOR_ns.isna().all()
        assert not affected.normalization_status.eq("ok").any()
        assert select(result,i,1940).KCOR.notna().any()
        assert select(result,i,0).KCOR.notna().any()
    assert select(result,1).normalization_status.eq("invalid_aggregate_counts").all()
    assert select(result,2).KCOR.notna().any()
    for age in (-2,0,1940):
        assert select(result,4,age).KCOR.isna().all()
        assert not select(result,4,age).normalization_status.eq("ok").any()
    assert select(result,4,1950).KCOR.notna().any()
    np.testing.assert_allclose(select(result,2).KCOR_ns,arithmetic(b),rtol=1e-13,equal_nan=True)
    print("PASS contradictory supplied total, hidden invalid component, replicate/age/ASMR isolation")


def partial():
    a=extended().assign(mc_id=1)
    a=a[~(a.Dose.eq(1)&a.YearOfBirth.eq(1950))].copy()
    zero=a[a.YearOfBirth.eq(1940)&a.Dose.eq(1)].assign(YearOfBirth=1955,Alive=0,Dead=0)
    a=pd.concat([a,zero],ignore_index=True)
    b=input_frame(S).assign(mc_id=2); b_full=pd.concat([b,supplied_totals(b)],ignore_index=True)
    c=input_frame(S).assign(mc_id=3); c_one=pd.concat([c,supplied_totals(c,[0])],ignore_index=True)
    out,_,log=run(pd.concat([a,b_full,c_one],ignore_index=True),scope="partial",write=True)
    partial_rows=select(out,1); full=select(out,2)
    assert len(partial_rows)==20 and len(full)==20 and len(select(out,3))==0
    assert partial_rows.population_label.eq("All available ages").all()
    assert partial_rows.population_age_groups_num.astype(str).eq("1940,1955,1970").all()
    assert partial_rows.population_age_groups_den.astype(str).eq("1940,1950,1970").all()
    assert partial_rows.population_zero_enrollment_groups_num.astype(str).eq("1955").all()
    assert partial_rows.population_zero_enrollment_groups_den.eq("none").all()
    assert partial_rows.population_coverage_status.eq("different_supplied_age_groups").all()
    assert full.population_label.eq("All Ages").all() and full.population_coverage_status.isna().all()
    np.testing.assert_allclose(partial_rows.KCOR_ns,arithmetic(a),rtol=1e-13,equal_nan=True)
    np.testing.assert_allclose(full.KCOR_ns,arithmetic(b),rtol=1e-13,equal_nan=True)
    assert "All available ages (dose 1: 1940,1955,1970; dose 0: 1940,1950,1970)" in log
    assert "Dose combination: 1 vs 0 - All Ages" in log
    print("PASS mixed partial/full/one-dose totals, per-dose raw coverage incl hidden and zero groups, raw comparator, distinct summary labels")


def calendars():
    a=input_frame(S).assign(mc_id=1)
    b=input_frame(S).assign(mc_id=2)
    b=b[b.DateDied < sorted(b.DateDied.unique())[17]].copy()
    batch,_,_=run(pd.concat([a,b],ignore_index=True))
    alone,_,_=run(b)
    for age in (-2,0,1940,1950):
        pd.testing.assert_frame_equal(select(batch,2,age),select(alone,2,age),check_exact=True)
        assert len(select(alone,2,age))==17
    no_agg,_,_=run(pd.concat([a,b],ignore_index=True),include_aggregates=False)
    assert not no_agg.YearOfBirth.eq(-2).any()
    print("PASS unequal replicate horizons and opt-out of aggregate reporting")


if __name__=="__main__":
    matrix(); raw_only_and_malformed(); partial(); calendars()

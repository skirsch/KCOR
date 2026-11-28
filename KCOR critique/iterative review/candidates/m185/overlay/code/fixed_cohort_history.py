"""Validate original fixed-cohort components before any grouping."""
import os
import numpy as np
import pandas as pd

def risk_history_policy(layout="disjoint"):
    """Default to death-only; nondeath exits require an explicit opt-in."""
    if layout not in ("disjoint", "table8"):
        raise ValueError(f"Unknown KCOR_BOOTSTRAP_LAYOUT: {layout}")
    policy = os.environ.get("KCOR_RISK_HISTORY_POLICY", "death_only")
    if policy not in ("death_only", "allow_exits"):
        raise ValueError(f"Unknown KCOR_RISK_HISTORY_POLICY: {policy}")
    if layout == "table8" and policy != "death_only":
        raise ValueError("Overlapping Table 8 layout requires death-only accounting")
    return policy


def aggregate_scope(layout="disjoint"):
    """Declare whether supplied inclusive totals partition the displayed strata."""
    scope = os.environ.get("KCOR_AGGREGATE_SCOPE", "exhaustive")
    if scope not in ("exhaustive", "partial"):
        raise ValueError(f"Unknown KCOR_AGGREGATE_SCOPE: {scope}")
    if layout == "table8" and scope != "exhaustive":
        raise ValueError("Table 8 requires the declared overlapping layout")
    return scope


def aggregate_reconciliation(frame, scope="exhaustive"):
    """Return invalid total histories and totals unsupported by joint resampling.

    A sole total is a population in its own right. With displayed components,
    the exhaustive converter layout requires equality, allowing fractional
    floating-point roundoff only. An explicitly
    partial extract may contain a larger, internally valid supplied total;
    its missing member strata cannot be jointly resampled.
    """
    aggregate = frame[frame.YearOfBirth.isin([-2,-60])]
    base = frame[~frame.YearOfBirth.isin([-2,-60])]
    invalid = {}
    unsupported = set()
    if aggregate.empty:
        return invalid, unsupported
    cols = ["Alive","Dead"] + (["Dead_COVID"] if "Dead_COVID" in frame else [])
    def count_tolerance(a, *inputs):
        # Integer counts retain exact reconciliation. Fractional expected
        # counts permit only a few ulps of floating-point arithmetic error.
        if all(np.array_equal(x,np.floor(x)) for x in inputs):
            return np.zeros_like(a)
        scale=np.maximum.reduce([np.ones_like(a),np.abs(a),*[np.abs(x) for x in inputs]])
        return 16*np.finfo(float).eps*scale

    def feasible(counts, *operands):
        """Check a difference of histories, allowing only fractional roundoff."""
        a=np.asarray(counts,dtype=float)
        inputs=[np.asarray(x,dtype=float) for x in operands]
        if not np.isfinite(a).all() or any(not np.isfinite(x).all() for x in inputs):
            return False
        # Integer input counts remain exact; fractional differences can lose
        # a few ulps when the original population totals are subtracted.
        tol=count_tolerance(a,*inputs)
        if np.any(a < -tol):
            return False
        alive,dead=a[:,0],a[:,1]
        if np.any(dead-alive>tol[:,0]+tol[:,1]):
            return False
        if "Dead_COVID" in cols and np.any(a[:,2]-dead>tol[:,2]+tol[:,1]):
            return False
        loss=alive[:-1]-dead[:-1]-alive[1:]
        loss_tol=tol[:-1,0]+tol[:-1,1]+tol[1:,0]
        return not np.any(loss < -loss_tol) and (risk_history_policy()=="allow_exits" or not np.any(np.abs(loss)>loss_tol))

    for (age,dose), template in aggregate.groupby(["YearOfBirth","Dose"],sort=False) if not base.empty else []:
        key=(age,dose)
        source=base[base.Dose.eq(dose)]
        if age==-60:
            source=source[source.YearOfBirth.ge(1960)]
        reason=None
        if template.DateDied.duplicated().any():
            reason="duplicate supplied aggregate week"
        elif ("Sex" in template and not template.Sex.eq("O").all() and (source.Sex.nunique()!=1 or not template.Sex.eq(source.Sex.iloc[0]).all())):
            reason="unexpected supplied aggregate sex layout"
        elif "DCCI" in template and not template.DCCI.eq(-1).all() and (source.DCCI.nunique()!=1 or not template.DCCI.eq(source.DCCI.iloc[0]).all()):
            reason="unexpected supplied aggregate comorbidity layout"
        elif ((base.YearOfBirth<=0)&(base.YearOfBirth!=-1)).any():
            reason="unrecognized overlapping strata with converter totals"
        else:
            observed=template.set_index("DateDied")[cols].sort_index()
            expected=source.groupby("DateDied")[cols].sum().sort_index()
            observed,expected=observed.align(expected,fill_value=0)
            a,b=observed.to_numpy(float),expected.to_numpy(float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                reason="nonfinite supplied or component counts"
            elif np.any(np.abs(a-b)>count_tolerance(a-b,a,b)):
                if scope=="partial":
                    if not feasible(a-b,a,b):
                        reason="supplied partial total has an inconsistent undisplayed population"
                    else:
                        unsupported.add(key)
                else:
                    reason=f"aggregate {age} does not match disjoint source counts"
        if reason:
            invalid[key]=[(key,reason)]
            unsupported.add(key)
    # All ages includes the known-born-1960+ subtotal. Validate both the
    # complementary population and its remainder after displayed older and
    # unknown-birth-year cells (the latter are outside the known-1960+ subtotal).
    # This check also applies when the workbook contains only the nested totals.
    for dose,group in aggregate.groupby("Dose",sort=False):
        all_rows=group[group.YearOfBirth.eq(-2)]
        young_rows=group[group.YearOfBirth.eq(-60)]
        if all_rows.empty or young_rows.empty:
            continue
        keys=[(-2,dose),(-60,dose)]
        reason=None
        if all_rows.DateDied.duplicated().any() or young_rows.DateDied.duplicated().any():
            reason="duplicate week prevents nested aggregate reconciliation"
        else:
            full=all_rows.set_index("DateDied")[cols].sort_index()
            young=young_rows.set_index("DateDied")[cols].sort_index()
            full,young=full.align(young,fill_value=0)
            remainder=full-young
            known_outside=base[base.Dose.eq(dose)&((base.YearOfBirth.eq(-1))|((base.YearOfBirth>0)&(base.YearOfBirth<1960)))]
            outside=known_outside.groupby("DateDied")[cols].sum().sort_index()
            remainder,outside=remainder.align(outside,fill_value=0)
            # Use the original operands for roundoff scales, not only the
            # potentially small cancellation remainder.
            full=full.reindex(remainder.index,fill_value=0)
            young=young.reindex(remainder.index,fill_value=0)
            if not feasible(remainder,full,young) or not feasible(remainder-outside,full,young,outside):
                reason="incompatible nested inclusive totals or complementary risk/death history"
        if reason:
            # In an exhaustive extract an exact independently reconciled total
            # remains identified by the complete disjoint population.
            affected=keys
            if scope=="exhaustive" and not base.empty and sum(k in invalid for k in keys)==1:
                affected=[k for k in keys if k in invalid]
            for key in affected:
                invalid.setdefault(key,[]).append((key,reason))
                unsupported.add(key)
    return invalid,unsupported


def invalid_components(frame, start, horizon=None, layout="disjoint"):
    """Return (age, dose) groups containing an invalid source history.

    A positive residual risk set cannot silently disappear before the sheet
    horizon. A contiguous suffix missing after complete exhaustion is allowed.
    Death-only accounting is the default. The disjoint exit-capable sensitivity
    mode explicitly permits nondeath exits. Unexplained increases
    and missing weeks are not. Fractional expected counts are permitted here
    (the person bootstrap independently requires integer history cells).
    """
    policy=risk_history_policy(layout)
    if frame.empty:
        return {}
    start=pd.Timestamp(start)
    horizon=pd.Timestamp(horizon if horizon is not None else frame.DateDied.max())
    keys=["YearOfBirth","Dose"]+[c for c in ("Sex","DCCI") if c in frame]
    bad={}
    if layout=="table8":
        # Known age-shift input has intentionally relabelled overlapping rows,
        # without original constituent IDs. Combine indistinguishable rows only;
        # preserve identifiable sex/DCCI histories until validation is complete.
        for key,g in frame.groupby(keys,sort=False,dropna=False):
            counts=g.groupby("DateDied").size()
            if counts.nunique()>1:
                bad.setdefault(tuple(key[:2]),[]).append((tuple(key),"inconsistent age-shift source-row multiplicity"))
        frame=frame.groupby(keys+["DateDied"],as_index=False,dropna=False)[["Alive","Dead"]].sum()

    for key,g in frame.groupby(keys,sort=False,dropna=False):
        x=g.sort_values("DateDied")
        dates=pd.DatetimeIndex(x.DateDied)
        a=pd.to_numeric(x.Alive,errors="coerce").to_numpy(float)
        d=pd.to_numeric(x.Dead,errors="coerce").to_numpy(float)
        tol=1e-9*np.maximum(1.,np.abs(a))
        reason=None
        if dates.has_duplicates or dates[0]!=start or not np.all(np.diff(dates.values)==np.timedelta64(7,"D")):
            reason="missing or duplicate component week"
        elif not (np.isfinite(a).all() and np.isfinite(d).all()) or np.any(a<0) or np.any(d<0) or np.any(d-a>tol):
            reason="invalid component risk/death counts"
        elif np.any(a[1:]>a[:-1]-d[:-1]+tol[:-1]):
            reason="unexplained component risk-set increase"
        elif policy=="death_only" and np.any(np.abs(a[1:]-(a[:-1]-d[:-1]))>tol[:-1]):
            reason="component violates death-only accounting (unexplained nondeath loss)"
        elif dates[-1]<horizon and a[-1]-d[-1]>tol[-1]:
            reason="truncated positive-risk component"
        if reason:
            bad.setdefault(tuple(key[:2]),[]).append((tuple(key),reason))
    if layout=="table8":
        # The 1960 label is an inclusive population containing both the
        # 1940 and 1950 pseudo-cohorts, not a third disjoint population.
        # An impossible remainder invalidates the inclusive history (and its
        # direct aggregate and pooled contrasts), not the constituent histories
        # or comparisons between other doses. Validate each dose independently
        # because point reporting can request fewer than all three doses.
        grouped=frame.groupby(["YearOfBirth","Dose","DateDied"])[["Alive","Dead"]].sum()
        for dose in frame.loc[frame.YearOfBirth.eq(1960),"Dose"].unique():
            keys=[(age,dose) for age in (1940,1950,1960)]
            if any(key in bad for key in keys):
                # Component-level failures already have their own diagnostic.
                # A missing/invalid history is not evidence of a negative
                # remainder in an otherwise independently valid total.
                continue
            group=grouped[grouped.index.get_level_values("Dose")==dose]
            try:
                overlap_history_cells(group,start,horizon,doses=(dose,))
            except ValueError as exc:
                key=(1960,dose)
                bad.setdefault(key,[]).append((key,str(exc)))
    return bad


def rectangular_overlap_counts(group, start, horizon, doses=(0,1,2)):
    """Build the declared Table 8 rectangle after source-history validation.

    Only omitted trailing zeros following verified complete exhaustion are
    inserted. An absent group, interior gap or remaining risk is an error.
    """
    dates=pd.date_range(start,horizon,freq="7D")
    idx=pd.MultiIndex.from_product([[1940,1950,1960],doses,dates],names=["YearOfBirth","Dose","DateDied"])
    if group.index.has_duplicates or not group.index.isin(idx).all():
        raise ValueError("unexpected overlapping age-shift layout")
    result=group.reindex(idx)
    for age in (1940,1950,1960):
        for dose in doses:
            cells=result.loc[(age,dose)]
            missing=cells.isna().any(axis=1).to_numpy()
            if not missing.any():continue
            first=int(np.flatnonzero(missing)[0])
            if first==0 or not missing[first:].all() or not cells.iloc[first:].isna().all().all():
                raise ValueError("incomplete overlapping histories")
            last=cells.iloc[first-1]
            if last.Alive!=last.Dead:
                raise ValueError("truncated positive-risk overlapping history")
            result.loc[(age,dose,dates[first:]),["Alive","Dead"]]=0
    if result.isna().any().any():raise ValueError("incomplete overlapping histories")
    return dates,result


def overlap_history_cells(group, start, horizon, doses=(0,1,2)):
    """Reconstruct disjoint death-week and final-survivor cells for Table 8.

    A nonnegative residual death/survivor history is both necessary and
    sufficient for the declared nested layout, given death-only accounting
    in each supplied group. Integer counts must reconcile exactly; fractional
    expected counts used by point reporting allow only floating-point roundoff.
    The person bootstrap separately requires integer raw counts.
    """
    dates,v=rectangular_overlap_counts(group,start,horizon,doses)
    a=v.Alive.to_numpy().reshape(3,len(doses),-1)
    d=v.Dead.to_numpy().reshape(3,len(doses),-1)
    if not (np.isfinite(a).all() and np.isfinite(d).all()) or np.any(a<0) or np.any(d<0):
        raise ValueError("invalid overlapping history counts")
    integral=np.all(a==np.floor(a)) and np.all(d==np.floor(d))
    def tolerance(*operands):
        if integral:return np.zeros_like(operands[0])
        scale=np.maximum.reduce([np.ones_like(operands[0]),*[np.abs(x) for x in operands]])
        return 16*np.finfo(float).eps*scale
    if np.any(np.abs(a[:,:,:-1]-d[:,:,:-1]-a[:,:,1:])>tolerance(a[:,:,:-1],d[:,:,:-1],a[:,:,1:])):
        raise ValueError("overlap requires death-only accounting")
    survivors=(a-d)[:,:,-1:]
    if np.any(d-a>tolerance(a,d)):
        raise ValueError("invalid overlapping history counts")
    cells=np.concatenate([d,survivors],axis=2)
    # Preserve the scales of the original survivor operands, since the
    # survivor counts may be the small difference of large stock/event counts.
    scale=np.concatenate([np.abs(d),np.maximum(np.abs(a[:,:,-1:]),np.abs(d[:,:,-1:]))],axis=2)
    remainder=cells[2]-cells[0]-cells[1]
    if np.any(remainder < -tolerance(scale[0],scale[1],scale[2])):
        raise ValueError("invalid overlapping history counts: negative residual deaths or final survivors under death-only accounting")
    cells[2]=np.maximum(remainder,0)
    cells[:2]=np.maximum(cells[:2],0)
    return dates,cells

"""Fixed-cohort person bootstrap for ordinary KCOR reporting.

Each mutually exclusive enrollment cohort is sampled using its death/exit/survival
history cells. Each draw is passed through the ordinary fitter without recursive
interval generation. Overlapping age-shift controls require their known layout.
"""
import os,sys,io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout,redirect_stderr
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

KEYS=["EnrollmentDate","YearOfBirth","Dose_num","Dose_den","Date"]
def keyframe(df):
    a=df.copy()
    a["Date"]=pd.to_datetime(a["Date"])
    if a.duplicated(KEYS).any():raise ValueError("nonunique ordinary output keys")
    return a.set_index(KEYS)

def histories(frame,sheet,layout):
    need={"DateDied","ISOweekDied","YearOfBirth","Sex","Dose","Alive","Dead"}
    if not need<=set(frame.columns):raise ValueError("missing fixed-cohort history columns")
    if "PT" in frame.columns:
        if not np.array_equal(frame.PT.to_numpy(float),frame.Alive.to_numpy(float)):
            raise ValueError("person-time histories require individual histories")
        frame=frame.drop(columns="PT")
    from KCOR import _parse_enrollment_date
    start=_parse_enrollment_date(sheet)
    frame=frame.copy();frame["DateDied"]=pd.to_datetime(frame.DateDied)
    frame=frame[frame.DateDied>=start]
    if not len(frame):raise ValueError("no post-enrollment rows")
    from fixed_cohort_history import invalid_components,aggregate_reconciliation,aggregate_scope
    scope=aggregate_scope(layout)
    bad=invalid_components(frame,start,layout=layout)
    supplied=frame.YearOfBirth.isin([-2,-60])
    # In a mixed extract, only malformed supplied totals can be excluded
    # without changing the disjoint component populations.
    if bad and (layout!="disjoint" or supplied.all() or any(k[0] not in (-2,-60) for k in bad)):
        raise ValueError(f"invalid source component histories: {bad}")

    if layout=="table8":
        raw_counts=frame[["Alive","Dead"]].to_numpy(float)
        if not np.isfinite(raw_counts).all() or np.any(raw_counts!=np.floor(raw_counts)):
            raise ValueError("invalid overlapping history counts")
        from fixed_cohort_history import overlap_history_cells
        group=frame.groupby(["YearOfBirth","Dose","DateDied"])[["Alive","Dead"]].sum()
        dates,cells=overlap_history_cells(group,start,frame.DateDied.max())
        return ("table8",dates,cells.astype(np.int64))
    if layout!="disjoint":raise ValueError("unknown bootstrap layout")
    if not set(frame.Sex.astype(str).unique())<= {"F","M","O"}:
        raise ValueError("unrecognized sex category in fixed cohort histories")
    # The converter emits disjoint birth-year/sex/DCCI/dose cells plus known
    # overlapping totals: -2 = all ages, -60 = birth year 1960 or later.
    aggregate=frame[frame.YearOfBirth.isin([-2,-60])].copy()
    base=frame[~frame.YearOfBirth.isin([-2,-60])].copy()
    if not aggregate.empty and base.empty:
        if aggregate.YearOfBirth.nunique()!=1:
            raise ValueError("overlapping aggregates without their disjoint components")
        # A single aggregate on its own is a population, not overlapping data.
        base=aggregate;aggregate=aggregate.iloc[:0]
    if not aggregate.empty and ((base.YearOfBirth<=0)&(base.YearOfBirth!=-1)).any():
        raise ValueError("unrecognized birth-year aggregates mixed with converter totals")
    if ((base.YearOfBirth<=0)&(base.YearOfBirth!=-1)).any() and base.YearOfBirth.nunique()>1:
        raise ValueError("unknown overlapping birth-year strata")
    invalid,unsupported=aggregate_reconciliation(frame,scope)
    unsupported |= set(bad)
    # Keep verified redundant total templates only. Do not draw from a total
    # and its constituent people independently or silently replace its population.
    if not aggregate.empty:
        keys=list(zip(aggregate.YearOfBirth,aggregate.Dose))
        aggregate=aggregate.loc[[k not in unsupported for k in keys]].copy()
    group=base.groupby(["YearOfBirth","Dose"],sort=True)
    cohorts=[]
    for (age,dose),g in group:
        components=[];cells=[]
        strata=["Sex"]+(["DCCI"] if "DCCI" in g else [])
        for key,x in g.groupby(strata,sort=True,dropna=False):
            x=x.sort_values("DateDied")
            dates=pd.DatetimeIndex(x.DateDied)
            if dates.has_duplicates or dates[0]!=start or not np.all(np.diff(dates.values)==np.timedelta64(7,"D")):
                raise ValueError("incomplete or duplicate weekly risk history")
            a=x.Alive.to_numpy(float);d=x.Dead.to_numpy(float)
            exits=a[:-1]-d[:-1]-a[1:]
            if "Dead_COVID" in x:
                covid=x.Dead_COVID.to_numpy(float)
                c=np.r_[d-covid,covid,exits,a[-1]-d[-1]]
            else:
                c=np.r_[d,exits,a[-1]-d[-1]]
            if not np.isfinite(c).all() or np.any(c<0) or np.any(c!=np.floor(c)):
                raise ValueError("risk sets do not reconstruct nonnegative integer person histories")
            components.append((key,x.copy(),len(c)));cells.extend(c.astype(np.int64))
        cohorts.append((components,np.array(cells,dtype=np.int64)))
    if not cohorts:raise ValueError("no disjoint cohort histories")
    return ("disjoint",cohorts,aggregate,unsupported)


def draw_history(record,rng):
    if record[0]=="table8":
        _,dates,cells=record
        n=int(cells.sum());sample=rng.multinomial(n,cells.ravel()/n).reshape(cells.shape)
        sample[2]+=sample[0]+sample[1]
        alive=np.cumsum(sample[:,:,::-1],axis=2)[:,:,::-1][:,:,:-1]
        rows=[]
        for i,age in enumerate([1940,1950,1960]):
            for j,dose in enumerate([0,1,2]):
                rows.append(pd.DataFrame(dict(DateDied=dates,ISOweekDied=dates.strftime("%G-%V"),YearOfBirth=age,Dose=dose,Sex="F",Alive=alive[i,j],Dead=sample[i,j,:-1])))
        return pd.concat(rows,ignore_index=True)
    rows=[]
    for components,cells in record[1]:
        n=int(cells.sum())
        sampled=rng.multinomial(n,cells/n) if n else cells.copy()
        pos=0
        for key,x,size in components:
            c=sampled[pos:pos+size];pos+=size
            nw=len(x)
            y=x.copy()
            if "Dead_COVID" in x:
                y["Dead_COVID"]=c[nw:2*nw]
                dead=c[:nw]+c[nw:2*nw];exits=c[2*nw:-1]
            else:
                dead=c[:nw];exits=c[nw:-1]
            y["Dead"]=dead;y["Alive"]=int(c.sum())-np.r_[0,np.cumsum(dead[:-1]+exits)]
            if "cumDead_prev" in y:y["cumDead_prev"]=np.r_[0,np.cumsum(dead[:-1])]
            rows.append(y)
    base=pd.concat(rows,ignore_index=True)
    # Reconstruct only the aggregate rows present in the original workbook.
    # They are sums of the very same sampled people, not extra draws.
    aggregate=record[2] if len(record)>2 else pd.DataFrame()
    count_cols=["Alive","Dead"]+(["Dead_COVID"] if "Dead_COVID" in base else [])
    for age,template in aggregate.groupby("YearOfBirth",sort=True) if not aggregate.empty else []:
        source=base if age==-2 else base[base.YearOfBirth>=1960]
        sums=source.groupby(["Dose","DateDied"])[count_cols].sum()
        y=template.copy()
        idx=pd.MultiIndex.from_frame(y[["Dose","DateDied"]])
        y[count_cols]=sums.reindex(idx,fill_value=0).to_numpy()
        if "cumDead_prev" in y:
            y=y.sort_values(["Dose","DateDied"])
            y["cumDead_prev"]=y.groupby("Dose")["Dead"].transform(lambda z:z.cumsum().shift(fill_value=0))
        rows.append(y)
    return pd.concat(rows,ignore_index=True)


def init_worker(hist, settings, seed, input_path, sheet_indices=None):
    global HIST,SETTINGS,SEED,INPUT,SHEET_INDICES
    HIST,SETTINGS,SEED,INPUT=hist,settings,seed,input_path
    SHEET_INDICES=sheet_indices or {sh:i for i,sh in enumerate(hist)}

def worker(draw):
    import KCOR as K
    for k,v in SETTINGS.items():setattr(K,k,v)
    frames={sh:draw_history(h,np.random.default_rng([SEED,draw,SHEET_INDICES[sh]])) for sh,h in HIST.items()}
    try:
        with TemporaryDirectory(prefix="kcor-person-draw-") as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out=K.process_workbook(INPUT,str(Path(tmp)/"unused.xlsx"),bootstrap_frames=frames,return_rows=True)
        result=keyframe(out)
        result.attrs["gap_first_pass"] = out.attrs.get("gap_first_pass", {})
        return result,None
    except (ValueError,ArithmeticError) as e:
        return None,f"{type(e).__name__}: {e}"

def annotate_residual_screen(frame, params, *, mc_batched=False):
    """Separate exploratory lack-of-fit from numerical fit failure."""
    states = {}
    for key,p in params.items():
        d=p.get("sampling_diagnostic",{})
        states[key] = "unavailable" if not d.get("sampling_n",0) else ("flag" if d.get("sampling_flag",False) else "clear")
    result=[]
    for row in frame.itertuples(index=False):
        if row.YearOfBirth==0:
            label=getattr(row,"pool_contributors","")
            ages=[int(a) for a in str(label).split(",") if a.lstrip("-").isdigit()]
        else: ages=[int(row.YearOfBirth)]
        prefix=(row.EnrollmentDate,row.mc_id) if mc_batched else (row.EnrollmentDate,)
        v=[states.get(prefix+(a,d),"unavailable") for a in ages for d in (row.Dose_num,row.Dose_den)]
        result.append(("flag" in v, not v or "unavailable" in v))
    frame["residual_screen_flag"]=[x[0] for x in result]
    frame["residual_screen_unavailable"]=[x[1] for x in result]


def flagged(frame,unanchored=False):
    # Preserve the estimator's configured fallback; expose its use rather than
    # conditioning the bootstrap on accepted draws.
    bad=np.zeros(len(frame),dtype=bool)
    fit_col="unanchored_abnormal_fit" if unanchored and "unanchored_abnormal_fit" in frame else "abnormal_fit"
    status_col="unanchored_normalization_status" if unanchored and "unanchored_normalization_status" in frame else "normalization_status"
    if fit_col in frame:
        bad |= frame[fit_col].fillna(False).astype(bool).to_numpy()
    if status_col in frame:
        status = frame[status_col]
        bad |= (status.notna() & ~status.astype(str).str.lower().isin(["ok", ""])).to_numpy()
    for col in ("theta0_status_num","theta0_status_den"):
        if col in frame:
            status=frame[col]
            bad |= (status.notna() & ~status.astype(str).str.lower().isin(["ok",""])).to_numpy()
    # Direct and pooled rows already carry their own contributing-fit flags.
    return bad


def gap_diagnostic(point, draws, point_seed_status=None):
    """Pointwise 95% percentile intervals for signed first-pass gap offsets.

    Missing/incomplete draws are not treated as a pass. No fits or intervals
    are changed by these assumption diagnostics.
    """
    summary=[];status={}
    n=len(draws)
    for key,gaps in point.items():
        if gaps is None:
            status[key]="not_assessed";continue
        if not gaps:
            status[key]="no_gaps";continue
        status[key]="no_deficit_detected"
        for week,value in gaps:
            sample=np.asarray([dict(d.get(key) or []).get(week,np.nan) for d in draws],float)
            finite=int(np.isfinite(sample).sum())
            complete=bool(n and finite==n and np.isfinite(float(value) if value is not None else np.nan))
            lower,upper=np.quantile(sample,[.025,.975]) if complete else (np.nan,np.nan)
            flag="violation" if complete and upper<0 else ("no_deficit_detected" if complete else "not_assessed")
            if flag=="violation":status[key]=flag
            elif flag=="not_assessed" and status[key]!="violation":status[key]=flag
            summary.append(dict(EnrollmentDate=key[0],YearOfBirth=key[1],Dose=key[2],rebased_reentry_week=week,
                signed_first_pass_offset=value,lower95=float(lower),upper95=float(upper),valid_draws=finite,requested_draws=n,status=flag,
                reason=((point_seed_status or {}).get(key) or "point_offset_unavailable") if not np.isfinite(float(value) if value is not None else np.nan) else "incomplete_draws" if not complete else ""))
    return summary,status


def annotate_gap_diagnostic(model,out,point,draws,log):
    summary,status=gap_diagnostic(point,draws,out.attrs.get("gap_seed_status",{}))
    out.attrs["gap_diagnostics"]=summary
    out["gap_assumption_status"]="not_assessed"
    out["abnormal_fit"]=out["abnormal_fit"].astype(object)
    # Direct comparison uses its two fits; pooled comparisons use the
    # positive-weight paired ages, matching the original point population.
    for (sheet,age,num,den),idx in out.groupby(["EnrollmentDate","YearOfBirth","Dose_num","Dose_den"],sort=False).groups.items():
        if age==0:
            groups=[([int(a) for a in label.split(",") if a], subidx.index) for label,subidx in out.loc[idx,"pool_contributors"].fillna("").groupby(out.loc[idx,"pool_contributors"].fillna(""))] if "pool_contributors" in out else [([],idx)]
        else:groups=[([age],idx)]
        for ages,subidx in groups:
            states=[status.get((sheet,a,d),"not_assessed") for a in ages for d in (num,den)]
            state=("violation" if "violation" in states else "not_assessed" if not states or "not_assessed" in states else "no_deficit_detected" if "no_deficit_detected" in states else "no_gaps")
            out.loc[subidx,"gap_assumption_status"]=state
            if state=="violation":
                out.loc[subidx,"abnormal_fit"]=True
                out.loc[subidx,"CI_point_flagged"]=True
                out.loc[subidx,"CI_unanchored_point_flagged"]=True
                if "unanchored_abnormal_fit" in out:
                    out.loc[subidx,"unanchored_abnormal_fit"]=True
                if "CI_method" in out:
                    use=out.index.isin(subidx)&out.CI_method.eq("person_bootstrap_percentile")
                    out.loc[use,"CI_method"]="person_bootstrap_with_fit_flags"
                if "CI_unanchored_method" in out:
                    use=out.index.isin(subidx)&out.CI_unanchored_method.eq("person_bootstrap_percentile")
                    out.loc[use,"CI_unanchored_method"]="person_bootstrap_with_fit_flags"
    for row in summary:
        if row["status"]=="violation":log(f"[KCOR_GAP_ASSUMPTION_VIOLATION] {row}")
    return out


def attach(model,frame,src,sheets,log):
    out=frame.copy()
    point_gaps=frame.attrs.get("gap_first_pass",{})
    out["CI_lower"]=np.nan;out["CI_upper"]=np.nan
    out["CI_method"]="unavailable";out["CI_valid_draws"]=0
    out["CI_flagged_draws"]=0;out["CI_point_flagged"]=flagged(out)
    out["CI_residual_flagged_draws"]=0;out["CI_residual_unavailable_draws"]=0
    out["CI_undefined_anchor_draws"]=0;out["CI_missing_draws"]=0
    out["CI_pool_composition_changed_draws"]=0
    out["CI_population_scope_changed_draws"]=0
    out["CI_unanchored_lower"]=np.nan;out["CI_unanchored_upper"]=np.nan;out["CI_unanchored_valid_draws"]=0
    out["CI_unanchored_method"]="unavailable"
    out["CI_unanchored_point_flagged"]=flagged(out,unanchored=True)
    out["CI_unanchored_flagged_draws"]=0
    layout=os.environ.get("KCOR_BOOTSTRAP_LAYOUT","disjoint")
    draws=int(os.environ.get("KCOR_BOOTSTRAP_DRAWS","999"))
    if draws<=0:
        log("[BOOTSTRAP] disabled; intervals unavailable");return annotate_gap_diagnostic(model,out,point_gaps,[],log)
    hist={};sheet_indices={sh:i for i,sh in enumerate(sheets)}
    for sh in sheets:
        try:
            hist[sh]=histories(pd.read_excel(src,sheet_name=sh),sh,layout)
            if hist[sh][0]=="disjoint" and len(hist[sh])>3:
                for age,dose in sorted(hist[sh][3]):
                    log(f"[BOOTSTRAP] {sh}: supplied aggregate {age}, dose {dose} cannot be jointly resampled from source components; affected intervals unavailable")
        except (ValueError,TypeError,KeyError) as e:
            log(f"[BOOTSTRAP] {sh} unavailable: {e}")
    if not hist:return annotate_gap_diagnostic(model,out,point_gaps,[],log)
    keys=keyframe(out).index
    supported=keys.get_level_values("EnrollmentDate").isin(hist)
    aggregate_unsupported=np.array([
        sh in hist and hist[sh][0]=="disjoint" and len(hist[sh])>3
        and ((age,num) in hist[sh][3] or (age,den) in hist[sh][3])
        for sh,age,num,den,_ in keys],dtype=bool)
    supported &= ~aggregate_unsupported
    unsupported_groups={(sh,age,dose) for sh,h in hist.items()
                        if h[0]=="disjoint" and len(h)>3 for age,dose in h[3]}
    pooled=keys.get_level_values("YearOfBirth")==0
    point_members=keyframe(out).get("pool_contributors",pd.Series("",index=keys)).fillna("").astype(str).to_numpy()
    point_labels=keyframe(out).get("population_label",pd.Series("",index=keys)).fillna("").astype(str).to_numpy()
    coverage_cols=("population_age_groups_num","population_age_groups_den","population_zero_enrollment_groups_num","population_zero_enrollment_groups_den")
    point_coverage={col:keyframe(out).get(col,pd.Series("",index=keys)).fillna("").astype(str).to_numpy() for col in coverage_cols}
    point_pool=pooled & np.isfinite(out["KCOR"].to_numpy(float)) & (point_members!="")
    settings={k:getattr(model,k) for k in ("ENROLLMENT_DATES","OVERRIDE_DOSE_PAIRS","OVERRIDE_YOBS","KCOR_NORMALIZATION_WEEKS","KCOR_NORMALIZATION_WEEKS_EFFECTIVE","DYNAMIC_HVE_SKIP_WEEKS","YEAR_RANGE","AGE_RANGE","MONTE_CARLO_MODE")}
    settings["ENROLLMENT_DATES"]=list(hist)
    seed=int(os.environ.get("KCOR_BOOTSTRAP_SEED","33080"))
    values=[];flags=[];raw_values=[];raw_flags=[];invalid_anchors=[];missing=[];gap_draws=[];composition=[];scope_changes=[];residual_flags=[];residual_missing=[]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("KCOR_BOOTSTRAP_WORKERS","5")),initializer=init_worker,initargs=(hist,settings,seed,str(Path(src).resolve()),sheet_indices)) as pool:
        for i,(v,error) in enumerate(pool.map(worker,range(1,draws+1),chunksize=1),1):
            if error:
                log(f"[BOOTSTRAP] draw {i} unavailable: {error}")
                val=np.full(len(keys),np.nan);bad=np.ones(len(keys),dtype=bool)
                raw=val.copy();raw_bad=bad.copy();invalid=np.zeros(len(keys),dtype=bool);absent=np.ones(len(keys),dtype=bool);changed=np.zeros(len(keys),dtype=bool);scope_changed=changed.copy()
            else:
                val=v["KCOR"].reindex(keys).to_numpy(float)
                raw=v["KCOR_unanchored"].reindex(keys).to_numpy(float).copy()
                anchor=v["anchor_status"].reindex(keys)
                invalid=anchor.isin(["undefined_reference_ratio","missing_reference_week","pooled_no_valid_reference"]).to_numpy(bool).copy()
                absent=~keys.isin(v.index)
                bad=pd.Series(flagged(v),index=v.index).reindex(keys).fillna(True).to_numpy(bool)
                raw_bad=pd.Series(flagged(v,unanchored=True),index=v.index).reindex(keys).fillna(True).to_numpy(bool)
                members=v.get("pool_contributors",pd.Series("",index=v.index)).reindex(keys).fillna("").astype(str).to_numpy()
                changed=point_pool & ~absent & (members!=point_members)
                draw_labels=v.get("population_label",pd.Series("",index=v.index)).reindex(keys).fillna("").astype(str).to_numpy()
                scope_changed=~absent & (point_labels!=draw_labels) & (point_labels!="")
                for col, labels in point_coverage.items():
                    draw_coverage=v.get(col,pd.Series("",index=v.index)).reindex(keys).fillna("").astype(str).to_numpy()
                    scope_changed |= ~absent & (labels!="") & (labels!=draw_coverage)
                # A finite result for a different pool is not a valid draw
                # for the point's population; retain counts, not that value.
                val=val.copy();val[changed|scope_changed]=np.nan
                raw[scope_changed]=np.nan
                bad=bad | changed | scope_changed
                raw_bad=raw_bad | scope_changed
            val[aggregate_unsupported]=np.nan
            raw[aggregate_unsupported]=np.nan
            invalid[aggregate_unsupported]=False
            gap_draws.append({} if error else {k:x for k,x in v.attrs.get("gap_first_pass",{}).items() if k not in unsupported_groups})
            residual_flags.append(np.zeros(len(keys),bool) if error else v.get("residual_screen_flag",pd.Series(False,index=v.index)).reindex(keys).fillna(False).to_numpy(bool))
            residual_missing.append(np.ones(len(keys),bool) if error else v.get("residual_screen_unavailable",pd.Series(True,index=v.index)).reindex(keys).fillna(True).to_numpy(bool))
            values.append(val);flags.append(bad);raw_values.append(raw);raw_flags.append(raw_bad);invalid_anchors.append(invalid);missing.append(absent);composition.append(changed);scope_changes.append(scope_changed)
            if i%100==0:log(f"[BOOTSTRAP] {i}/{draws}")
    a=np.asarray(values);count=np.isfinite(a).sum(axis=0);out["CI_valid_draws"]=count
    out["CI_flagged_draws"]=np.where(supported,np.asarray(flags).sum(axis=0),0)
    out["CI_residual_flagged_draws"]=np.where(supported,np.asarray(residual_flags).sum(axis=0),0)
    out["CI_residual_unavailable_draws"]=np.where(supported,np.asarray(residual_missing).sum(axis=0),0)
    out["CI_undefined_anchor_draws"]=np.asarray(invalid_anchors).sum(axis=0)
    out["CI_missing_draws"]=np.where(supported,np.asarray(missing).sum(axis=0),0)
    out["CI_pool_composition_changed_draws"]=np.where(supported,np.asarray(composition).sum(axis=0),0)
    out["CI_population_scope_changed_draws"]=np.where(supported,np.asarray(scope_changes).sum(axis=0),0)
    raw_a=np.asarray(raw_values);raw_count=np.isfinite(raw_a).sum(axis=0)
    out["CI_unanchored_valid_draws"]=raw_count
    out["CI_unanchored_flagged_draws"]=np.where(supported,np.asarray(raw_flags).sum(axis=0),0)
    raw_good=(raw_count==draws)&np.isfinite(out["KCOR_unanchored"].to_numpy(float))
    if raw_good.any():
        out.loc[raw_good,["CI_unanchored_lower","CI_unanchored_upper"]]=np.quantile(raw_a[:,raw_good],[.025,.975],axis=0).T
        out.loc[raw_good,"CI_unanchored_method"]="person_bootstrap_percentile"
        raw_flagged=raw_good & ((out["CI_unanchored_flagged_draws"]>0)|out["CI_unanchored_point_flagged"]).to_numpy()
        out.loc[raw_flagged,"CI_unanchored_method"]="person_bootstrap_with_fit_flags"
    # Do not silently condition the interval on a subset of successful draws.
    good=(count==draws)&np.isfinite(out["KCOR"].to_numpy())
    if good.any():
        out.loc[good,["CI_lower","CI_upper"]]=np.quantile(a[:,good],[.025,.975],axis=0).T
        out.loc[good,"CI_method"]="person_bootstrap_percentile"
        flagged_rows=good & ((out["CI_flagged_draws"]>0)|out["CI_point_flagged"]).to_numpy()
        out.loc[flagged_rows,"CI_method"]="person_bootstrap_with_fit_flags"
    log(f"[BOOTSTRAP] {draws} person draws; {int(good.sum())}/{len(out)} rows received full intervals")
    return annotate_gap_diagnostic(model,out,point_gaps,gap_draws,log)

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
    if "PT" in frame.columns:raise ValueError("person-time histories require individual histories")
    start=pd.Timestamp.fromisocalendar(*map(int,sheet.split("_")),1)
    frame=frame.copy();frame["DateDied"]=pd.to_datetime(frame.DateDied)
    frame=frame[frame.DateDied>=start]
    if not len(frame):raise ValueError("no post-enrollment rows")
    if layout=="table8":
        group=frame.groupby(["YearOfBirth","Dose","DateDied"])[["Alive","Dead"]].sum().reset_index()
        if sorted(group.YearOfBirth.unique())!=[1940,1950,1960] or sorted(group.Dose.unique())!=[0,1,2]:
            raise ValueError("unexpected overlapping age-shift layout")
        dates=pd.date_range(start,group.DateDied.max(),freq="7D")
        idx=pd.MultiIndex.from_product([[1940,1950,1960],[0,1,2],dates],names=["YearOfBirth","Dose","DateDied"])
        v=group.set_index(["YearOfBirth","Dose","DateDied"]).reindex(idx)
        if v.isna().any().any():raise ValueError("incomplete overlapping histories")
        a=v.Alive.to_numpy().reshape(3,3,-1);d=v.Dead.to_numpy().reshape(3,3,-1)
        if np.any(a[:,:,:-1]-d[:,:,:-1]!=a[:,:,1:]):raise ValueError("overlap requires death-only fixed histories")
        cells=np.concatenate([d,(a-d)[:,:,-1:]],axis=2)
        cells[2]-=cells[0]+cells[1]
        if np.any(cells<0) or np.any(cells!=np.floor(cells)):raise ValueError("invalid overlapping history counts")
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
    count_cols=["Alive","Dead"]+(["Dead_COVID"] if "Dead_COVID" in frame else [])
    agg_keys=["Dose","DateDied"]
    for age,template in aggregate.groupby("YearOfBirth",sort=True):
        if template.duplicated(agg_keys).any() or not template.Sex.eq("O").all() or ("DCCI" in template and not template.DCCI.eq(-1).all()):
            raise ValueError("unexpected converter aggregate layout")
        source=base if age==-2 else base[base.YearOfBirth>=1960]
        observed=template.set_index(agg_keys)[count_cols].sort_index()
        expected=source.groupby(agg_keys)[count_cols].sum().sort_index()
        if not observed.index.equals(expected.index) or not np.array_equal(observed.to_numpy(),expected.to_numpy()):
            raise ValueError(f"aggregate {age} does not match disjoint source counts")
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
    return ("disjoint",cohorts,aggregate)


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
        y[count_cols]=sums.reindex(idx).to_numpy()
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
        return keyframe(out),None
    except (ValueError,ArithmeticError) as e:
        return None,f"{type(e).__name__}: {e}"

def flagged(frame):
    # Preserve the estimator's configured fallback; expose its use rather than
    # conditioning the bootstrap on accepted draws.
    bad=np.zeros(len(frame),dtype=bool)
    if "abnormal_fit" in frame:
        bad |= frame["abnormal_fit"].fillna(False).astype(bool).to_numpy()
    for col in ("theta0_status_num","theta0_status_den"):
        if col in frame:
            status=frame[col]
            bad |= (status.notna() & ~status.astype(str).str.lower().isin(["ok",""])).to_numpy()
    # For aggregated rows, inherit flags from the contributing age strata.
    z=frame.reset_index() if isinstance(frame.index,pd.MultiIndex) else frame.copy()
    z["_bootstrap_flag"]=bad
    groups=["EnrollmentDate","Dose_num","Dose_den","Date"]
    pool=z.YearOfBirth<=0
    any_group=z.groupby(groups,dropna=False)["_bootstrap_flag"].transform("any").to_numpy()
    return bad | (pool.to_numpy() & any_group)

def attach(model,frame,src,sheets,log):
    out=frame.copy()
    out["CI_analytic_lower"]=out["CI_lower"];out["CI_analytic_upper"]=out["CI_upper"]
    out["CI_lower"]=np.nan;out["CI_upper"]=np.nan
    out["CI_method"]="unavailable";out["CI_valid_draws"]=0
    out["CI_flagged_draws"]=0;out["CI_point_flagged"]=flagged(out)
    out["CI_undefined_anchor_draws"]=0;out["CI_missing_draws"]=0
    out["CI_unanchored_lower"]=np.nan;out["CI_unanchored_upper"]=np.nan;out["CI_unanchored_valid_draws"]=0
    layout=os.environ.get("KCOR_BOOTSTRAP_LAYOUT","disjoint")
    draws=int(os.environ.get("KCOR_BOOTSTRAP_DRAWS","999"))
    if draws<=0:
        log("[BOOTSTRAP] disabled; analytic intervals separately labeled");return out
    hist={};sheet_indices={sh:i for i,sh in enumerate(sheets)}
    for sh in sheets:
        try:
            hist[sh]=histories(pd.read_excel(src,sheet_name=sh),sh,layout)
        except (ValueError,TypeError,KeyError) as e:
            log(f"[BOOTSTRAP] {sh} unavailable: {e}")
    if not hist:return out
    keys=keyframe(out).index
    supported=keys.get_level_values("EnrollmentDate").isin(hist)
    settings={k:getattr(model,k) for k in ("ENROLLMENT_DATES","OVERRIDE_DOSE_PAIRS","OVERRIDE_YOBS","KCOR_NORMALIZATION_WEEKS","KCOR_NORMALIZATION_WEEKS_EFFECTIVE","DYNAMIC_HVE_SKIP_WEEKS","YEAR_RANGE","MONTE_CARLO_MODE")}
    settings["ENROLLMENT_DATES"]=list(hist)
    seed=int(os.environ.get("KCOR_BOOTSTRAP_SEED","33080"))
    values=[];flags=[];raw_values=[];invalid_anchors=[];missing=[]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("KCOR_BOOTSTRAP_WORKERS","5")),initializer=init_worker,initargs=(hist,settings,seed,str(Path(src).resolve()),sheet_indices)) as pool:
        for i,(v,error) in enumerate(pool.map(worker,range(1,draws+1),chunksize=1),1):
            if error:
                log(f"[BOOTSTRAP] draw {i} unavailable: {error}")
                val=np.full(len(keys),np.nan);bad=np.ones(len(keys),dtype=bool)
                raw=val.copy();invalid=np.zeros(len(keys),dtype=bool);absent=np.ones(len(keys),dtype=bool)
            else:
                val=v["KCOR"].reindex(keys).to_numpy(float)
                raw=v["KCOR_unanchored"].reindex(keys).to_numpy(float)
                anchor=v["anchor_status"].reindex(keys)
                invalid=anchor.isin(["undefined_reference_ratio","missing_reference_week","pooled_no_valid_reference"]).to_numpy(bool)
                absent=~keys.isin(v.index)
                bad=pd.Series(flagged(v),index=v.index).reindex(keys).fillna(True).to_numpy(bool)
            values.append(val);flags.append(bad);raw_values.append(raw);invalid_anchors.append(invalid);missing.append(absent)
            if i%100==0:log(f"[BOOTSTRAP] {i}/{draws}")
    a=np.asarray(values);count=np.isfinite(a).sum(axis=0);out["CI_valid_draws"]=count
    out["CI_flagged_draws"]=np.where(supported,np.asarray(flags).sum(axis=0),0)
    out["CI_undefined_anchor_draws"]=np.asarray(invalid_anchors).sum(axis=0)
    out["CI_missing_draws"]=np.where(supported,np.asarray(missing).sum(axis=0),0)
    raw_a=np.asarray(raw_values);raw_count=np.isfinite(raw_a).sum(axis=0)
    out["CI_unanchored_valid_draws"]=raw_count
    raw_good=(raw_count==draws)&np.isfinite(out["KCOR_unanchored"].to_numpy(float))
    if raw_good.any():
        out.loc[raw_good,["CI_unanchored_lower","CI_unanchored_upper"]]=np.quantile(raw_a[:,raw_good],[.025,.975],axis=0).T
    # Do not silently condition the interval on a subset of successful draws.
    good=(count==draws)&np.isfinite(out["KCOR"].to_numpy())
    good &= out["CI_analytic_lower"].notna().to_numpy()
    if good.any():
        out.loc[good,["CI_lower","CI_upper"]]=np.quantile(a[:,good],[.025,.975],axis=0).T
        out.loc[good,"CI_method"]="person_bootstrap_percentile"
        flagged_rows=good & ((out["CI_flagged_draws"]>0)|out["CI_point_flagged"]).to_numpy()
        out.loc[flagged_rows,"CI_method"]="person_bootstrap_with_fit_flags"
    log(f"[BOOTSTRAP] {draws} person draws; {int(good.sum())}/{len(out)} rows received full intervals")
    return out
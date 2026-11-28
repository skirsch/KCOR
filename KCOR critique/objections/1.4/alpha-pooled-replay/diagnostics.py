"""Vectorized historical objective and matched diagnostic resamples."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("pooled_alpha_replay",Path(__file__).resolve().parent/"replay.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
import numpy as np,pandas as pd,json
from scipy.special import gammaln

def objective(subset,factors,mode="exclude_nonpositive"):
    h=r.hist;a=r.config["analysis"];grid=r.grid
    raw=subset.excess.to_numpy(float);dead=subset.Dead.to_numpy(float)
    floor=h.build_floor(subset.excess)
    transformed,valid=h.transform_excess(raw,mode,floor)
    pair=np.zeros(len(grid));collapse=np.zeros(len(grid));npairs=np.zeros(len(grid),int);npoints=np.zeros(len(grid),int);pairweeks=np.zeros(len(grid),int);clweeks=np.zeros(len(grid),int)
    for _,g in subset.groupby("iso_int"):
        ii=g.index.to_numpy(int)
        fac=np.maximum(factors[ii],floor)
        ix=ii[valid[ii]]
        if len(ix)>=2:
            u,v=np.triu_indices(len(ix),1);i,j=ix[u],ix[v]
            w=np.minimum(dead[i],dead[j]);keep=w>0;i,j,w=i[keep],j[keep],w[keep]
            if len(w):
                diff=(transformed[i]-transformed[j])[:,None]-(np.log(np.maximum(factors[i],floor))-np.log(np.maximum(factors[j],floor)))
                pair+=np.sum(w[:,None]*diff**2,axis=0);npairs+=len(w);pairweeks+=1
        neutral=raw[ii,None]/fac
        if mode=="exclude_nonpositive":
            mask=np.isfinite(neutral)&(neutral>0);val=np.log(np.maximum(neutral,floor))
        elif mode=="truncate_zero":
            mask=np.isfinite(neutral);val=np.log(np.maximum(neutral,floor))
        else:
            mask=np.isfinite(neutral)&(abs(neutral)>0);val=np.sign(neutral)*np.log(np.maximum(abs(neutral),floor))
        val=np.where(mask,val,0.);weights=dead[ii,None]*mask
        tot=weights.sum(axis=0);mu=np.divide((weights*val).sum(axis=0),tot,where=tot>0,out=np.zeros_like(tot))
        n=mask.sum(axis=0);use=n>=a["min_cohorts_per_week"]
        collapse+=np.where(use,(weights*(val-mu)**2).sum(axis=0),0.);npoints+=n*use;clweeks+=use
    return {name:pd.DataFrame(dict(alpha=grid,objective=np.where(counts>=minn,ob,np.nan),n_pairs=counts,n_weeks_used=weeks))
        for name,ob,counts,weeks,minn in (("pairwise",pair,npairs,pairweeks,a["min_pairs_per_alpha"]),("collapse",collapse,npoints,clweeks,a["min_cohorts_per_week"]))}

def factor_matrix(df,arm,moments=None):
    if arm=="exact":
        key=pd.MultiIndex.from_frame(df[["cohort_id","t_week"]])
        matrix=moments.reindex(key)[r.grid.tolist()].to_numpy(float)
        assert np.isfinite(matrix).all() and (matrix>0).all()
        return matrix
    return np.column_stack([r.hist.gamma_moment_alpha(df.theta_t_gamma.to_numpy(float),v) for v in r.grid])

def diagnostics(df,F,arm):
    h=r.hist;cfg=r.config
    # All subsets rebase index to align factor matrix including duplicated cohort rows.
    def curves(ix,mode="exclude_nonpositive"):return objective(df.iloc[ix].reset_index(drop=True),F[ix],mode)
    main=curves(np.arange(len(df)));summ={k:h.summarize_best_curve(c,cfg) for k,c in main.items()}
    original_cohorts=sorted(df.cohort_id.unique())
    indices={key:np.flatnonzero(df.cohort_id.to_numpy()==key) for key in original_cohorts}
    loo=[]
    for key in original_cohorts:
        ix=np.flatnonzero(df.cohort_id.to_numpy()!=key)
        for name,c in curves(ix).items():
            best=h.summarize_best_curve(c,cfg)
            if best:loo.append(dict(arm=arm,estimator=name,omitted=key,**best))
    rng=np.random.default_rng(cfg["bootstrap"]["seed"]);boot=[]
    for rep in range(cfg["bootstrap"]["reps"]):
        ids=rng.choice(original_cohorts,size=len(original_cohorts),replace=True);ix=np.concatenate([indices[k] for k in ids])
        for name,c in curves(ix).items():
            best=h.summarize_best_curve(c,cfg)
            if best:boot.append(dict(arm=arm,estimator=name,rep=rep,**best))
    seg=[]
    for name in ("early_wave","late_wave"):
        ix=np.flatnonzero(df.time_segment.to_numpy()==name)
        for estimator,c in curves(ix).items():
            best=h.summarize_best_curve(c,cfg)
            if best:seg.append(dict(arm=arm,estimator=estimator,segment=name,**best))
    for name in ("truncate_zero","signed_stable"):
        for estimator,c in curves(np.arange(len(df)),name).items():
            best=h.summarize_best_curve(c,cfg)
            if best:seg.append(dict(arm=arm,estimator=estimator,segment=name,**best))
    return main,summ,pd.DataFrame(loo),pd.DataFrame(boot),pd.DataFrame(seg)

def main():
    r.setup();h=r.hist
    original=pd.read_csv(r.HERE/"historical-wave-table.csv",parse_dates=["week_monday"])
    working=pd.read_csv(r.HERE/"working-wave-table.csv",parse_dates=["week_monday"])
    moments=pd.read_csv(r.HERE/"exact-moments.csv").pivot(index=["cohort_id","t_week"],columns="alpha",values="factor")
    # Original and working eligible masks must match for direct comparison.
    a=h.build_primary_subset(original,r.config).reset_index(drop=True)
    b=h.build_primary_subset(working,r.config).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["cohort_id","t_week","excess","Dead"]],b[["cohort_id","t_week","excess","Dead"]])
    print("Common baseline eligible rows",len(a),"cohorts",a.cohort_id.nunique(),flush=True)
    all_curves=[];all_summary=[];all_loo=[];all_boot=[];all_seg=[]
    gates=h.get_identifiability_cfg(r.config)
    for arm,df in (("historical",a),("working",b),("exact",b)):
        F=factor_matrix(df,arm,moments)
        L,s,loo,boot,seg=diagnostics(df,F,arm)
        for name,curve in L.items():
            if arm!="exact":
                ref=pd.read_csv(r.HERE/(arm+"-original-curves.csv")).query("estimator==@name")
                np.testing.assert_allclose(curve.objective,ref.objective,rtol=1e-12,atol=1e-10)
            all_curves.append(curve.assign(arm=arm,estimator=name))
            lo=loo[loo.estimator==name].alpha_hat.to_numpy();bo=boot[boot.estimator==name].alpha_hat.to_numpy()
            loq,hiq=np.quantile(bo,[.025,.975]);iqr=np.diff(np.quantile(bo,[.25,.75]))[0]
            row=dict(arm=arm,estimator=name,**s[name],bootstrap_q025=loq,bootstrap_q975=hiq,bootstrap_iqr=iqr,
                    bootstrap_boundary_fraction=float(boot[boot.estimator==name].boundary_optimum.mean()),loo_min=float(lo.min()),loo_max=float(lo.max()),loo_max_shift=float(np.max(abs(lo-s[name]["alpha_hat"]))),cohorts=int(df.cohort_id.nunique()))
            row["passes_stability"]=bool(row["identified_curve"] and row["loo_max_shift"]<=gates["max_leave_one_out_shift"] and row["bootstrap_boundary_fraction"]<=gates["max_bootstrap_boundary_fraction"] and row["bootstrap_iqr"]<=gates["max_bootstrap_iqr"])
            all_summary.append(row)
        all_loo.append(loo);all_boot.append(boot);all_seg.append(seg)
    for name,rows in (("objective-curves",all_curves),("leave-one-out",all_loo),("bootstrap",all_boot),("sensitivity",all_seg)):pd.concat(rows).to_csv(r.HERE/(name+".csv"),index=False)
    summary=pd.DataFrame(all_summary);summary.to_csv(r.HERE/"summary.csv",index=False)
    print(summary[["arm","estimator","alpha_hat","curvature_metric","n_pairs","n_weeks_used","bootstrap_q025","bootstrap_q975","bootstrap_iqr","bootstrap_boundary_fraction","loo_max_shift","passes_stability"]].round(5).to_string(index=False),flush=True)
    fits=pd.read_csv(r.HERE/"exact-fits.csv")
    print("Exact candidate failed",int((~fits.success).sum()),"flags",int(fits.flag.sum()),"fit calls",len(fits),flush=True)
if __name__=="__main__":main()

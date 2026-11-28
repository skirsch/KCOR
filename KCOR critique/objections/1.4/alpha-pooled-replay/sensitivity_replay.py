"""Diagnostic summaries for original-grid flag sensitivity and wider grid."""
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location("historical_objective_diagnostics",Path(__file__).resolve().parent/"diagnostics.py")
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
import numpy as np,pandas as pd
def main():
    r=d.r;r.setup();h=r.hist
    original=pd.read_csv(r.HERE/"historical-wave-table.csv",parse_dates=["week_monday"])
    working=pd.read_csv(r.HERE/"working-wave-table.csv",parse_dates=["week_monday"])
    a=h.build_primary_subset(original,r.config).reset_index(drop=True);b=h.build_primary_subset(working,r.config).reset_index(drop=True)
    f=pd.read_csv(r.HERE/"exact-fits.csv")
    eligible=set(a.cohort_id)
    flags=set(f[f.flag & f.cohort_id.isin(eligible)].cohort_id)
    print("Flagged groups retained in main:",sorted(flags))
    def run(a,b,wide):
        m=pd.read_csv(r.HERE/"exact-moments.csv")
        if wide:m=pd.concat([pd.read_csv(r.HERE/"lower-grid-moments.csv").query("alpha < 1"),m],ignore_index=True)
        moments=m.pivot(index=["cohort_id","t_week"],columns="alpha",values="factor")
        r.grid=np.array(sorted(moments.columns))
        curves=[];rows=[]
        for label,frame in (("historical",a),("working",b),("exact",b)):
            F=d.factor_matrix(frame,label,moments)
            for scenario,mask in (("all",np.ones(len(frame),bool)),("exclude_flagged",~frame.cohort_id.isin(flags).to_numpy())):
                subset=frame[mask].reset_index(drop=True);C=d.objective(subset,F[mask])
                for name,c in C.items():
                    best=h.summarize_best_curve(c,r.config)
                    rows.append(dict(arm=label,scenario=scenario,estimator=name,**best))
                    curves.append(c.assign(arm=label,scenario=scenario,estimator=name))
        prefix="wider-grid" if wide else "flag-sensitivity"
        pd.DataFrame(rows).to_csv(r.HERE/(prefix+"-summary.csv"),index=False)
        pd.concat(curves).to_csv(r.HERE/(prefix+"-curves.csv"),index=False)
        print(prefix,pd.DataFrame(rows)[["arm","scenario","estimator","alpha_hat","curvature_metric","identified_curve","n_pairs"]].round(6).to_string(index=False))
    run(a,b,False)
    if (r.HERE/"lower-grid-moments.csv").exists():run(a,b,True)
if __name__=="__main__":main()


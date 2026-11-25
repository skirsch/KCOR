"""Historical objective fidelity and numerical comparison."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import replay as r

def original():
    r.setup();h=r.hist;cfg=r.config;grid=r.grid
    for label in ("historical","working"):
        wave=pd.read_csv(r.HERE/(label+"-wave-table.csv"),parse_dates=["week_monday"])
        df=h.build_primary_subset(wave,cfg)
        a=cfg["analysis"]
        curves=[]
        for name,fn,w,minn in (("pairwise",h.evaluate_pairwise_objective,a["weight_mode_pairwise"],a["min_pairs_per_alpha"]),("collapse",h.evaluate_collapse_objective,a["weight_mode_collapse"],a["min_cohorts_per_week"])):
            curve=pd.DataFrame(fn(df,grid,a["primary_excess_mode"],w,"theta_t_gamma",minn))
            print(label,name,h.summarize_best_curve(curve,cfg),flush=True)
            curves.append(curve.assign(arm=label,estimator=name))
        pd.concat(curves).to_csv(r.HERE/(label+"-original-curves.csv"),index=False)
if __name__=="__main__":original()

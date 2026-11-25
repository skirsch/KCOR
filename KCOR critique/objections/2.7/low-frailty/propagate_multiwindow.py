"""Local variance propagation through every step of the current joint fitter."""
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"1.3"))
import table8_bootstrap as table

GROUPS = ("two-dose","unvaccinated","all-persons")


def initialize():
    table.initialize([],"entry-moment")


def analyze(job):
    enrollment,group,dose,dates,risk,dead = job
    model = table.K
    gamma = table.CONFIG["gamma_per_week"]
    time = np.arange(len(dates),dtype=float)
    rebased = time-model.DYNAMIC_HVE_SKIP_WEEKS
    quiet,_,_ = model._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(),time)
    quiet &= rebased>=0
    indices = np.flatnonzero(quiet)
    breaks = np.flatnonzero(np.diff(indices)>1)
    first = indices[:breaks[0]+1] if len(breaks) else indices
    first_mask = np.isin(time,first)
    p = dead/(risk+model.EPS)
    h = np.where(rebased>=0,-np.log1p(-np.clip(p,0,.999)),0.)
    var_h = np.divide(p,risk*(1-p),out=np.zeros_like(h),where=risk>0)
    used = (rebased>=0)&(time<=indices[-1])

    def fit(hazard):
        (k,theta),diag = table.compare.joint_fit(hazard,rebased,quiet,4,gamma,deaths_arr=dead)
        return float(k),float(theta),diag

    k,theta,diag = fit(h)
    row = dict(enrollment=enrollment,subgroup=group,pseudo_dose=dose,theta=theta,
               seed_theta=diag["theta0_init"],status=diag["status"],local_se=np.nan,
               local_se_alt_step=np.nan,first_quiet_variance=np.nan,later_quiet_variance=np.nan,
               other_weeks_variance=np.nan,branch_changes=0)
    if not diag["success"] or diag["insufficient_signal"] or theta<1e-5 or theta>=99.9:
        row["linearization"]="boundary or fallback"
        return row

    def derivatives(scale):
        values = np.zeros_like(h)
        branch_changes = 0
        for i in np.flatnonzero(used & (var_h>0)):
            step = min(max(scale*np.sqrt(var_h[i]),1e-9),h[i]*.4)
            plus,minus = h.copy(),h.copy()
            plus[i] += step
            minus[i] -= step
            _,tp,dp = fit(plus)
            _,tm,dm = fit(minus)
            if not (dp["success"] and dm["success"]) or dp["status"]!=diag["status"] or dm["status"]!=diag["status"]:
                branch_changes += 1
            values[i] = (tp-tm)/(2*step)
        return values**2*var_h,branch_changes

    variances,changes = derivatives(.02)
    alternative,changes2 = derivatives(.005)
    sd = np.sqrt(variances.sum())
    sd2 = np.sqrt(alternative.sum())
    stable = np.isfinite(sd) and np.isfinite(sd2) and abs(sd-sd2)<=.1*max(sd,sd2,1e-12)
    row.update(local_se=sd,local_se_alt_step=sd2,
               first_quiet_variance=variances[first_mask].sum(),
               later_quiet_variance=variances[quiet&~first_mask].sum(),
               other_weeks_variance=variances[used&~quiet].sum(),
               branch_changes=changes+changes2,
               linearization="stable" if stable and not changes+changes2 else "unstable")
    return row


def main():
    book = subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=HERE)
    tasks = []
    for enrollment in table.SHEETS:
        dates,counts = table.read_population(io.BytesIO(book),enrollment)
        counts[2] += counts[0]+counts[1]
        risk = np.cumsum(counts[:,:,::-1],axis=2)[:,:,::-1][:,:,:-1]
        for group,dose in np.ndindex((3,3)):
            tasks.append((enrollment,GROUPS[group],dose,dates,risk[group,dose],counts[group,dose,:-1]))
    with ProcessPoolExecutor(6,initializer=initialize) as pool:
        rows = list(pool.map(analyze,tasks))
    result = pd.DataFrame(rows)
    bootstrap = pd.read_csv(HERE.parents[1]/"1.2/weekly-survival/weekly-comparison.csv")
    bootstrap = bootstrap[bootstrap.wave_week==1][["enrollment","subgroup","pseudo_dose",
        "starting_theta","starting_theta_lower95","starting_theta_upper95"]]
    result = result.merge(bootstrap,on=["enrollment","subgroup","pseudo_dose"],validate="one_to_one")
    np.testing.assert_allclose(result.theta,result.starting_theta,atol=1e-8,rtol=1e-8)
    result.to_csv(HERE/"multiwindow-pipeline-propagation.csv",index=False)
    print(result.round(3).to_string(index=False))


if __name__=="__main__":
    main()

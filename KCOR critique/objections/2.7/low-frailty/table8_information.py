"""Local first-window information for the actual Table 8 joint seed fits."""
import io
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/"1.3"))
import table8_bootstrap as table

GROUPS = ("two-dose","unvaccinated","all-persons")


def local_information(model, k, theta, gamma, time, risk):
    start = np.expm1(gamma*time)/gamma
    increment = np.exp(gamma*time)*model._gompertz_week_factor(gamma)

    def hazard(log_rate, variance):
        scale = np.exp(log_rate)
        return model.weekly_gamma_hazard(scale*start,scale*increment,variance)

    log_k = np.log(k)
    eps = 1e-4
    h = hazard(log_k,theta)
    p = -np.expm1(-h)
    derivative_k = (hazard(log_k+eps,theta)-hazard(log_k-eps,theta))/(2*eps)
    derivative_theta = (hazard(log_k,theta+eps)-hazard(log_k,theta-eps))/(2*eps)
    jac = np.column_stack((derivative_k,derivative_theta))
    info = jac.T @ (jac*(risk*(1-p)/p)[:,None])
    cov = np.linalg.inv(info)
    return float(np.sqrt(cov[1,1])), float(np.sqrt(1/info[1,1])), float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]))


def main():
    book = subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB], cwd=HERE)
    populations = [table.read_population(io.BytesIO(book),sheet) for sheet in table.SHEETS]
    table.initialize(populations,"entry-moment")
    model = table.K
    gamma = table.CONFIG["gamma_per_week"]
    result = []
    for enrollment,(dates,time,quiet,keep,counts) in enumerate(table.DATA):
        counts = counts.copy()
        counts[2] += counts[0]+counts[1]
        alive = np.cumsum(counts[:,:,::-1],axis=2)[:,:,::-1][:,:,:-1]
        dead = counts[:,:,:-1]
        rebased = time-model.DYNAMIC_HVE_SKIP_WEEKS
        indices = np.flatnonzero(quiet & (rebased>=0))
        gaps = np.flatnonzero(np.diff(indices)>1)
        first = indices[:gaps[0]+1] if len(gaps) else indices
        for a,dose in np.ndindex((3,3)):
            hazard = model.hazard_from_mr(np.clip(dead[a,dose]/(alive[a,dose]+model.EPS),0,.999))
            effective = np.where(rebased>=0,hazard,0.)
            (k,applied),diag = table.compare.joint_fit(effective,rebased,quiet,4,gamma,deaths_arr=dead[a,dose])
            seed = diag["theta0_init"]
            se_joint,se_known,corr = local_information(model,k,seed,gamma,rebased[first],alive[a,dose,first])
            result.append(dict(enrollment=table.SHEETS[enrollment],subgroup=GROUPS[a],pseudo_dose=dose,
                first_week=str(dates[first[0]])[:10],last_week=str(dates[first[-1]])[:10],
                weeks=len(first),people=int(alive[a,dose,first[0]]),first_window_deaths=int(dead[a,dose,first].sum()),
                rate=k,seed_theta=seed,final_theta_raw=diag["theta0_raw"],applied_theta=applied,
                local_theta_se=se_joint,local_theta_se_known_rate=se_known,log_rate_theta_correlation=corr,
                status=diag["status"]))
    df = pd.DataFrame(result)
    df.to_csv(HERE/"table8-first-window-information.csv",index=False)
    print(df[["enrollment","subgroup","pseudo_dose","weeks","first_window_deaths","seed_theta",
              "final_theta_raw","applied_theta","local_theta_se","log_rate_theta_correlation"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

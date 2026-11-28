"""Trace the extreme fitted frailty in the archived coverage-stress dataset."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "1.3/multiwindow-coverage"))
import study
import fallback_stress

REP = 34


def main():
    fallback_stress.initialize()
    model = study.model
    batch, truth = study.samples("nonlinear_wave", REP, fallback_stress.DRAWS)
    weights = study.pooling_weights(batch)
    current = np.array(study.estimate(batch[0],weights,"nonlinear_wave"))
    with np.load(study.HERE/"fallback-stress-draws.npz") as f:
        archived = f["points"][REP]
        np.testing.assert_allclose(current, archived[0], rtol=1e-10, atol=1e-12)
        lo,hi = np.quantile(archived[1:,1,0],[.025,.975])
    dead, alive = study.arrays(batch[0])
    age, dose = 0, 1
    h = -np.log1p(-dead[age,dose]/(alive[age,dose]+model.EPS))
    h[:study.skip] = 0
    time = study.TIME-study.skip
    gamma = study.config["gamma_per_week"]
    unit_start = np.expm1(gamma*np.maximum(time,0))/gamma
    unit_week = np.exp(gamma*np.maximum(time,0))*model._gompertz_week_factor(gamma)
    observed = np.cumsum(h)
    first = study.quiet & (study.TIME <= study.WINDOWS[0][1])
    traces = {}
    for label,fit in (("fixed",study.compare.production_fit),("joint",study.compare.joint_fit)):
        iterations = []
        original_ls = model.least_squares

        def logged(*args,**kwargs):
            result = original_ls(*args,**kwargs)
            iterations.append(dict(parameters=result.x.tolist(), residual_sse=float(2*result.cost),
                                   success=bool(result.success)))
            return result
        with patch.object(model,"least_squares",logged):
            (k,theta),diag = fit(h,time,study.quiet,4,gamma,deaths_arr=dead[age,dose])
        reentry = study.WINDOWS[1][0]
        for idx,event in enumerate(iterations):
            th = event["parameters"][-1]
            event["iteration"] = idx
            event["gap_from_this_theta"] = float(model.invert_gamma_frailty(
                np.array([observed[reentry-1]]),th)[0] - k*unit_start[reentry])
        traces[label] = dict(k=float(k),theta_seed=diag["theta0_init"],theta_final=float(theta),
                            status=diag["status"],gap_raw=diag["delta_raw_values"],iterations=iterations)

    true_k, true_theta = study.RATES[age], study.THETA[age,dose]
    profile = []
    for th in (0,.5,4,10,25,50,75,100):
        def objective(log_k):
            k = np.exp(log_k)
            residual = h[first]-model.weekly_gamma_hazard(k*unit_start[first],k*unit_week[first],th)
            return np.dot(residual,residual)
        solution = minimize_scalar(objective, bounds=(np.log(1e-8),np.log(.1)),
                                   method="bounded",options={"xatol":1e-13})
        profile.append(dict(theta=th,profile_rate=np.exp(solution.x),sse=solution.fun))

    k,theta = traces["joint"]["k"],traces["joint"]["theta_final"]
    predicted_true = model.weekly_gamma_hazard(true_k*unit_start[first],true_k*unit_week[first],true_theta)
    predicted_seed = model.weekly_gamma_hazard(k*unit_start[first],k*unit_week[first],traces["joint"]["theta_seed"])
    weekly = pd.DataFrame(dict(modeled_week=time[first]+1,deaths=dead[age,dose,first],
        alive=alive[age,dose,first],expected_deaths_true=alive[age,dose,first]*-np.expm1(-predicted_true),
        expected_deaths_seed=alive[age,dose,first]*-np.expm1(-predicted_seed),
        observed_hazard=h[first],true_hazard=predicted_true,seed_hazard=predicted_seed))
    weekly.to_csv(HERE/"stress-first-window.csv",index=False)
    pd.DataFrame(profile).to_csv(HERE/"stress-seed-profile.csv",index=False)

    def growth(variance,rate, observed=observed):
        inverse = model.invert_gamma_frailty(observed,variance)
        curve,_ = model.apply_nph_correction_post_inversion(inverse,study.TIME,study.DATES.to_numpy(),
                variance,1.18,study.DATES[study.WAVE[0]],study.DATES[study.WAVE[1]],rate,gamma,
                observed_cumhaz=observed)
        return dict(growth=float(curve[-1]/curve[study.anchor]),
                    ordinary_inverted_endpoint=float(inverse[-1]),corrected_endpoint=float(curve[-1]))
    modeled = growth(theta,k)
    replacements = {"fitted":modeled,"true_theta":growth(true_theta,k),
                    "true_theta_and_rate":growth(true_theta,true_k)}
    for r in replacements.values():
        r["pooled_kcor"] = float(current[1,0]*np.exp(weights[age]*np.log(r["growth"]/modeled["growth"])))
    substitution_draws = {name: [] for name in replacements}
    for draw, counts in enumerate(batch):
        dead_draw, alive_draw = study.arrays(counts)
        h_draw = -np.log1p(-dead_draw[age,dose]/(alive_draw[age,dose]+model.EPS))
        h_draw[:study.skip] = 0.
        observed_draw = np.cumsum(h_draw)
        (rate, fitted_theta), diag = study.compare.joint_fit(
            h_draw,time,study.quiet,4,gamma,deaths_arr=dead_draw[age,dose])
        fitted_theta = fitted_theta if diag["success"] and np.isfinite(fitted_theta) and fitted_theta>=0 else 0.
        fitted_theta,_,_ = model._apply_theta_degenerate_guard(fitted_theta,study.guards)
        np.testing.assert_allclose(fitted_theta, archived[draw,1,7], atol=1e-10,rtol=1e-10)
        fitted_growth = growth(fitted_theta,rate,observed_draw)["growth"]
        for name, variance, baseline_rate in (("fitted",fitted_theta,rate),
              ("true_theta",true_theta,rate),("true_theta_and_rate",true_theta,true_k)):
            ratios = archived[draw,1,1:3].copy()
            ratios[age] *= growth(variance,baseline_rate,observed_draw)["growth"]/fitted_growth
            substitution_draws[name].append(float(model.safe_exp(np.average(model.safe_log(ratios),weights=weights))))
    np.testing.assert_allclose(substitution_draws["fitted"],archived[:,1,0],atol=1e-12,rtol=1e-10)
    for name, values in substitution_draws.items():
        np.testing.assert_allclose(values[0],replacements[name]["pooled_kcor"],rtol=1e-10)
        replacements[name]["pooled_interval"] = np.quantile(values[1:],[.025,.975]).tolist()
    result = dict(rep=REP, target=truth, observed_endpoint=float(observed[-1]),
                  true_k=float(true_k),true_theta=float(true_theta),
                  pooled_original=float(current[1,0]),pooled_interval=[float(lo),float(hi)],
                  weight=float(weights[age]),traces=traces,profile=profile,replacements=replacements)
    (HERE/"stress-trace.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(weekly.to_string(index=False))
    for label,r in traces.items():
        print(label, r["k"], r["theta_seed"],r["theta_final"],r["status"],r["gap_raw"])
    print(pd.DataFrame(profile).to_string(index=False))
    print(json.dumps(replacements,indent=2))


if __name__ == "__main__":
    main()

"""Isolate information about frailty from uncertainty in the fitted baseline rate."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "1.3/multiwindow-coverage"))
import study

SIZES = (50_000, 500_000, 5_000_000)
FRAILTIES = (0.5, 4.0)
WINDOW_LENGTHS = (4, 13, 52, 104)
REPETITIONS = 200
RATE = 0.000456
GAMMA = .085/52
TIME = np.arange(max(WINDOW_LENGTHS), dtype=float)
START = np.expm1(GAMMA*TIME)/GAMMA
WEEK = np.exp(GAMMA*TIME)*np.expm1(GAMMA)/GAMMA


def initialize():
    study.initialize()


def model_hazard(k, theta, length):
    return study.model.weekly_gamma_hazard(k*START[:length], k*WEEK[:length], theta)


def probabilities(theta):
    h = model_hazard(RATE, theta, len(TIME))
    survival = np.exp(-np.cumsum(h))
    return np.r_[-np.diff(np.r_[1., survival]), survival[-1]]


def oracle_fit(h, length):
    """Diagnostic fit with the actual baseline rate supplied."""
    fit = least_squares(lambda x: h[:length] - model_hazard(RATE, float(x[0]), length),
                       [1.], bounds=([0.], [100.]), method="trf",
                       ftol=1e-14, xtol=1e-14, gtol=1e-14)
    return float(fit.x[0])


def estimate(size, theta, rep, counts):
    alive = np.cumsum(counts[::-1])[::-1][:-1]
    dead = counts[:-1]
    h = -np.log1p(-dead/alive)
    rows = []
    for length in WINDOW_LENGTHS:
        quiet = TIME < length
        for label, fit in (("fixed", study.compare.production_fit), ("joint", study.compare.joint_fit)):
            (k, applied), diag = fit(h, TIME, quiet, 4, GAMMA, deaths_arr=dead)
            rows.append(dict(size=size, theta_true=theta, rep=rep, weeks=length, fitter=label,
                             rate=k, theta_raw=diag["theta0_raw"], theta_applied=applied,
                             theta_seed=diag["theta0_init"], success=diag["success"], status=diag["status"]))
        oracle = oracle_fit(h, length)
        rows.append(dict(size=size, theta_true=theta, rep=rep, weeks=length, fitter="known_rate",
                         rate=RATE, theta_raw=oracle, theta_applied=oracle,
                         theta_seed=oracle, success=True, status="diagnostic"))
    return rows


def run_batch(task):
    size, theta, start, end = task
    probs = probabilities(theta)
    records = []
    for rep in range(start, end):
        rng = np.random.default_rng([20261127, SIZES.index(size), FRAILTIES.index(theta), rep])
        counts = rng.multinomial(size, probs)
        records.extend(estimate(size, theta, rep, counts))
    return records


def information(size, theta, length):
    h = model_hazard(RATE, theta, length)
    p = -np.expm1(-h)
    risk = size*np.exp(-np.r_[0., np.cumsum(h)[:-1]])
    eps = 1e-4
    k_derivative = (model_hazard(RATE*np.exp(eps), theta, length) -
                    model_hazard(RATE*np.exp(-eps), theta, length))/(2*eps)
    theta_derivative = (model_hazard(RATE, theta+eps, length) -
                        model_hazard(RATE, theta-eps, length))/(2*eps)
    jac = np.column_stack((k_derivative, theta_derivative))
    # Conditional binomial death likelihood has information
    # E[Y] (dp/dx)(dp/dx)' / (p(1-p)), and dp/dx=(1-p) dh/dx.
    matrix = jac.T @ (jac * (risk*(1-p)/p)[:,None])
    covariance = np.linalg.inv(matrix)
    return dict(size=size, theta_true=theta, weeks=length,
                expected_deaths=float(np.dot(risk,p)),
                first_expected_deaths=float(risk[0]*p[0]),
                last_expected_deaths=float(risk[-1]*p[-1]),
                relative_hazard_change=float(h[-1]/h[0]-1),
                depletion_multiplier_change=float((h[-1]/WEEK[length-1])/(h[0]/WEEK[0])-1),
                expected_weekly_count_cv=float(1/np.sqrt(np.mean(risk*p))),
                theta_se_joint=float(np.sqrt(covariance[1,1])),
                theta_se_known_rate=float(1/np.sqrt(matrix[1,1])),
                log_rate_theta_correlation=float(covariance[0,1]/np.sqrt(covariance[0,0]*covariance[1,1])))


def summarize(rows):
    output = []
    for (size, theta, weeks, fitter), group in rows.groupby(["size", "theta_true", "weeks", "fitter"]):
        x = group.theta_raw.dropna().to_numpy()
        if len(x) == 0:
            raise ValueError("All estimates missing")
        mean = x.mean()
        margin = t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
        low, high = np.quantile(x,[.025,.975])
        output.append(dict(size=size,theta_true=theta,weeks=weeks,fitter=fitter,runs=len(group),
             finite=len(x),mean_theta=mean,mean_theta_lower=mean-margin,mean_theta_upper=mean+margin,
             median_theta=np.median(x),theta_q025=low,theta_q975=high,
             rmse_theta=np.sqrt(np.mean((x-theta)**2)),near_zero=int((x<1e-6).sum()),
             at_upper_bound=int((x>=99.9).sum()),applied_mean=group.theta_applied.mean(),
             mean_rate=group.rate.mean(),rate_theta_correlation=group[["rate","theta_raw"]].corr().iloc[0,1]))
    return pd.DataFrame(output)


def main():
    initialize()
    theoretical = [information(size, theta, length) for size in SIZES for theta in FRAILTIES
                   for length in WINDOW_LENGTHS]
    pd.DataFrame(theoretical).to_csv(HERE/"information.csv", index=False)
    expected = pd.DataFrame(row for size in SIZES for theta in FRAILTIES
                            for row in estimate(size,theta,-1,size*probabilities(theta)))
    expected.to_csv(HERE/"noiseless.csv", index=False)
    tasks = [(size,theta,start,start+25) for size in SIZES for theta in FRAILTIES
             for start in range(0,REPETITIONS,25)]
    rows = []
    with ProcessPoolExecutor(7, initializer=initialize) as pool:
        for n, batch in enumerate(pool.map(run_batch,tasks), 1):
            rows.extend(batch)
            if n % 8 == 0:
                print(f"{n}/{len(tasks)} simulation batches", flush=True)
    samples = pd.DataFrame(rows)
    samples.to_csv(HERE/"fit-estimates.csv",index=False)
    summary = summarize(samples)
    summary.to_csv(HERE/"fit-summary.csv",index=False)
    print(summary[["size","theta_true","weeks","fitter","mean_theta","median_theta","near_zero","at_upper_bound"]].to_string(index=False))


if __name__ == "__main__":
    main()

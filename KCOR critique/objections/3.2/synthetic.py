"""Known-truth controls for the isolated 3.2 comparison, on a post-skip time axis."""
from functools import lru_cache
import json
import math

from experiment import HERE, entry_fit, np, table
from scipy.integrate import quad_vec
from scipy.stats import t as student

GAMMA = .085/52
WEEK = np.arange(75)
DATE = np.datetime64("2022-02-21") + WEEK*np.timedelta64(7,"D")
ANCHOR = 3
PULSES = ((5,.010),(8,.015),(11,.025))
EARLY = ((1,.010),(4,.015),(8,.010),(11,.015))
# The last two pulse weeks in the partial-window arm are outside the supplied wave window.
CASES = [
    ("no wave", 1., (), 12, 40, 11, 1.),
    ("published single shock", 1., ((8,.05),), 12, 40, 11, 1.),
    ("several shocks", 1., PULSES, 12, 40, 11, 1.),
    ("nonlinear, all pulses in window", 1.18, PULSES, 12, 40, 11, 1.),
    ("nonlinear, short first fit", 1.18, EARLY, 14, 17, 11, 1.),
    ("nonlinear, partial wave window", 1.18, EARLY, 14, 17, 4, 1.),
    ("nonlinear, partial window, unequal rates", 1.18, EARLY, 14, 17, 4, 1.25),
]


@lru_cache(None)
def population(theta,k,alpha,pulses):
    baseline = k*np.expm1(GAMMA*(WEEK+1))/GAMMA
    wave = np.zeros(len(WEEK))
    for week, amount in pulses:
        wave[week:] += amount
    if alpha == 1:
        survival = np.exp(-np.log1p(theta*(baseline+wave))/theta)
    else:
        # Integrate the individual survival under Z~Gamma(1/theta, theta).
        # The substitution Z=theta*u**theta removes the singularity at zero.
        def integrand(u):
            x = u**theta
            z = theta*x
            return np.exp(-x-z*baseline-z**alpha*wave)
        integral, error = quad_vec(integrand, 0., np.inf, epsabs=1e-12, epsrel=1e-12)
        survival = integral * theta/math.gamma(1/theta)
        assert error < 1e-9
    hazard = -np.diff(np.log(np.r_[1.,survival]))
    assert np.all(hazard>0)
    return hazard, baseline, wave, survival


def sample(survival, size, rng):
    probabilities = np.r_[-np.diff(np.r_[1.,survival]),survival[-1]]
    cells = rng.multinomial(size, probabilities)
    alive = np.cumsum(cells[::-1])[::-1][:-1]
    return -np.log1p(-cells[:-1]/alive), cells[:-1]


def run_case(case, fitters, model, rng=None):
    name,alpha,pulses,start,end,wave_end,rate_ratio = case
    quiet = (WEEK>=start)&(WEEK<=end)
    if end==17:
        quiet |= (WEEK>=50)&(WEEK<=70)
    output = [[],[]]
    truth = []
    for cohort,(theta,k) in enumerate(((4.,.004*rate_ratio),(.5,.004))):
        hazard, baseline, wave, survival = population(theta,k,alpha,pulses)
        truth.append((baseline[-1]+wave[-1])/(baseline[ANCHOR]+wave[ANCHOR]))
        counts = None
        if rng is not None:
            hazard, counts = sample(survival,50000,rng)
        observed = np.cumsum(hazard)
        for arm, fit in enumerate(fitters):
            (rate,frailty),diag = fit(hazard,WEEK,quiet,4,GAMMA,deaths_arr=counts)
            assert np.isfinite(rate)
            if not diag["success"]:
                frailty = 0.
            applied,_,_ = model._apply_theta_degenerate_guard(frailty,model.get_theta_degenerate_fit_config())
            keep = (WEEK<=wave_end)|(WEEK==ANCHOR)|(WEEK==WEEK[-1])
            curve,wd = model.apply_nph_correction_post_inversion(
                model.invert_gamma_frailty(observed[keep],applied),WEEK[keep],DATE[keep],applied,
                alpha,DATE[0],DATE[wave_end],rate,GAMMA,t_rebased=WEEK[keep],observed_cumhaz=observed[keep])
            output[arm].append(dict(k=rate,theta=applied,seed_theta=diag["theta0_init"],true_k=k,
                true_theta=theta,baseline_at_entry=rate*np.expm1(GAMMA*start)/GAMMA,
                true_baseline_at_entry=baseline[start-1],true_wave_at_exit=wave[wave_end],
                wave_at_exit=wd["post_wave_exposure"] if alpha!=1. else None,
                status=diag["status"], success=bool(diag["success"]),growth=curve[-1]/curve[ANCHOR]))
    target=truth[0]/truth[1]
    ratios = [arm[0]["growth"]/arm[1]["growth"] for arm in output]
    return dict(case=name,target=target,ratio=ratios,cohorts=output)


def main():
    table.compare.initialize("gap-refit")
    model = table.compare.adapter.K
    fits = (table.compare.joint_fit,entry_fit(model))
    results = [run_case(case,fits,model) for case in CASES]
    for r in results[:3]:
        np.testing.assert_allclose(r["ratio"][1],r["target"],atol=1e-6)
        for c in r["cohorts"][1]:
            np.testing.assert_allclose([c["k"],c["theta"]],[c["true_k"],c["true_theta"]],rtol=1e-5)
    noisy=[]
    sampled = {}
    for i in (2,5,6):
        draws=[]
        parameters=[]
        for rep in range(200):
            rng = np.random.default_rng([32022,i,rep])
            r=run_case(CASES[i],fits,model,rng)
            draws.append(r["ratio"])
            parameters.append([[[c["k"],c["theta"],c["growth"]] for c in arm] for arm in r["cohorts"]])
        arr=np.array(draws)
        sampled[f"case_{i}_ratio"]=arr
        sampled[f"case_{i}_k_theta_growth"]=np.asarray(parameters)
        target=r["target"]
        errors=np.abs(arr-target)
        delta=errors[:,1]-errors[:,0]
        half=student.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
        noisy.append(dict(case=CASES[i][0],repetitions=len(arr),people_per_cohort=50000,target=target,
            mean=arr.mean(axis=0).tolist(),median=np.median(arr,axis=0).tolist(),
            central95=np.quantile(arr,[.025,.975],axis=0).T.tolist(),
            median_absolute_error=np.median(errors,axis=0).tolist(),
            improved_tied_worse=[int((delta<0).sum()),int((delta==0).sum()),int((delta>0).sum())],
            mean_absolute_error=errors.mean(axis=0).tolist(),paired_change_in_mean_absolute_error=delta.mean(),
            paired_mc95=[delta.mean()-half,delta.mean()+half]))
    np.savez_compressed(HERE/"synthetic-draws.npz", **sampled)
    (HERE/"synthetic-results.json").write_text(json.dumps(dict(noiseless=results,sampled=noisy),indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()

"""Diagnostic on the saved 200 short-window nonlinear nulls and 999 paired Table 8 draws."""
import json
from scipy.optimize import least_squares
from synthetic import ANCHOR, CASES, DATE, GAMMA, HERE, WEEK, np, population, sample, table


def growth(model, observed, rate, theta, wave_end, alpha):
    keep = (WEEK <= wave_end) | (WEEK == ANCHOR) | (WEEK == WEEK[-1])
    curve, _ = model.apply_nph_correction_post_inversion(
        model.invert_gamma_frailty(observed[keep], theta), WEEK[keep], DATE[keep],
        theta, alpha, DATE[0], DATE[wave_end], rate, GAMMA,
        t_rebased=WEEK[keep], observed_cumhaz=observed[keep])
    return curve[-1] / curve[ANCHOR]


def compare_frailties():
    case = 5
    _, alpha, pulses, first, last, wave_end, _ = CASES[case]
    table.compare.initialize("gap-refit")
    model = table.compare.adapter.K
    with np.load(HERE/"synthetic-draws.npz") as archive:
        fitted = archive[f"case_{case}_k_theta_growth"]
        original = archive[f"case_{case}_ratio"]
    known = np.empty_like(fitted[:,:,:,2])
    start = np.expm1(GAMMA*WEEK)/GAMMA
    inc = np.exp(GAMMA*WEEK)*np.expm1(GAMMA)/GAMMA
    quiet = slice(first,last+1)
    for rep in range(len(fitted)):
        rng = np.random.default_rng([32022,case,rep])
        for cohort, theta in enumerate((4.,.5)):
            hazard, _ = sample(population(theta,.004,alpha,pulses)[3], 50000, rng)
            observed = hazard.cumsum()
            def residual(lograte):
                k = np.exp(lograte[0])
                return hazard[quiet]-model.weekly_gamma_hazard(k*start[quiet], k*inc[quiet], theta)
            seed = least_squares(residual, [np.log(np.mean(hazard[quiet]/inc[quiet]))],
                                 bounds=(np.log(1e-8),np.log(.1)), ftol=1e-14,xtol=1e-14,gtol=1e-14)
            assert seed.success
            r = np.exp(seed.x[0])
            entry = theta*observed[first-1]-np.log1p(theta*r*start[first])
            for arm, rate in enumerate((r,min(.1,r*np.exp(max(0.,entry))))):
                k, th, archived_growth = fitted[rep,arm,cohort]
                np.testing.assert_allclose(growth(model,observed,k,th,wave_end,alpha),
                                           archived_growth, rtol=1e-10)
                known[rep,arm,cohort] = growth(model,observed,rate,theta,wave_end,alpha)
    result = {}
    for label, numerator, denominator in (
        ("both estimated", fitted[:,:,0,2], fitted[:,:,1,2]),
        ("low variance supplied", fitted[:,:,0,2], known[:,:,1]),
        ("higher variance supplied", known[:,:,0], fitted[:,:,1,2]),
        ("both supplied", known[:,:,0], known[:,:,1])):
        ratio = numerator/denominator
        if label == "both estimated":
            np.testing.assert_allclose(ratio, original)
        e = abs(ratio-1.)
        result[label] = dict(median_absolute_error=np.median(e,axis=0).tolist(),
                             median=np.median(ratio,axis=0).tolist(),
                             central95=np.quantile(ratio,[.025,.975],axis=0).T.tolist(),
                             trial_worse=int(sum(e[:,1]>e[:,0]+1e-12)),
                             trial_better=int(sum(e[:,1]<e[:,0]-1e-12)))
    result["applied_theta_quantiles"] = np.quantile(fitted[:,:,:,1],[.025,.5,.975],axis=0).tolist()
    return result


def width_precision():
    with np.load(HERE.parent/"3.1/table8-draws.npz") as archive:
        current = archive["points"][list(archive["fitters"]).index("corrected_joint"),1,1:]
    with np.load(HERE/"entry-draws.npz") as archive:
        trial = archive["proposed"][1,1:]
    pairs = np.stack((current,trial),axis=-1)
    rng = np.random.default_rng(320221)
    delta = []
    for _ in range(100):
        indices = rng.integers(0,len(pairs),size=(100,len(pairs)))
        limits = np.quantile(pairs[indices],[.025,.975],axis=1)
        widths = limits[1]-limits[0]
        delta.extend(widths[:,:,1]-widths[:,:,0])
    return np.quantile(delta,[.025,.975],axis=0).T.tolist()


if __name__ == "__main__":
    results = dict(null=compare_frailties(), paired_width_change_mc95=width_precision())
    (HERE/"variance-check.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf8")

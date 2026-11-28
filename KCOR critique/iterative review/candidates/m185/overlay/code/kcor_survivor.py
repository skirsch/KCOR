"""Weekly inversion for the optional NPH module.

Use the existing KCOR reference rate, wave dates and post-wave continuation.
If a week cannot support a positive wave increment, retain prior wave exposure
and recover that week's baseline increment from conditional survival.
"""
from functools import lru_cache

import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaincinv, logsumexp, roots_legendre


@lru_cache(maxsize=None)
def _probability_rule(order):
    nodes, weights = roots_legendre(order)
    edges = np.array([0.0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, .001, .01, .1, .5])
    half_width = np.diff(edges)[:, None] / 2
    probs = (edges[:-1, None] + half_width * (nodes + 1)).ravel()
    weights = (half_width * weights).ravel()
    return np.r_[probs, 1 - probs], np.tile(weights, 2)


@lru_cache(maxsize=256)
def gamma_quadrature(theta, order=32):
    # Probability-space rule with extra resolution in both tails.
    probs, weights = _probability_rule(order)
    z = theta * gammaincinv(1 / theta, probs)
    return z, weights


# All three stages (fit, weekly correction, post-wave inversion) use this
# positive-weight integral. Increasing wave exposure decreases survival at
# every quadrature node. Order refinement is accepted only after convergence.
SURVIVAL_ATOL = 2e-13
SURVIVAL_RTOL = 2e-12
SURVIVAL_ORDERS = (32, 64, 128, 256, 512)

@lru_cache(maxsize=256)
def _wave_nodes(theta, alpha, order):
    z, weights = gamma_quadrature(theta, order)
    return np.power(z, alpha), weights


def _wave_log_at_order(u, theta, alpha, order):
    za, weights = _wave_nodes(theta, alpha, order)
    powers = u[:, None] * za
    lost = (-np.expm1(-powers)) @ weights
    result = np.empty_like(lost)
    small = lost < .5
    result[small] = np.log1p(-lost[small])
    result[~small] = logsumexp(-powers[~small], b=weights, axis=1)
    return result


def _wave_log_scalar(u, theta, alpha):
    if u == 0.:
        return 0.
    prev = None
    for order in SURVIVAL_ORDERS:
        za, weights = _wave_nodes(theta, alpha, order)
        powers = u * za
        lost = float(np.dot(weights, -np.expm1(-powers)))
        cur = math.log1p(-lost) if lost < .5 else float(logsumexp(-powers, b=weights))
        if prev is not None and math.isfinite(cur) and math.isfinite(prev) and abs(cur-prev) <= SURVIVAL_ATOL + SURVIVAL_RTOL*abs(cur):
            return cur
        prev = cur
    raise ArithmeticError("Gamma survival quadrature did not converge")


def wave_log_survival(baseline, wave, theta, alpha):
    """Log of the remaining wave term under baseline-tilted gamma frailty."""
    if np.isscalar(baseline) and np.isscalar(wave):
        if theta == 0:
            return -float(wave)
        if alpha == 1:
            return -math.log1p(theta*wave/(1+theta*baseline))/theta
        return _wave_log_scalar(wave/(1+theta*baseline)**alpha, theta, alpha)
    B, W = np.broadcast_arrays(np.asarray(baseline, float), np.asarray(wave, float))
    if theta == 0:
        out = -W
    elif alpha == 1:
        out = -np.log1p(theta * W / (1 + theta * B)) / theta
    else:
        shape = B.shape
        u = (W / np.power(1 + theta * B, alpha)).ravel()
        out = np.zeros_like(u)
        pending = np.flatnonzero(u != 0.)
        if len(pending):
            prev = _wave_log_at_order(u[pending], theta, alpha, SURVIVAL_ORDERS[0])
            for order in SURVIVAL_ORDERS[1:]:
                cur = _wave_log_at_order(u[pending], theta, alpha, order)
                good = np.isfinite(cur) & np.isfinite(prev) & (
                    np.abs(cur - prev) <= SURVIVAL_ATOL + SURVIVAL_RTOL * np.abs(cur))
                out[pending[good]] = cur[good]
                pending = pending[~good]
                if not len(pending):
                    break
                prev = cur[~good]
            if len(pending):
                raise ArithmeticError("Gamma survival quadrature did not converge")
        out = out.reshape(shape)
    return float(out) if out.ndim == 0 else out


def cumulative_hazard(baseline, wave, theta, alpha):
    """Negative log survival under the starting gamma law, scalar or array."""
    if np.isscalar(baseline) and np.isscalar(wave):
        if theta == 0:
            return float(baseline + wave)
        if alpha == 1:
            return math.log1p(theta*(baseline+wave))/theta
        return math.log1p(theta*baseline)/theta - wave_log_survival(baseline, wave, theta, alpha)
    B, W = np.broadcast_arrays(np.asarray(baseline, float), np.asarray(wave, float))
    if theta == 0:
        out = B + W
    elif alpha == 1:
        out = np.log1p(theta * (B + W)) / theta
    else:
        out = np.log1p(theta * B) / theta - wave_log_survival(B, W, theta, alpha)
    return float(out) if out.ndim == 0 else out


def solve_increase(hazard, target, initial_upper):
    lower = hazard(0.0)
    if target <= lower:
        return 0.0
    upper = max(initial_upper, 1e-10)
    while hazard(upper) < target:
        upper *= 2
        if not np.isfinite(upper):
            raise ArithmeticError("Could not bracket the weekly exposure")
    return brentq(lambda x: hazard(x) - target, 0, upper, xtol=1e-14)


def make_correction(model):
    original = model.apply_nph_correction_post_inversion

    def correct(H0, t_vals, dates, theta_applied, alpha, start_dt, end_dt,
                k_hat, gamma_per_week, t_rebased=None, observed_cumhaz=None):
        theta = float(theta_applied)
        H0 = np.asarray(H0, dtype=float)
        t_r = np.asarray(t_vals, dtype=float) - model.DYNAMIC_HVE_SKIP_WEEKS if t_rebased is None else np.asarray(t_rebased)
        dates = np.asarray(dates, dtype="datetime64[ns]")
        mask = (dates >= np.datetime64(start_dt)) & (dates <= np.datetime64(end_dt)) & (t_r >= 0)
        indices = np.flatnonzero(mask)
        identity = abs(alpha - 1) <= model.NPH_IDENTITY_TOL or (0 <= theta < model.KCOR6_THETA_EPS)
        valid_params = (np.isfinite(theta) and theta >= 0 and np.isfinite(k_hat)
                        and k_hat > 0 and np.isfinite(gamma_per_week))
        if identity:
            # Proportional exposure is exactly additive when alpha=1 or theta=0
            # (using the existing zero-frailty / alpha-one numerical conventions).
            # Attribute each week's increment independently; never modify H0.
            diag = dict(correction_mode="identity", wave_points=len(indices),
                        deficit_weeks=np.nan, skipped_nonpositive_excess=np.nan,
                        corrected_points=0, max_abs_change=0., post_wave_level_shift=0.,
                        post_wave_exposure=np.nan, any_nonmonotone_fixed=False,
                        f_min=np.nan, f_max=np.nan)
            if not len(indices):
                diag.update(deficit_weeks=0, skipped_nonpositive_excess=0,
                            post_wave_exposure=0.)
                return H0.copy(), diag
            if not valid_params:
                return H0.copy(), diag
            effective_theta = 0. if theta < model.KCOR6_THETA_EPS else theta
            opening = np.r_[0., H0[:-1]][indices]
            total = H0[indices] - opening
            planned = k_hat * model.safe_exp(gamma_per_week * t_r[indices]) * model._gompertz_week_factor(gamma_per_week)
            if not (np.isfinite(opening).all() and np.isfinite(total).all()
                    and np.isfinite(planned).all() and (opening>=0).all()
                    and (total>=0).all() and (planned>=0).all()):
                return H0.copy(), diag
            if effective_theta == 0:
                observed_increment, planned_increment = total, planned
            else:
                # Compare strict deficits on observed integrated-hazard scale,
                # using the same opening survivor distribution for both increments.
                den = 1. + effective_theta * opening
                observed_increment = np.log1p(effective_theta * total / den) / effective_theta
                planned_increment = np.log1p(effective_theta * planned / den) / effective_theta
            if observed_cumhaz is not None:
                observed_increment = np.diff(np.asarray(observed_cumhaz,float),prepend=0.)[indices]
            if not (np.isfinite(observed_increment).all() and np.isfinite(planned_increment).all()):
                return H0.copy(), diag
            roundoff = 8*np.finfo(float).eps*np.maximum(np.abs(opening)+np.abs(total),np.abs(planned))
            wave_add = np.where(total-planned > roundoff, total-planned, 0.)
            base_add = total - wave_add
            wave_path = np.cumsum(wave_add)
            base_path = opening[0] + np.cumsum(base_add)
            diag.update(deficit_weeks=int(np.count_nonzero(observed_increment < planned_increment - 1e-12)),
                        skipped_nonpositive_excess=int(np.count_nonzero(wave_add == 0.)),
                        corrected_points=int(np.count_nonzero(wave_add > 0.)),
                        wave_exposure_path=wave_path.tolist(), baseline_exposure_path=base_path.tolist(),
                        post_wave_exposure=float(wave_path[-1]), post_wave_baseline=float(base_path[-1]))
            return H0.copy(), diag
        if not valid_params or not len(indices):
            output, diag = original(H0, t_vals, dates, theta, alpha, start_dt, end_dt, k_hat,
                                    gamma_per_week, t_rebased=t_r, observed_cumhaz=observed_cumhaz)
            diag.update(wave_points=len(indices), deficit_weeks=np.nan if len(indices) else 0,
                        skipped_nonpositive_excess=np.nan if len(indices) else 0)
            return output, diag

        observed = (np.log1p(theta * H0) / theta if observed_cumhaz is None
                    else np.asarray(observed_cumhaz, dtype=float))
        first, last = indices[0], indices[-1]
        baseline = float(H0[first - 1]) if first else 0.0
        wave = 0.0
        entry_loss = cumulative_hazard(baseline, wave, theta, alpha)
        entry_observed = observed[first - 1] if first else 0.0
        baseline_path = k_hat * model.safe_exp(gamma_per_week * t_r) * model._gompertz_week_factor(gamma_per_week)
        output = H0.copy()
        baseline_values, wave_values = [], []
        deficits = 0
        max_residual = 0.0
        for i in indices:
            target = entry_loss + observed[i] - entry_observed
            planned = baseline + baseline_path[i]
            planned_loss = cumulative_hazard(planned, wave, theta, alpha)
            if target > planned_loss:
                baseline = planned
                wave += solve_increase(
                    lambda x: cumulative_hazard(baseline, wave + x, theta, alpha),
                    target, target - planned_loss)
            else:
                deficits += int(target < planned_loss - 1e-12)
                baseline += solve_increase(
                    lambda x: cumulative_hazard(baseline + x, wave, theta, alpha),
                    target, baseline_path[i])
            output[i] = baseline + wave
            baseline_values.append(baseline)
            wave_values.append(wave)
            max_residual = max(max_residual, abs(cumulative_hazard(baseline, wave, theta, alpha) - target))

        if last + 1 < len(output):
            tail = slice(last + 1, None)
            increases = model.post_wave_baseline_increases(
                observed[tail] - observed[last], theta, alpha, baseline, wave)
            output[tail] = output[last] + increases

        new_wave = np.diff(wave_values, prepend=0)
        diag = dict(correction_mode="weekly_survival", wave_points=len(indices),
                    corrected_points=int(np.count_nonzero(new_wave > 0)),
                    skipped_nonpositive_excess=int(np.count_nonzero(new_wave == 0)),
                    deficit_weeks=deficits, max_survival_residual=max_residual,
                    wave_exposure_path=wave_values, baseline_exposure_path=baseline_values,
                    post_wave_exposure=wave, post_wave_baseline=baseline,
                    max_abs_change=float(np.max(np.abs(output - H0))),
                    post_wave_level_shift=float(output[last] - H0[last]),
                    any_nonmonotone_fixed=False, f_min=float("nan"), f_max=float("nan"))
        return output, diag

    return correct


from scipy.optimize import least_squares


class Survival:
    def __init__(self,theta,alpha):
        self.th=float(theta);self.a=float(alpha)
    def hazard(self,B,W):
        return cumulative_hazard(B,W,self.th,self.a)
    def increase(self,B,W,target,axis,guess):
        # Solve cumulative survival on the monotone indicated exposure axis.
        base=self.hazard(B,W)
        if target<=base+1e-15:return 0.
        def f(x):return self.hazard(B+x,W)-target if axis==0 else self.hazard(B,W+x)-target
        upper=max(guess,1e-10)
        while f(upper)<0:
            upper*=2
            if not np.isfinite(upper) or upper>1e150:raise ArithmeticError("exposure not bracketed")
        return brentq(f,0.,upper,xtol=1e-12,rtol=1e-13)

class Profile:
    def __init__(self,h,t,quiet,anchors,gamma,wave):
        self.idx=np.flatnonzero(np.isfinite(h)&np.isfinite(t)&(t>=0))
        self.h=h[self.idx];self.t=t[self.idx]
        self.obs=np.r_[0.,np.cumsum(np.maximum(self.h,0.))]
        self.F=np.expm1(gamma*np.r_[self.t,self.t[-1]+1])/gamma
        assert np.allclose(np.diff(self.t),1.)
        self.wave=(self.t>=wave[0])&(self.t<=wave[1])
        self.fit=(quiet[self.idx]|(self.t<anchors))&~self.wave
        self.segments=[]
        kind=np.where(self.wave,2,np.where(self.fit,0,1))
        bounds=np.r_[0,np.flatnonzero(np.diff(kind)!=0)+1,len(kind)]
        # Only data through final fitting week matter, unless wave is later.
        last=np.flatnonzero(self.fit)[-1]+1 if self.fit.any() else 0
        for a,b in zip(bounds[:-1],bounds[1:]):
            if a>=last:break
            self.segments.append((int(a),int(min(b,last)),int(kind[a])))
    def predict(self,k,theta,alpha,*,return_exposure_span=False):
        sv=Survival(theta,alpha); B=W=0.;pred=[];rawgap=[];applied=[];deficits=0
        exposure_first=exposure_last=None
        for a,b,kind in self.segments:
            increments=k*np.diff(self.F[a:b+1])
            if kind==0:
                before=B+np.r_[0.,np.cumsum(increments)]
                q=np.diff(sv.hazard(before,np.full(len(before),W)))
                if exposure_first is None: exposure_first=float(before[0]+W)
                exposure_last=float(before[-2]+W)
                pred.extend(q);B=before[-1]
            elif kind==1:
                # Reconstruct observed survival at quiet re-entry. As in
                # proportional KCOR, nonpositive additional gap exposure is zero.
                B+=float(increments.sum())
                target=self.obs[b];raw=target-sv.hazard(B,W)
                extra=sv.increase(B,W,target,0,max(raw,1e-10))
                rawgap.append(raw);applied.append(extra);B+=extra
            else:
                # At wave opening, the observed pre-wave survival implies an
                # exact proportional accumulated exposure (same as trial 1.2).
                # If the wave starts at modeled origin this is zero.
                B=np.expm1(theta*self.obs[a])/theta if theta else self.obs[a]
                W=0.
                # A sufficient no-deficit condition: with positive alpha, prior
                # wave selection cannot increase the mean surviving frailty.
                # If observations exceed the no-wave conditional baseline in
                # every week, every sequential W increment is nonnegative.
                # Then one exact solve at exit equals the weekly recursion.
                path=B+np.r_[0.,np.cumsum(increments)]
                baseline_only=np.diff(sv.hazard(path,np.zeros(len(path))))
                if np.all(self.h[a:b]>=baseline_only):
                    B=float(path[-1])
                    W=sv.increase(B,0.,self.obs[b],1,max(self.obs[b]-sv.hazard(B,0.),1e-10))
                    continue
                for j,inc in zip(range(a,b),increments):
                    planned=B+inc;target=self.obs[j+1]
                    loss=sv.hazard(planned,W)
                    if target>loss:
                        B=planned;W+=sv.increase(B,W,target,1,max(target-loss,1e-10))
                    else:
                        deficits+=int(target<loss-1e-12)
                        B+=sv.increase(B,W,target,0,inc)
        result=(np.array(pred),rawgap,applied,deficits)
        if return_exposure_span:
            span=exposure_last-exposure_first if exposure_first is not None else np.nan
            return (*result,span)
        return result

def make_fitter(model,wave,alpha=1.18):
    base=model._fit_all_window_gamma
    def fit(h_arr,t_rebased,quiet_mask,k_anchor_weeks,gamma_per_week,theta0_init=1.,deaths_arr=None,**kwargs):
        h=np.asarray(h_arr);t=np.asarray(t_rebased);quiet=np.asarray(quiet_mask,bool)
        # Identity for no nonlinear wave within the modeled fit history.
        if not np.any((t>=max(0,wave[0]))&(t<=wave[1])):
            return base(h,t,quiet,k_anchor_weeks,gamma_per_week,theta0_init,deaths_arr)
        # Keep the configured quiet-only gate separate from anchor-only objective rows.
        valid=np.isfinite(h)&np.isfinite(t)&(t>=0)
        marked=(t>=wave[0])&(t<=wave[1])
        quiet_eligible=quiet&valid&~marked
        eligible=valid&(quiet|(t<k_anchor_weeks))&~marked
        n_quiet=int(quiet_eligible.sum())
        n_fit=int(eligible.sum())
        # The proportional fit is only an initializer. Its quiet gate must use
        # the same quiet rows as the final fit (not the excluded wave rows).
        initial,diag=base(h,t,quiet_eligible,k_anchor_weeks,gamma_per_week,theta0_init,deaths_arr)
        # The proportional initializer's gap model is not the nonlinear model.
        # Do not present its provisional gap offsets as nonlinear diagnostics.
        diag["gap_first_pass"]=None
        diag["gap_seed_status"]="not_applicable"
        deaths=None if deaths_arr is None else np.asarray(deaths_arr,dtype=float)
        total=float(np.nansum(np.where(quiet_eligible,np.clip(deaths,0.,None),0.))) if deaths is not None and deaths.shape==h.shape else np.nan
        minimum=int(model.get_theta_degenerate_fit_config().get("min_quiet_deaths",30))
        diag.update(n_obs=n_fit,n_fit=n_fit,fit_mask_theta=eligible,
                    n_quiet_bins=n_quiet,total_quiet_deaths=total,min_quiet_deaths=minimum,
                    quiet_window_start=float(np.min(t[eligible])) if n_fit else np.nan,
                    quiet_window_end=float(np.max(t[eligible])) if n_fit else np.nan,
                    identifiability_flag=n_fit<6)
        if n_fit<3 or n_quiet<3:
            diag.update(success=False,status="insufficient_data",
                        message="nonlinear_fit_too_few_eligible_weeks",
                        k_hat=np.nan,theta0_raw=np.nan,anchor_only=False,
                        insufficient_signal=True,rmse_hazard=np.nan,
                        fit_residual_max=np.nan,relRMSE_hazard_fit=np.nan)
            return (np.nan,np.nan),diag
        if np.isfinite(total) and total<minimum:
            diag.update(insufficient_deaths=True,status="insufficient_deaths",
                        message=f"INSUFFICIENT_DEATHS: eligible quiet-window deaths {total:.1f} < min_quiet_deaths={minimum}; applied theta=0.0",
                        success=bool(np.isfinite(initial[0])),theta0_raw=0.,
                        rmse_hazard=np.nan,fit_residual_max=np.nan,relRMSE_hazard_fit=np.nan)
            return (initial[0],0.),diag
        if not np.isfinite(initial[0]):return initial,diag
        p=Profile(h,t,quiet_eligible,k_anchor_weeks,gamma_per_week,wave)
        target=p.h[p.fit]
        # Scale residuals for numerical optimization only (same LS minimum).
        scale=max(float(target.mean()),1e-8)
        def residual(x):
            k,th=np.exp(x[0]),x[1]
            try:
                pred,_,_,_=p.predict(k,th,alpha)
                e=(target-pred)/scale
                if np.isfinite(e).all():return e
            except (ArithmeticError,ValueError,OverflowError,FloatingPointError):
                pass
            return np.full(len(target),1e6)
        thmax=float(model.get_theta_degenerate_fit_config().get("theta0_max",500.))
        k0=float(np.clip(initial[0],1e-8,.1));th0=float(np.clip(initial[1],0.,thmax)) if np.isfinite(initial[1]) else min(.5,thmax)
        candidates=[]
        for th in (th0,min(.5,thmax)):
            if candidates and abs(th-th0)<1e-7:continue
            try:
                r=least_squares(residual,[np.log(k0),th],bounds=([np.log(1e-8),0.],[np.log(.1),thmax]),
                                  ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=200)
                candidates.append(r)
            except (ValueError, ArithmeticError):
                if not candidates:raise
        valid=[r for r in candidates if r.success and np.isfinite(r.cost)]
        r=min(valid or candidates,key=lambda r:r.cost)
        k,theta=float(np.exp(r.x[0])),float(r.x[1])
        pred,gaps,applied,deficits,span=p.predict(k,theta,alpha,return_exposure_span=True)
        e=target-pred;rel=float(np.sqrt(np.mean((e/(target+1e-12))**2)))
        guard=model.get_theta_degenerate_fit_config()
        bound=theta>=thmax*.999
        bad=rel>float(guard.get("bad_fit_relrmse_threshold",1e5))
        flag=bound
        weak=bool(n_fit<6 or (np.isfinite(span) and span<1e-4))
        mask=np.zeros_like(quiet);mask[p.idx]=p.fit
        diag["gap_first_pass"] = [] if r.success and not any(kind==1 for _,_,kind in p.segments) else None
        diag.update(success=bool(r.success),status=("theta_upper_bound" if flag else ("weak_identifiability" if weak else "ok")) if r.success else "error",message="nonlinear survivor fit",
                    n_obs=len(target),n_fit=len(target),fit_mask_theta=mask,k_hat=k,theta0_raw=theta,
                    theta_hit_bound=bound,bad_fit=False,legacy_relative_error_flag=bad,fitted_weekly_hazard=pred.copy(),insufficient_signal=False,identifiability_flag=weak,
                    rmse_hazard=float(np.sqrt(np.mean(e**2))),fit_residual_max=float(np.max(abs(e))),relRMSE_hazard_fit=rel,
                    delta_negative_clamped=any(x<=0 for x in gaps),delta_negative=any(x<=0 for x in gaps),
                    nonlinear_gap_observed_deficit=gaps,nonlinear_gap_exposure=applied,nonlinear_wave_deficits=deficits,
                    delta_raw_values=list(gaps),delta_applied_values=list(applied),delta_raw_scale="observed_log_survival",
                    trial_nfev=sum(x.nfev for x in candidates))
        return (k,theta),diag
    return fit


"""Exact discrete conditional gamma-frailty fit for one contiguous quiet window.

The parameters are an effective baseline rate at quiet-window entry and gamma
frailty variance. The likelihood conditions on each week's risk set, so prior
deaths (including in skipped weeks) are not modeled here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy


@dataclass(frozen=True)
class Fit:
    k_effective: float
    theta: float
    success: bool
    objective: float
    gradient_norm: float
    iterations: int
    near_zero: bool
    hit_upper_bound: bool


def weekly_hazard(k_effective: float, theta: float, shape: np.ndarray):
    """Return hazard and derivatives for log(k) and theta on contiguous weeks."""
    shape = np.asarray(shape, dtype=float)
    b = k_effective * shape
    a = k_effective * (np.cumsum(shape) - shape)
    den_before = 1.0 + theta * a
    den_after = 1.0 + theta * (a + b)
    dh_dlogk = b / (den_before * den_after)
    if theta < 1e-3:
        first = a*b + b*b/2.0
        second = a*a*b + a*b*b + b*b*b/3.0
        h = b - theta*first + theta*theta*second
        dh_dtheta = -first + 2.0*theta*second
    else:
        x = theta*b/den_before
        log_term = np.log1p(x)
        h = log_term/theta
        dx_dtheta = b/(den_before*den_before)
        dh_dtheta = (theta*dx_dtheta/(1.0+x)-log_term)/(theta*theta)
    return h, dh_dlogk, dh_dtheta


def _objective(x, dead, alive, shape):
    k = float(np.exp(x[0]))
    theta = float(x[1])
    h, dk, dtheta = weekly_hazard(k, theta, shape)
    if not np.isfinite(h).all() or np.any(h <= 0):
        return 1e100, np.zeros(2)
    p = -np.expm1(-h)
    observed = dead/alive
    # Conditional binomial deviance divided by two. The saturated constant
    # keeps the objective near zero on exact population counts.
    value = np.sum(xlogy(dead, observed) + xlogy(alive-dead, 1.0-observed)
                   - dead*np.log(p) + (alive-dead)*h)
    grad_h = (alive-dead) - dead/np.expm1(h)
    grad = np.array([np.sum(grad_h*dk), np.sum(grad_h*dtheta)], dtype=float)
    return float(value), grad


def fit_conditional(dead, alive, *, gamma_per_week: float, starts=(0.1, 1.0, 5.0)) -> Fit:
    dead = np.asarray(dead, dtype=float)
    alive = np.asarray(alive, dtype=float)
    if dead.ndim != 1 or dead.shape != alive.shape or len(dead) < 6:
        raise ValueError('Expected one contiguous quiet window of at least six weeks')
    if not np.isfinite(dead).all() or not np.isfinite(alive).all() or np.any(alive <= 0) or np.any(dead < 0) or np.any(dead >= alive):
        raise ValueError('Invalid weekly deaths or risk sets')
    if gamma_per_week <= 0 or not np.isfinite(gamma_per_week):
        raise ValueError('Invalid Gompertz slope')
    shape = np.exp(gamma_per_week*np.arange(len(dead)))
    k_start = max(1e-8, float(np.mean(-np.log1p(-dead[:4]/alive[:4]) / shape[:4])))
    candidates=[]
    for theta_start in starts:
        result = minimize(_objective,[np.log(k_start),float(theta_start)],
            args=(dead,alive,shape),method='L-BFGS-B',jac=True,
            bounds=[(np.log(1e-8),np.log(1e-1)),(0.0,100.0)],
            options=dict(maxiter=1000,ftol=1e-15,gtol=1e-9,maxls=50))
        candidates.append(result)
    result=min(candidates,key=lambda r:float(r.fun))
    k=float(np.exp(result.x[0]))
    theta=float(result.x[1])
    gradient_norm=float(np.linalg.norm(result.jac,ord=np.inf))
    success=bool(np.isfinite(result.fun) and np.isfinite(k) and np.isfinite(theta)
                 and (result.success or gradient_norm < 1e-5))
    return Fit(k,theta,success,float(result.fun),gradient_norm,int(result.nit),
               bool(theta < .01),bool(theta >= 99.9))

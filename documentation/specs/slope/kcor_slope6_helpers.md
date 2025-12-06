# KCOR Slope Normalization Python Helpers

This file defines Python utilities to implement the KCOR slope normalization procedure described in `kcor_slope_normalization_spec.md`.

The helpers use:

- `statsmodels` for linear median (τ = 0.5) quantile regression.
- `cvxpy` for constrained quadratic median regression with `c >= 0`.
- `numpy` for core array handling.

You can either copy-paste these into your KCOR project, or import them from a module.

---

## Imports

```python
import numpy as np
import cvxpy as cp

try:
    import statsmodels.formula.api as smf
    import pandas as pd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
```
---

## Helper: linear median regression slope

Fits:

    logh ≈ a_lin + b_lin * t_c

with τ = 0.5 and returns `(a_lin, b_lin)`.

If `statsmodels` is not available, this can be implemented via `cvxpy` as well.

```python
def fit_linear_median(t, logh, tau=0.5):
    """
    Fit logh ≈ a_lin + b_lin * t_c  using median quantile regression.

    Parameters
    ----------
    t : array-like
        Time values (1D array).
    logh : array-like
        log(hazard) values aligned with t.
    tau : float
        Quantile level, default 0.5 (median).

    Returns
    -------
    a_lin, b_lin, t_mean : floats
        Fitted intercept, slope, and the time centering constant t_mean.
    """

    t = np.asarray(t)
    logh = np.asarray(logh)
    t_mean = t.mean()
    t_c = t - t_mean

    if HAS_STATSMODELS:
        df = pd.DataFrame({
            "logh": logh,
            "t_c": t_c,
        })
        fit = smf.quantreg("logh ~ t_c", df).fit(q=tau)
        a_lin = float(fit.params["Intercept"])
        b_lin = float(fit.params["t_c"])
    else:
        # Fallback: use cvxpy for linear quantile regression
        X = np.column_stack([
            np.ones_like(t_c),
            t_c,
        ])
        beta = cp.Variable(2)
        residuals = logh - X @ beta
        loss = cp.sum(cp.maximum(tau * residuals, (tau - 1) * residuals))
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve()
        a_lin, b_lin = beta.value

    return a_lin, b_lin, float(t_mean)
```
---

## Helper: constrained quadratic median regression (c >= 0)

Fits:

    logh ≈ a + b * t_c + c * t_c^2

with τ = 0.5 and constraint `c >= 0` using `cvxpy`.

```python
def fit_quadratic_quantile(t, logh, tau=0.5):
    """
    Fit logh ≈ a + b * t_c + c * t_c^2  with c >= 0
    using quantile regression via cvxpy.

    Parameters
    ----------
    t : array-like
        Time values (1D array).
    logh : array-like
        log(hazard) values aligned with t.
    tau : float
        Quantile level, default 0.5 (median).

    Returns
    -------
    a, b, c, t_mean : floats
        Fitted parameters and the time centering constant t_mean.
    """

    t = np.asarray(t)
    logh = np.asarray(logh)
    # Center time for stability
    t_mean = t.mean()
    t_c = t - t_mean

    # Design matrix: [1, t_c, t_c^2]
    X = np.column_stack([
        np.ones_like(t_c),
        t_c,
        t_c**2,
    ])

    # Variables: beta = [a, b, c]
    beta = cp.Variable(3)
    residuals = logh - X @ beta

    # Quantile check loss
    loss = cp.sum(cp.maximum(tau * residuals, (tau - 1) * residuals))

    # Constraint: c >= 0  (beta[2] is c)
    constraints = [beta[2] >= 0]

    # Solve
    problem = cp.Problem(cp.Minimize(loss), constraints)
    problem.solve()

    a, b, c = beta.value
    return float(a), float(b), float(c), float(t_mean)
```
---

## Helper: KCOR per-cohort normalization

This is the high-level function you call per cohort. It:

1. Computes `logh` from `h`.
2. Fits a linear median regression on `logh` vs centred `t`.
3. If the linear slope `b_lin` is non-negative, uses linear mode.
4. If `b_lin` is negative, falls back to quadratic mode with `c >= 0`.
5. Returns the normalized hazard and the fitted parameters.

```python
def kcor_normalize_hazard(t, h, tau=0.5, min_h=1e-12):
    """
    KCOR slope normalization for a single cohort.

    Uses:
    - Linear median regression if b_lin >= 0.
    - Quadratic median regression with c >= 0 if b_lin < 0.

    Parameters
    ----------
    t : array-like
        Time values (1D array).
    h : array-like
        Hazard values at each time t (1D array, same length).
    tau : float
        Quantile for regression, default 0.5 (median).
    min_h : float
        Minimum hazard to avoid log(0). Values below this are clipped.

    Returns
    -------
    h_norm : np.ndarray
        Slope-normalized hazard values, same shape as h.
    params : dict
        Dictionary with fitted parameters and mode:
        {
            "mode": "linear" or "quadratic",
            "a": ...,
            "b": ...,
            "c": ... (0 in linear mode),
            "t_mean": ...,
            "tau": ...,
        }
    """

    t = np.asarray(t, dtype=float)
    h = np.asarray(h, dtype=float)

    # Clip hazards to avoid log(0)
    h_clipped = np.clip(h, min_h, np.inf)
    logh = np.log(h_clipped)

    # 1. Linear median regression
    a_lin, b_lin, t_mean_lin = fit_linear_median(t, logh, tau=tau)

    if b_lin >= 0:
        # Linear mode
        t_c = t - t_mean_lin
        logh_norm = logh - b_lin * t_c
        h_norm = np.exp(logh_norm)

        params = {
            "mode": "linear",
            "a": a_lin,
            "b": b_lin,
            "c": 0.0,
            "t_mean": t_mean_lin,
            "tau": tau,
        }
        return h_norm, params

    # 2. Quadratic mode if b_lin < 0
    a, b, c, t_mean = fit_quadratic_quantile(t, logh, tau=tau)
    t_c = t - t_mean
    logh_norm = logh - (b * t_c + c * t_c**2)
    h_norm = np.exp(logh_norm)

    params = {
        "mode": "quadratic",
        "a": a,
        "b": b,
        "c": c,
        "t_mean": t_mean,
        "tau": tau,
    }
    return h_norm, params
```
---

## Example usage

```python
# Suppose you have arrays t and h for one cohort
h_norm, params = kcor_normalize_hazard(t, h, tau=0.5)

print("Mode:", params["mode"])
print("b:", params["b"], "c:", params["c"])

# h_norm can now be used to compute cumulative hazards and KCOR ratios.
```

You can apply kcor_normalize_hazard independently to each cohort (unvaccinated, dose 1, dose 2, dose 3, etc.) using the same τ and baseline window logic.

This keeps the implementation consistent with the spec and preserves the KCOR negative-control behavior while handling negative-slope cohorts in a principled way.

# KCOR Slope Normalization: Linear + Quadratic Quantile Regression Spec

This document specifies how to estimate and remove cohort-specific baseline hazard trends in KCOR using median (τ = 0.5) quantile regression, with an optional quadratic term to handle depletion-driven curvature, and a non‑negative curvature constraint (c ≥ 0).

The core idea:

- For each cohort, we model log h(t) over a chosen baseline window with
  - a linear term in centred time; or
  - a quadratic term in centred time with c ≥ 0 when the linear slope is implausibly low (e.g., negative).
- We then subtract the fitted trend (linear or quadratic) from log h(t) to obtain a slope‑normalized hazard.
- This normalization uses b alone when the linear slope is reasonable, and b + c (full quadratic shape) when the cohort shows a pathological apparent slope.

All cohorts (unvaccinated, vax, boosted, etc.) must use the same decision rule to avoid bias.

---

## 1. Inputs and notation

Per cohort, over a chosen baseline time window:

- t[i] – time values for each hazard point (in years, weeks, or days; but consistent across cohorts).
- h[i] – estimated hazard at time t[i], e.g. weekly deaths / person‑weeks.
- logh[i] = log(h[i]) – natural log of hazard (only where h[i] > 0).
- n – number of time points in the baseline window (after exclusions).

Constraints/assumptions:

- Only include time points with sufficient events (e.g., deaths >= min_deaths) so that hazard estimates are not dominated by noise.
- Time values may be arbitrary (calendar date, days since start, etc.) as long as they are numeric and strictly increasing.

---

## 2. Time centering

For numerical stability and reduced collinearity between the linear and quadratic terms, centre time within the baseline window:

1. Compute the mean time:

   t_mean = mean(t[i] for i in baseline window)

2. Define centred time:

   t_c[i] = t[i] - t_mean

All regression fits below use t_c[i] instead of t[i].

This centering does not change the shape of the fitted trend or the KCOR-normalized hazards (it only shifts the intercept by a constant that cancels in ratios).

---

## 3. Overview of the decision rule

For each cohort:

1. Fit linear median regression on logh vs t_c.
2. If the fitted linear slope b_lin is non‑negative (and within a plausible range), use linear normalization with that slope.
3. If b_lin is negative (or clearly implausible biologically), refit using a quadratic median regression with a non‑negative curvature constraint c ≥ 0 and use the full quadratic correction for slope normalization.

This gives:

- Simple exponential trend removal when things look normal.
- Curved trend removal when depletion/selection artefacts make the cohort’s apparent slope go negative, while enforcing upward curvature consistent with frailty models.

---

## 4. Linear median regression (τ = 0.5)

### 4.1 Model

For linear mode, we fit:

logh[i] ≈ a_lin + b_lin * t_c[i]

using quantile regression with τ = 0.5 (median regression).

### 4.2 Fit

Implementation (conceptual):

- Design matrix X_lin has two columns:
  - Intercept = 1
  - t_c[i]

Fit via any quantile regression implementation (statsmodels or cvxpy). Example with statsmodels is fine here, since we do not need constraints in the linear case.

### 4.3 Plausibility check

Once (a_lin, b_lin) are estimated:

- If b_lin >= 0 (and optionally within a reasonable upper bound, e.g. < 0.15 / year):
  - We accept linear mode.
- Otherwise (b_lin < 0), we escalate to quadratic mode.

The threshold and any upper bound can be configurable, but the key trigger is: negative slope ⇒ quadratic mode.

### 4.4 Linear slope normalization

If linear mode is accepted, define the linear trend function:

f_lin(t_c) = b_lin * t_c

Normalize the log-hazard for this cohort as:

logh_norm[i] = logh[i] - f_lin(t_c[i])

Equivalently in hazard space:

h_norm[i] = h[i] * exp(-b_lin * t_c[i])

Within the baseline window, this removes the linear trend from logh and produces a flat baseline hazard for that cohort in the KCOR sense.

---

## 5. Quadratic median regression with c ≥ 0 (τ = 0.5)

Quadratic mode is used if the linear fit yields a negative slope (b_lin < 0).

### 5.1 Model

We fit:

logh[i] ≈ a + b * t_c[i] + c * t_c[i]^2

with the constraint:

c ≥ 0

This allows the model to capture upward curvature due to frailty depletion (negative or low slope early, bending upward over time), but disallows downward-curving baselines that contradict the frailty story.

### 5.2 Optimization problem

Given:

- y[i] = logh[i]
- x1[i] = t_c[i]
- x2[i] = t_c[i]^2

We solve the median (τ = 0.5) quantile regression:

Minimize over (a, b, c):

sum_i ρ_τ( y[i] - (a + b * x1[i] + c * x2[i]) )

where the check loss is:

ρ_τ(u) = τ * u          if u ≥ 0
       = (τ - 1) * u    if u < 0

Subject to the linear constraint:

c ≥ 0

This is a convex linear-programming problem and can be solved with cvxpy.

### 5.3 Python implementation with cvxpy

Below is a reference implementation for the quadratic median regression with c ≥ 0:

```python
import numpy as np
import cvxpy as cp

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
        t_c**2
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

Notes:

- This function centres time internally and returns t_mean so that the caller can reproduce the same t_c[i] = t[i] - t_mean when doing normalization.
- tau is defaulted to 0.5 for median regression, but is configurable.

### 5.4 Quadratic slope normalization

Once (a, b, c, t_mean) are fitted for a cohort in quadratic mode, we define:

t_c[i] = t[i] - t_mean
f_quad(t_c[i]) = b * t_c[i] + c * t_c[i]^2

Then slope-normalize:

logh_norm[i] = logh[i] - f_quad(t_c[i])

Equivalently in hazard space:

h_norm[i] = h[i] * exp( - (b * t_c[i] + c * t_c[i]**2) )

Within the baseline window, this flattens the fitted curved trend, making the normalized hazard approximately constant despite the depletion-driven negative apparent slope.

Outside the baseline window, this acts as an extrapolated shape correction, analogous to the exponential correction in the linear case.

---

## 6. Full per-cohort algorithm (linear + quadratic mode)

For each cohort and chosen baseline window:

1. Compute hazard and log-hazard
   - For each time point t[i] in the window, compute hazard h[i] and logh[i] = log(h[i]).
   - Exclude points with too few events if desired.

2. Centre time
   - Compute t_mean = mean(t) over included points.
   - Set t_c[i] = t[i] - t_mean.

3. Linear median regression (τ = 0.5)
   - Fit logh[i] ≈ a_lin + b_lin * t_c[i] using quantile regression (e.g. statsmodels or cvxpy).
   - If b_lin >= 0 (and optionally within a plausible upper bound), use linear mode:
     - Define logh_norm[i] = logh[i] - b_lin * t_c[i] for all times.
     - Done for this cohort.
   - Else (b_lin < 0), escalate to quadratic mode.

4. Quadratic median regression with c ≥ 0
   - Fit logh[i] ≈ a + b * t_c[i] + c * t_c[i]^2 with c >= 0 via the fit_quadratic_quantile function above.
   - Use the returned t_mean for consistency when normalizing.

5. Quadratic normalization
   - For all times t[i] (not just those in the baseline window), compute:
     - t_c[i] = t[i] - t_mean,
     - logh_norm[i] = logh[i] - (b * t_c[i] + c * t_c[i]**2),
     - Or h_norm[i] = h[i] * exp( - (b * t_c[i] + c * t_c[i]**2) ).

6. Common handling across cohorts
   - Apply the same decision rule (linear vs. quadratic) independently but identically to all cohorts (unvaccinated, vaccinated, boosted, etc.).
   - Use the resulting h_norm(t) curves to compute cumulative hazards and KCOR ratios.

This preserves KCOR’s negative-control property:

- In a null world where all differences are due to frailty/selection, each cohort’s log-hazard can be approximated by a smooth trend (linear or curved).
- KCOR removes that trend; each normalized hazard becomes flat in the baseline window.
- Any persistent KCOR signal outside the baseline window is then less confounded by baseline slope artefacts.

---

## 7. Implementation notes

- Use double precision (float64) for all calculations.
- Consider adding basic safeguards:
  - Require a minimum number of time points and events in the baseline window before fitting.
  - Handle failures of the solver gracefully (e.g. retry, fall back to linear).
- Document the exact τ, baseline window, and trigger conditions (b_lin < 0) in any paper or code comments for reproducibility.
- The same framework can be extended to other τ (e.g. τ = 0.25) if you want to emphasise lower-envelope behavior instead of the median.

This spec should be sufficient for an implementor to add robust linear/quadratic slope normalization to KCOR using Python and cvxpy.

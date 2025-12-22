# KCOR 6.0 – Fit Diagnostics & Reporting Guide

This document defines **what fit diagnostics should be computed and printed** when estimating
$(k_d, \theta_d)$ in KCOR 6.0.

The goal of diagnostics in KCOR is **sanity checking and robustness**, not hypothesis testing.

This file is intended to be referenced by the KCOR 6.0 spec and sample code.

---

## 1. Purpose of Fit Diagnostics in KCOR

In KCOR, gamma-frailty fitting is used to:

- remove cohort-composition–driven hazard curvature,
- identify stable normalization parameters during quiet periods,
- detect pathological or unstable fits.

Diagnostics are therefore used to:
- verify that the quiet window is appropriate,
- confirm numerical stability,
- compare fits across windows or cohorts.

They are **not** used to claim causality or statistical significance.

---

## 2. Required Outputs (Per Cohort)

For each cohort (dose) $d$, the following values **must be printed**.

### 2.1 Parameter estimates

```text
k_hat     = estimated baseline hazard level
theta_hat = estimated frailty variance
```

Example:

```text
Dose 1:
  k̂ = 2.31e-4
  θ̂ = 1.87
```

---

### 2.2 Log-likelihood (or negative log-likelihood)

For Poisson MLE fits, report either:

- log-likelihood $\log L$, or
- negative log-likelihood $-\log L$

Example:

```text
logL = -183.4
```

Use cases:
- compare fits across different quiet windows,
- detect obviously bad or unstable fits.

---

### 2.3 Number of observations

Always print:

```text
n_obs    = number of time bins used in the quiet window
n_params = 2
```

Example:

```text
n_obs = 104
n_params = 2
```

This provides essential context for interpreting fit quality.

---

## 3. Recommended Diagnostics (Strongly Encouraged)

### 3.1 Residual error in cumulative-hazard space

When fitting in cumulative-hazard space, compute:

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_t \left( H^{obs}(t) - H^{model}(t) \right)^2 }.
$$

Example:

```text
RMSE(H_obs) = 3.1e-4
```

Why this matters:
- sensitive to quiet-window contamination,
- highlights residual seasonality or epidemic structure,
- more informative than pointwise hazard residuals.

---

### 3.2 Visual residual check (optional but recommended)

Plot, for the quiet window:

- $H^{obs}(t)$ vs. fitted $H^{model}(t)$
- or residuals $H^{obs}(t) - H^{model}(t)$ vs. $t$

This is often the fastest way to spot violations of quiet-period assumptions.

---

## 4. Optional Advanced Diagnostics

These are optional but useful when investigating instability.

### 4.1 Hessian / curvature diagnostics

If available from the optimizer:

- approximate Hessian at optimum,
- condition number of the Hessian,
- standard errors for $(k_d, \theta_d)$.

Large condition numbers indicate:
- weak identifiability,
- insufficient quiet-window length,
- near-collinearity between parameters.

---

### 4.2 Information criteria (AIC / BIC)

If comparing alternate baseline models (e.g., $g(t)=1$ vs Gompertz):

$$
\text{AIC} = 2k - 2\log L,
$$

where $k=2$ parameters per cohort in the default model.

Use **only for relative comparison**, not absolute model validation.

---

## 5. What NOT to Report

The following are **not appropriate** for KCOR normalization fits:

- $R^2$
- p-values for $k_d$ or $\theta_d$
- hypothesis tests against $\theta_d = 0$
- claims of statistical significance

KCOR uses fitting to **remove bias**, not to test causal effects.

---

## 6. Recommended Print Template

Use a consistent, minimal printout like this:

```text
Dose 1:
  k̂ = 2.31e-4
  θ̂ = 1.87
  logL = -183.4
  RMSE(H_obs) = 3.1e-4
  n_obs = 104
```

This level of reporting is:
- technically rigorous,
- easy to audit,
- appropriate for both internal validation and external review.

---

## 7. Summary

Every KCOR 6.0 fit should report:

**Required**
- $(k_d, \theta_d)$
- log-likelihood
- number of observations

**Recommended**
- RMSE in cumulative-hazard space
- visual residual checks

These diagnostics strengthen KCOR without changing its interpretation or goals.

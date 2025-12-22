# KCOR 6.0 – Gamma-Frailty Normalization Specification

**Purpose.**  
KCOR 6.0 formalizes the selection-neutralization step in KCOR using a gamma-frailty model fitted during epidemiologically quiet periods. The goal is to remove cohort-composition–driven hazard curvature before comparing cumulative outcomes across cohorts.

This document is intended as a **spec for implementation** (e.g., in Cursor) rather than a narrative paper.

---

## 1. Recap: KCOR 5.4 (Current Method)

KCOR 5.4 proceeds as follows:

1. Compute cohort-specific hazard functions

$$
h_d(t) = \frac{D_{d,t}}{Y_{d,t}}
$$

where:
- $D_{d,t}$ = deaths in cohort $d$ during time bin $t$
- $Y_{d,t}$ = person-time at risk

2. **Slope-normalize** $h_d(t)$ using a quiet period to remove curvature induced by cohort heterogeneity (selection / depletion).

3. Integrate the adjusted hazards to obtain cumulative adjusted hazards:

$$
\tilde H_d(t) = \int_0^t \tilde h_d(s)\,ds
$$

4. Compare cohorts using ratios of cumulative adjusted hazards:

$$
\text{KCOR}(t) = \frac{\tilde H_{d_1}(t)}{\tilde H_{d_0}(t)}
$$

KCOR 6.0 replaces the heuristic slope-normalization step with a **parametric gamma-frailty normalization** that makes the selection correction explicit and reproducible.

---

## 2. KCOR 6.0 Overview

### Key idea

Observed cohort hazards are distorted by **heterogeneous baseline risk**. Individuals with higher latent risk die earlier, inducing downward curvature in cohort hazards even in the absence of any intervention effect.

KCOR 6.0:
- **Estimates cohort-specific selection dynamics** during quiet periods using a gamma-frailty model
- **Fits and normalizes in cumulative-hazard space**
- Compares cohorts using **depletion-neutralized cumulative hazards**

### Implementation decisions (for `code/KCOR.py` in this repo)

This spec is intended to drive the `code/KCOR.py` v6.0 implementation. The following decisions are **fixed**:

- **Cohort definition**: a cohort is indexed by
  - enrollment period (sheet), and
  - `YearOfBirth` group (e.g., 1920, 1930, …), and
  - `Dose`
  
  Additionally, include an **All Ages** cohort with `YearOfBirth = -2`.
- **Fit method**: nonlinear least squares in cumulative-hazard space (**do not** optimize a likelihood / **do not** use MLE).
- **Observed hazard used to build $H^{\\text{obs}}$**: use the project’s discrete hazard transform `hazard_from_mr_improved(MR)` (see §6.1.1).
- **Time axis for fitting**: use `t` as **weeks since enrollment** (`t_fit = t`), and exclude pre-skip weeks from fitting (see §6.1.2).
- **Fit window selection**: use only time bins whose **calendar** ISO week lies in `QUIET_WINDOW`.
- **Logging**: print one fit summary line per `(EnrollmentDate, YearOfBirth, Dose)` in `KCOR_summary.log` using a `KCOR6_FIT,...` tagged line (see §6.4).

---

## 3. Model Definition (Default Specification)

### 3.1 Hazard model

For cohort $d$ (enrollment period × age group × dose):

$$
h_d(t) = \frac{k_d\,g(t)}{1 + \theta_d\,k_d\,G(t)}
$$

where:
- $k_d$ = cohort-specific baseline hazard level
- $\theta_d$ = cohort-specific frailty variance (strength of selection)
- $g(t)$ = shared baseline time shape
- $G(t)=\int_0^t g(s)\,ds$

### 3.2 Default baseline

**KCOR 6.0 default:**

$$
g(t) = 1 \quad \Rightarrow \quad G(t)=t
$$

This choice:
- Minimizes degrees of freedom
- Forces all curvature into frailty (selection)
- Is conservative and highly stable in practice

---

## 4. Gamma Frailty Identity (Core Mathematics)

Assume individual hazards:

$$
h_{i,d}(t) = z_{i,d}\,h_{0,d}(t), \quad z_{i,d} \sim \text{Gamma}(\text{mean}=1,\,\text{var}=\theta_d)
$$

After integrating over frailty, the **observed cumulative hazard** satisfies:

$$
H_d^{\text{obs}}(t) = \frac{1}{\theta_d}\,\log\!\left(1+\theta_d H_{0,d}(t)\right)
$$

This can be inverted exactly:

$$
H_{0,d}(t) = \frac{e^{\theta_d H_d^{\text{obs}}(t)} - 1}{\theta_d}
$$

This inversion is the **key normalization step** in KCOR 6.0.

---

## 5. Quiet Window Definition

### Default

```text
QUIET_WINDOW = [2022-24, 2024-16]
```

(ISO week notation: YYYY-WW)

### Meaning

A quiet window is a period with:
- minimal COVID incidence and mortality
- no major epidemic or policy shocks
- hazards dominated by baseline risk + selection

The quiet window is used **only** to estimate $(k_d, \theta_d)$, not to assume absence of prior selection effects.

---

## 6. Parameter Estimation (Cumulative-Hazard Least Squares; Implementation Default)

### 6.1 Data inputs (per cohort $d$)

For each cohort $d$ and each weekly time bin $t$:
- deaths: $D_{d,t}$
- person-time: $Y_{d,t}$
- time index: $t$ (weeks since cohort enrollment; `t = 0, 1, 2, ...`)

#### 6.1.1 Observed hazard definition used in `code/KCOR.py`

In this repo, the weekly mortality input is treated as a discrete-time risk:

$$
\text{MR}_{d,t}=\frac{D_{d,t}}{Y_{d,t}}
$$

The observed hazard used for KCOR 6.0 fitting is the project’s discrete hazard transform:

$$
h_d^{\text{obs}}(t) = -\log\!\left(\frac{1 - 1.5\,\text{MR}_{d,t}}{1 - 0.5\,\text{MR}_{d,t}}\right)
$$

This matches the `hazard_from_mr_improved()` transform in `code/KCOR.py`.

#### 6.1.2 Accumulation start (skip weeks)

Observed cumulative hazards are accumulated only after a fixed skip period:

```text
SKIP_WEEKS = DYNAMIC_HVE_SKIP_WEEKS  (in code)
```

Define an effective hazard for accumulation:

$$
h_d^{\text{eff}}(t)=
\begin{cases}
0, & t < \text{SKIP\_WEEKS} \\
h_d^{\text{obs}}(t), & t \ge \text{SKIP\_WEEKS}
\end{cases}
$$

Then compute the observed cumulative hazard (weekly bins, $\Delta t = 1$):

$$
H_d^{\text{obs}}(t)=\sum_{s \le t} h_d^{\text{eff}}(s)\,\Delta t
$$

### 6.2 Model in cumulative-hazard space

With $g(t)=1$ and $G(t)=t$, the depletion-neutralized cumulative hazard is:

$$
H_{0,d}(t)=k_d\,t
$$

Therefore the observable cumulative hazard satisfies:

$$
H_d^{\text{obs}}(t)=\frac{1}{\theta_d}\log\!\left(1+\theta_d k_d t\right)
$$

### 6.3 Estimation strategy

Implementation-default approach (no MLE):

- Fit **each cohort independently**, where cohort = (EnrollmentDate sheet, YearOfBirth group, Dose), including `YearOfBirth=-2` (All Ages).
- Use **only quiet-window data** defined in **calendar ISO week** space (`QUIET_WINDOW`), not in `t` space.
- Use only points with `t >= SKIP_WEEKS` (avoid the forced-flat pre-skip segment in $H^{\\text{obs}}$).
- Estimate $(k_d,\theta_d)$ by minimizing squared error in cumulative-hazard space:

$$
H_d^{\text{obs}}(t) \approx \frac{1}{\theta_d}\log\!\left(1+\theta_d k_d t\right)
$$

over the quiet window using nonlinear least squares with constraints:

```text
k_d > 0
theta_d >= 0
```

### 6.4 Fit diagnostics and `KCOR_summary.log` reporting

For each fitted cohort `(EnrollmentDate, YearOfBirth, Dose)`, print **one line** (not one per week) to `KCOR_summary.log`:

```text
KCOR6_FIT,EnrollmentDate=...,YoB=...,Dose=...,k_hat=...,theta_hat=...,RMSE_Hobs=...,n_obs=...,success=...,note=...
```

Minimum required values:
- $\\hat{k}_d$
- $\\hat{\\theta}_d$
- `n_obs` (number of quiet-window time bins used for the fit)
- `RMSE_Hobs` in cumulative-hazard space:

$$
\text{RMSE}=\sqrt{\frac{1}{n}\sum_t\left(H_d^{\text{obs}}(t)-H_d^{\text{model}}(t)\right)^2}
$$

---

## 7. Normalization Step (Core KCOR 6.0 Operation)

For each cohort $d$:

1. Compute observed cumulative hazard:

$$
H_d^{\text{obs}}(t) = \sum_{s \le t} h_d^{\text{eff}}(s)\,\Delta t
$$

2. Apply gamma-frailty inversion:

$$
H_{0,d}(t) = \frac{e^{\theta_d H_d^{\text{obs}}(t)} - 1}{\theta_d}
$$

This yields the **depletion-neutralized cumulative hazard**.

Optionally recover an adjusted hazard via differencing:

$$
h_{0,d}(t) \approx H_{0,d}(t)-H_{0,d}(t-1)
$$

### Application interval

- The fit uses only quiet-window data (calendar ISO weeks).
- The normalization is applied to the full post-enrollment cumulative hazard trajectory as it is accumulated (i.e., starting at `t >= SKIP_WEEKS`).

---

## 8. KCOR Computation

KCOR is computed using **normalized cumulative hazards**:

$$
\text{KCOR}(t) = \frac{H_{0,d_1}(t)}{H_{0,d_0}(t)}
$$

where $d_0$ is the reference cohort (typically dose-0).

---

## 9. Robustness & Diagnostics (Recommended)

- Refit using alternate quiet windows
- Compare against KCOR 5.4 slope-normalized results
- Age-stratify, then standardize
- Test $g(t)=e^{\gamma t}$ and Gompertz+season as sensitivity (not default)
- Validate using negative-control outcomes

---

## 10. Summary

KCOR 6.0 replaces heuristic slope normalization with an explicit gamma-frailty correction that:
- models selection-induced hazard curvature mechanistically
- uses quiet periods for parameter identification
- **fits and operates in cumulative-hazard space**
- preserves KCOR’s core philosophy of fixed cohorts and cumulative outcomes

This is the recommended default specification for future KCOR analyses.

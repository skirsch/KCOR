
# KCOR: Kirsch Cumulative Outcomes Ratio
## Mortality-Neutralized Cohort Comparison Using Generalized Gompertz Dynamics

---

## 1. Overview

**KCOR (Kirsch Cumulative Outcomes Ratio)** is a method for estimating the **net mortality impact of an intervention**
(e.g., vaccination) from **retrospective observational data** when cohorts differ in **baseline mortality and frailty**
and therefore cannot be validly compared using standard epidemiological techniques.

KCOR addresses a common failure mode in retrospective intervention studies: when cohort selection induces
**systematic differences in mortality curvature**, standard methods confound these structural differences
with treatment effects.

---

## 2. Why Standard Methods Fail

Vaccinated and unvaccinated cohorts are often **not exchangeable**:

- Selection bias (static and dynamic Healthy Vaccinee Effect, HVE) creates cohorts with different frailty compositions
- These differences manifest as **different hazard slopes (curvature)**, not merely different hazard levels
- This violates assumptions underlying Cox proportional hazards models, ASMRs, IPTW/probabilistic matching, and 1:1 matching

Even when covariates are balanced, differential frailty depletion over event time breaks proportional hazards.

---

## 3. Core Insight of KCOR

> **Cohorts cannot be compared fairly unless their intrinsic mortality curvature is first neutralized.**

Observed mortality differences reflect:
1. Baseline Gompertz mortality  
2. Frailty mixing and depletion of susceptibles  
3. Selection bias (HVE)  
4. Any true intervention effect  

KCOR removes (2) and (3) by **directly estimating and neutralizing cohort-specific mortality curvature in log-hazard space**
*before* comparing cohorts.

---

## 4. Cohort Construction

- Cohorts are fixed at enrollment
- Assignment is based on intervention status at enrollment week
- No censoring and no cohort switching (avoids health-seeking bias)
- Analysis proceeds forward in **event time**

---

## 5. Hazard Computation

Let:

- **s** = event time since cohort enrollment (e.g., weeks since enrollment)
- **D(s)** = deaths occurring during interval *s*
- **N(s)** = number alive at the start of interval *s*

Hazard is treated as **piecewise-constant** over weekly intervals.  
The discrete-time hazard is computed as:

$$
h(s) = -\ln\left(1 - \frac{D(s)}{N(s)}\right)
$$

---

## 6. Mortality Model

At the individual level, adult human mortality is well approximated by Gompertz growth:

$$
h_i(s) = z_i e^{k s}
$$

where:
- $z_i$ is individual frailty
- $k$ is the Gompertz log-slope

Population-level deviations arise from frailty mixing and selective depletion.

---

## 7. Generalized Gompertz–Frailty Model (Used in KCOR)

Cohorts formed by selection (e.g., vaccination choice or timing) are not well described by gamma frailty.
KCOR therefore models cohort-level mortality curvature using the following **4-parameter generalized form**:

$$
\log h(s)
= C + k_\infty s + (k_0 - k_\infty)\,\tau\left(1 - e^{-s/\tau}\right)
$$

**Parameters:**

- $C$ : scale constant  
- $k_0$ : initial cohort log-slope  
- $k_\infty$ : asymptotic Gompertz log-slope  
- $\tau$ : relaxation time constant  

The instantaneous log-slope is:

$$
k(s) = \frac{d}{ds}\log h(s)
= k_\infty + (k_0 - k_\infty)e^{-s/\tau}
$$

Thus the slope relaxes smoothly from $k_0$ at enrollment toward $k_\infty$ as $s \to \infty$,
capturing depletion and resilience dynamics without assuming gamma frailty.

---

## 8. Slope Estimation

- **Ages < 80**: fit  
  $$
  \log h(s) = \alpha + \beta s
  $$
  using quantile regression (τ = 0.5), which is robust to transient shocks and outliers

- **Ages ≥ 80**: fit the generalized model above and estimate
  $(k_0, k_\infty, \tau)$

---

## 9. Slope / Shape Normalization

KCOR removes **cohort-specific mortality curvature** by subtracting the fitted shape in **log-hazard space**.
Normalization is performed in hazard space, not cumulative mortality space.

### 9.1 Linear (Gompertz) case — ages < 80

When cohort mortality is well approximated by a log-linear model:

$$
\log h(s) = \alpha + \beta s
$$

the cohort-specific shape is:

$$
g(s) = \beta s
$$

Normalization is:

$$
\log h_{\text{adj}}(s) = \log h(s) - \beta s
$$

or equivalently,

$$
h_{\text{adj}}(s) = h(s)e^{-\beta s}
$$

### 9.2 Generalized (curved) case — ages ≥ 80

For cohorts modeled using the generalized form, the cohort-specific shape is:

$$
g(s) = k_\infty s + (k_0 - k_\infty)\,\tau\left(1 - e^{-s/\tau}\right)
$$

Shape normalization subtracts this entire function:

$$
\log h_{\text{adj}}(s) = \log h(s) - g(s)
$$

or equivalently,

$$
h_{\text{adj}}(s)
= h(s)\exp\!\left[-k_\infty s - (k_0 - k_\infty)\,\tau(1 - e^{-s/\tau})\right]
$$

The scale parameter $C$ is absorbed by later baseline normalization.

---

## 10. KCOR Definition

For cohorts $A$ and $B$:

$$
\text{KCOR}(t)
=
\frac{\sum_{s \le t} h_{\text{adj},A}(s)}
{\sum_{s \le t} h_{\text{adj},B}(s)}
$$

**Interpretation:**

- KCOR ≈ 1 : mortality equivalence  
- KCOR > 1 : excess mortality in numerator cohort  
- KCOR < 1 : mortality benefit in numerator cohort  

---

## 11. Stabilization Rules

- KCOR is normalized to 1 using weeks **3–6** post-enrollment
- Weeks **1–2** are excluded to reduce dynamic HVE and acute deferral effects

The 3–6 week window empirically provides stable anchoring while avoiding immediate post-enrollment artifacts.

---

## 12. Validation

- **Negative controls**: KCOR remains flat under null, including extreme pathological selection scenarios
- **Positive controls**: transient wave-associated mortality bursts produce detectable departures from unity

---

## 13. Interpretation

KCOR estimates the **net cumulative mortality impact** after removing cohort mortality structure
in log-hazard space.

---

## 14. Confidence Intervals

Confidence intervals are computed via **Monte Carlo resampling with replacement**, incorporating uncertainty from:
- stochastic death counts within cohorts
- slope / shape estimation error

---

## 15. Summary

KCOR enables valid mortality comparisons in retrospective data when traditional approaches are confounded
by selection-induced hazard curvature, by explicitly modeling and removing cohort-specific mortality
structure before comparing cumulative hazards.

---

## 16. Schematic Illustration (Conceptual)

A useful way to visualize KCOR is in **log-hazard space**:

- Before normalization, cohorts exhibit different slopes and curvature due to selection and frailty depletion.
- Shape normalization removes these cohort-specific trends, flattening both curves under the null.
- Any remaining divergence after normalization represents excess mortality attributable to the intervention.

Such a schematic typically shows:
1. Two diverging log-hazard curves (pre-normalization)
2. Both curves flattened and aligned (post-normalization)
3. Residual separation only when a true effect exists

---

## 17. Why Cox / IPTW Fail Here (Explicit Contrast)

Cox proportional hazards and IPTW-based methods assume that, after adjustment, hazards differ only by a
*time-invariant multiplicative factor*.

In the presence of selection-induced curvature:

- Hazards are **non-proportional by construction**
- Matching on covariates cannot equalize frailty depletion dynamics
- IPTW reweights levels but does not remove **time-varying slope differences**

As a result, Cox/IPTW interpret curvature differences as treatment effects, producing biased estimates.
KCOR avoids this by removing curvature *prior* to comparison.

---

## 18. When KCOR Should NOT Be Used

KCOR is not appropriate when:

- Cohorts are already randomized or exchangeable (e.g., large RCTs)
- Event counts are too sparse to estimate hazard slopes reliably
- Follow-up is too short to observe mortality curvature
- Outcomes are not well modeled by hazard-based processes

In such cases, simpler estimators may be preferable.

---

## Appendix A: Derivation Sketch (Short)

Frailty depletion and selection alter the effective cohort frailty distribution over event time,
inducing curvature in $\log h(s)$. Allowing the cohort log-slope to relax exponentially from an
initial value $k_0$ to an asymptotic value $k_\infty$ yields the generalized model above without
requiring gamma frailty assumptions.

---

## Appendix B: Reviewer Framing

- KCOR does not assume proportional hazards; it estimates and removes cohort-specific hazard curvature
- Normalization occurs in log-hazard space, not cumulative mortality space
- The generalized model preserves Gompertz asymptotics while capturing depletion/selection curvature
- Negative controls demonstrate KCOR does not generate spurious signals under extreme selection
- The method is generic: it applies to any cohort comparison with selection-induced curvature

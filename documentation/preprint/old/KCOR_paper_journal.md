
# KCOR: Mortality-Neutralized Cohort Comparison Under Selection-Induced Hazard Curvature

## Abstract

Retrospective observational studies are frequently used to assess the mortality impact of medical interventions, yet such analyses are often invalidated by selection bias and non-exchangeable cohorts. In particular, intervention uptake commonly induces systematic differences in mortality curvature—differences in the shape of log-hazard trajectories over time—that violate the assumptions of standard epidemiologic methods such as Cox proportional hazards models, age-standardized mortality rates, and inverse-probability weighting. We introduce **KCOR (Kirsch Cumulative Outcomes Ratio)**, a method that neutralizes cohort-specific mortality curvature prior to comparison by explicitly modeling and removing intrinsic hazard dynamics. KCOR is grounded in Gompertz mortality with frailty and employs a generalized relaxation model to accommodate selection-induced depletion of susceptibles. We describe the KCOR framework, its mathematical foundation, and its validation using negative and positive control tests. KCOR enables valid mortality comparisons in settings where traditional retrospective methods fail.

## 1. Introduction

### 1.1 Retrospective evaluation of intervention mortality

Randomized controlled trials (RCTs) are the gold standard for causal inference but are typically underpowered to detect rare outcomes such as all-cause mortality. Consequently, retrospective observational analyses are widely used to evaluate mortality effects of medical interventions.

### 1.2 Failure of exchangeability under selection

When intervention uptake is voluntary or prioritized, cohorts are not exchangeable. The Healthy Vaccinee Effect (HVE), both static and dynamic, induces systematic differences in frailty and mortality dynamics that persist even after covariate adjustment.

### 1.3 Contribution of this work

This paper introduces KCOR, a method that explicitly removes cohort-specific mortality curvature prior to comparison, enabling valid inference from retrospective data where standard methods fail.

## 2. Methods

### 2.1 Conceptual framework: mortality level vs curvature

Differences in mortality between cohorts can arise from level effects (intercepts) or curvature effects (slopes and higher-order shape). Selection primarily induces curvature differences, which must be neutralized before comparison.

### 2.2 Cohort construction

Cohorts are fixed at enrollment and defined by intervention status at a specified time. No censoring or cohort switching is permitted. Analysis proceeds in event time.

### 2.3 Hazard estimation

Let $s$ denote event time since enrollment, $D(s)$ deaths during interval $s$, and $N(s)$ the number at risk at the start of $s$. Hazards are treated as piecewise-constant and computed as

$$
h(s) = -\ln\left(1 - \frac{D(s)}{N(s)}\right).
$$

### 2.4 Mortality modeling

#### 2.4.1 Individual-level Gompertz mortality

Adult human mortality is well approximated by

$$
h_i(s) = z_i e^{k s},
$$

where $z_i$ is individual frailty and $k$ the Gompertz log-slope.

#### 2.4.2 Frailty mixing and depletion

At the cohort level, frailty heterogeneity and selective depletion distort log-hazard trajectories, inducing curvature.

### 2.5 Generalized Gompertz–Frailty model

To accommodate selection-induced curvature, KCOR uses

$$
\log h(s)
= C + k_\infty s + (k_0 - k_\infty)\tau\left(1 - e^{-s/\tau}\right).
$$

The instantaneous slope is

$$
k(s) = k_\infty + (k_0 - k_\infty)e^{-s/\tau}.
$$

### 2.6 Slope estimation

#### 2.6.1 Linear case (ages < 80)

For younger cohorts,

$$
\log h(s) = \alpha + \beta s
$$

is estimated via quantile regression ($\tau=0.5$).

#### 2.6.2 Generalized case (ages ≥ 80)

For older cohorts, the generalized model is fit and $(k_0, k_\infty, \tau)$ estimated.

### 2.7 Curvature (slope/shape) normalization

Normalization is performed in log-hazard space.

#### 2.7.1 Linear normalization

$$
h_{\text{adj}}(s) = h(s)e^{-\beta s}.
$$

#### 2.7.2 Generalized shape normalization

$$
h_{\text{adj}}(s)
= h(s)\exp\left[-k_\infty s - (k_0 - k_\infty)\tau(1 - e^{-s/\tau})\right].
$$

### 2.8 KCOR estimator

For cohorts $A$ and $B$,

$$
\text{KCOR}(t)
= \frac{\sum_{s \le t} h_{\text{adj},A}(s)}{\sum_{s \le t} h_{\text{adj},B}(s)}.
$$

### 2.9 Stabilization and anchoring

KCOR is normalized using weeks 3–6 post-enrollment; weeks 1–2 are excluded.

### 2.10 Validation strategy

#### 2.10.1 Negative controls

Closed-population nulls and pathological frailty mixtures.

#### 2.10.2 Positive controls

Known mortality waves and transient risk periods.

### 2.11 Sensitivity analyses

Variation of normalization windows, slope estimation, and age stratification.

### 2.12 Uncertainty quantification

Monte Carlo resampling captures uncertainty from death counts and slope estimation.

## 3. Results

(Results to be presented for selected datasets.)

## 4. Discussion

KCOR provides a principled approach to retrospective mortality analysis by addressing selection-induced curvature directly.

## 5. Limitations

KCOR is not intended to replace RCTs and requires sufficient follow-up to estimate curvature reliably.

## 6. Conclusion

By neutralizing intrinsic mortality curvature, KCOR enables valid cohort comparisons where traditional methods fail.

## Supplementary Material

### Appendix A. Mathematical derivations
### Appendix B. Pathological frailty examples
### Appendix C. Additional figures
### Appendix D. Implementation details

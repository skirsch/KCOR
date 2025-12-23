# KCOR: Depletion-Neutralized Cohort Comparison via Gamma-Frailty Normalization Under Selection-Induced Hazard Curvature

<!--
paper_v5.md (methods-only manuscript; updated to KCOR v6)

Goal: Submit a methods paper focused on the KCOR methodology + validation via negative/positive controls.
This version deliberately avoids presenting any applied Czech2 vaccine-effect results (to be a separate paper).

KCOR v6 update:
- Replace v4 “slope/curvature normalization + anchoring” with a gamma-frailty normalization in cumulative-hazard space.
- Math formatting follows documentation/style_guide/KCOR_Markdown_Math_Style_Guide.md:
  use `$...$` for inline math and `$$ ... $$` for display math, with `$$` alone on its line.
-->

## Manuscript metadata (for journal submission)

- **Article type**: Methods / Statistical method
- **Running title**: KCOR via gamma-frailty normalization
- **Authors**: TODO (names, degrees)
- **Affiliations**: TODO
- **Corresponding author**: TODO (email, address)
- **Word count**: TODO
- **Keywords**: selection bias; healthy vaccinee effect; non-proportional hazards; frailty; gamma frailty; negative controls; causal inference

---

## Abstract

Retrospective observational studies are frequently used to assess the mortality impact of medical interventions, yet such analyses are often invalidated by selection bias and non-exchangeable cohorts. In particular, selective uptake can induce systematic differences in mortality curvature—differences in the time-evolution of cohort hazards driven by frailty heterogeneity and depletion of susceptibles—that violate the assumptions of standard epidemiologic tools such as Cox proportional hazards models, age-standardized mortality rates, and inverse-probability weighting. We introduce **KCOR (Kirsch Cumulative Outcomes Ratio)**, a method that neutralizes selection-induced curvature before cohort comparison by estimating and inverting a gamma-frailty mixture model in **cumulative-hazard space**. KCOR v6 fits cohort-specific selection parameters during epidemiologically quiet periods and transforms observed cumulative hazards to depletion-neutralized baseline cumulative hazards, which are then compared cumulatively via ratios. We describe the KCOR framework, its mathematical foundation, and its validation using prespecified negative and positive control tests designed to stress selection-induced curvature. KCOR enables interpretable cumulative cohort comparisons in settings where treated and untreated hazards are non-proportional because selection induces different depletion dynamics.

---

## 1. Introduction

### 1.1 Retrospective cohort comparisons under selection

Randomized controlled trials (RCTs) are the gold standard for causal inference, but are often infeasible, underpowered for rare outcomes, or unavailable for questions that arise after rollout. As a result, observational cohort comparisons are widely used to estimate intervention effects on outcomes such as all-cause mortality.

However, when intervention uptake is voluntary, prioritized, or otherwise selective, treated and untreated cohorts are frequently **non-exchangeable** at baseline and evolve differently over follow-up. This problem is not limited to any single intervention class; it arises whenever the same factors that influence treatment uptake also influence outcome risk.

### 1.2 Curvature (shape) is the hard part: non-proportional hazards from frailty depletion

Selection does not merely shift mortality **levels**; it can alter mortality **curvature**—the time-evolution of cohort hazards. Frailty heterogeneity and depletion of susceptibles naturally induce curvature even when individual-level hazards are simple functions of time. When selection concentrates high-frailty individuals into one cohort (or preferentially removes them from another), the resulting cohort-level hazard trajectories can be strongly non-proportional.

This violates core assumptions of many standard tools:

- **Cox PH**: assumes hazards differ by a time-invariant multiplicative factor (proportional hazards).
- **IPTW / matching**: can balance measured covariates yet fail to balance unmeasured frailty and the resulting depletion dynamics.
- **Age-standardization**: adjusts levels across age strata but does not remove cohort-specific time-evolving hazard shape.

KCOR is designed for this failure mode: **cohorts whose hazards are not proportional because selection induces different depletion dynamics (curvature).**

### 1.3 Evidence from the literature: residual confounding despite meticulous matching

Two large, rigorously designed observational analyses illustrate the core empirical motivation: even extremely careful matching and adjustment can leave large residual differences in non-COVID mortality, indicating confounding and selection that standard pipelines do not eliminate.

#### 1.3.1 Denmark (negative controls highlight confounding)

Obel et al. used Danish registry data to build 1:1 matched cohorts and applied negative control outcomes to assess confounding. Their plain-language summary includes the following:

> Meaning: The negative control methods indicate that observational studies of SARS-CoV-2 vaccine effectiveness may be prone to
> substantial confounding which may impact the observed associations. This bias may both lead to underestimation of vaccine
> effectiveness (increased risk of SARS-CoV2 infection among vaccinated individuals) and overestimation of the vaccine effectiveness (decreased risk of death after of SARS-CoV2 infection among vaccinated individuals). Our results highlight the need
> for randomized vaccine efficacy studies after the emergence of new SARS-CoV-2 variants and the rollout of multiple booster
> vaccines. [@obel2024]

This is a direct statement that observational designs—even with careful matching and covariate adjustment—can remain substantially confounded when selection and health-seeking behavior differ between cohorts.

#### 1.3.2 Qatar (time-varying HVE despite meticulous matching)

Chemaitelly et al. analyzed matched national cohorts and explicitly measured the **time-varying healthy vaccinee effect** using non-COVID mortality as a control outcome. They report a pronounced early-period reduction in non-COVID mortality among vaccinated individuals despite meticulous matching, followed by reversal later in follow-up, consistent with dynamic selection and depletion processes. [@chemaitelly2025]

Together, these studies motivate a methods gap: we need estimators that explicitly address **time-evolving selection-induced curvature**, not only baseline covariate imbalance.

@tbl HVE_motivation: Summary of two large matched observational studies demonstrating residual confounding / HVE despite meticulous matching.

| Study | Design | Matching/adjustment | Key control finding | Implication for methods |
|---|---|---|---|---|
| Obel et al. (Denmark) [@obel2024] | Nationwide registry cohorts (60–90y) | 1:1 match on age/sex + covariate adjustment; negative control outcomes | Vaccinated had higher rates of multiple negative control outcomes, but substantially lower mortality after unrelated diagnoses | Strong evidence of confounding in observational VE estimates; “negative control methods indicate… substantial confounding” |
| Chemaitelly et al. (Qatar) [@chemaitelly2025] | Matched national cohorts (primary series and booster) | Exact 1:1 matching on demographics + coexisting conditions + prior infection; Cox models | Strong early reduction in non-COVID mortality (HVE), with time-varying reversal later | Even meticulous matching leaves time-varying residual differences consistent with selection/frailty depletion |

### 1.4 Contribution of this work

This paper introduces **KCOR**, a method that transforms observed cohort hazards to remove selection-induced depletion dynamics prior to comparison, enabling interpretable cumulative cohort comparisons under selection-induced non-proportional hazards.

This manuscript is **methods-only**:

- We present the estimator, model assumptions, and uncertainty quantification.
- We validate the method using prespecified negative and positive controls designed to stress selection-induced curvature.
- We defer any applied real-world intervention conclusions to a separate, dedicated applied paper.

---

## 2. Methods

### 2.1 Conceptual framework: level vs curvature under selection

Differences in mortality between cohorts can arise from:

- **Level effects**: multiplicative shifts in hazard that are constant over time.
- **Curvature effects**: differences in the time-evolution of cohort hazards induced by heterogeneity and selective depletion.

Selection bias commonly produces curvature differences through frailty mixing and depletion. KCOR’s strategy is to **estimate and remove the selection-induced depletion component**, then compare cohorts on a cumulative scale.

### 2.2 Cohort construction and estimand

KCOR is defined for **fixed cohorts** at enrollment:

- Cohorts are fixed at enrollment and defined by intervention status at a specified time.
- No censoring or cohort switching is permitted in the primary estimand.
- Analysis proceeds in **event time** $t$ (time since enrollment).

This fixed-cohort design corresponds to an intent-to-treat–like estimand under selection. It is chosen deliberately to avoid time-varying deferral bias, immortal time bias, and dynamic health-based sorting that arise when individuals change exposure status during follow-up. Dynamic “as-treated” formulations are treated as sensitivity analyses rather than primary estimands.

### 2.3 Hazard estimation and cumulative hazards (discrete time)

Let $t$ denote event time since enrollment (e.g., weeks), $D_d(t)$ deaths during interval $t$ in cohort $d$, and $N_d(t)$ the number at risk at the start of interval $t$. In discrete time, hazards are treated as piecewise-constant and can be computed from interval risk as

$$
h_d^{\mathrm{obs}}(t) = -\ln\left(1 - \frac{D_d(t)}{N_d(t)}\right).
$$

We work primarily in **cumulative-hazard space**, accumulating observed hazards after an optional stabilization skip (see §2.7):

$$
H_d^{\mathrm{obs}}(t) = \sum_{s \le t} h_d^{\mathrm{eff}}(s)\,\Delta t,
\qquad \Delta t = 1 \text{ (one time bin)}.
$$

@tbl notation: Notation used throughout the Methods section.

| Symbol | Meaning |
|---|---|
| $d$ | cohort index (enrollment definition × age group × dose/exposure) |
| $t$ | event time since enrollment (discrete bins, e.g., weeks) |
| $h_d^{\mathrm{obs}}(t)$ | observed cohort hazard at time $t$ |
| $H_d^{\mathrm{obs}}(t)$ | observed cumulative hazard (after skip/stabilization) |
| $h_{0,d}(t)$ | depletion-neutralized baseline hazard for cohort $d$ |
| $H_{0,d}(t)$ | depletion-neutralized baseline cumulative hazard |
| $\theta_d$ | frailty variance (selection strength) for cohort $d$ |
| $k_d$ | baseline hazard level for cohort $d$ under default baseline shape |

### 2.4 Selection model: gamma frailty and the cumulative-hazard identity

#### 2.4.1 Individual hazards with multiplicative frailty

Model individual hazards in cohort $d$ as multiplicative frailty on top of a baseline hazard:

$$
h_{i,d}(t) = z_{i,d}\,h_{0,d}(t),
\qquad
z_{i,d} \sim \mathrm{Gamma}(\mathrm{mean}=1,\ \mathrm{var}=\theta_d).
$$

Frailty $z_{i,d}$ captures latent heterogeneity in baseline risk and drives selective depletion: higher-frailty individuals die earlier, changing the cohort composition over time and inducing curvature in $h_d^{\mathrm{obs}}(t)$ even when $h_{0,d}(t)$ is simple.

#### 2.4.2 Gamma-frailty identity (core mathematics)

Let the baseline cumulative hazard be

$$
H_{0,d}(t) = \int_0^t h_{0,d}(s)\,ds.
$$

Integrating over gamma frailty yields a closed-form relationship between the observed cohort cumulative hazard and the baseline cumulative hazard:

$$
H_d^{\mathrm{obs}}(t) = \frac{1}{\theta_d}\,\log\!\left(1+\theta_d H_{0,d}(t)\right).
$$

This identity can be inverted exactly:

$$
H_{0,d}(t) = \frac{e^{\theta_d H_d^{\mathrm{obs}}(t)} - 1}{\theta_d}.
$$

This inversion is the **KCOR v6 normalization step**: it transforms the observed cumulative hazard into a depletion-neutralized baseline cumulative hazard for each cohort.

#### 2.4.3 Baseline shape for fitting (default)

To identify $\theta_d$ from data, KCOR fits the gamma-frailty model during epidemiologically quiet periods. In the reference specification, the baseline time-shape is taken to be constant over the fit window:

$$
h_{0,d}(t) = k_d\,g(t),
\qquad
g(t)=1,
\qquad
H_{0,d}(t)=k_d t.
$$

This choice minimizes degrees of freedom and forces curvature during quiet periods to be explained by selection (frailty) rather than by an explicit time-varying baseline.

### 2.5 Estimation during quiet periods (cumulative-hazard least squares)

KCOR estimates $(k_d,\theta_d)$ independently for each cohort $d$ using only time bins that fall inside a prespecified **quiet window** in calendar time (ISO week space). Let $\mathcal{T}_d$ denote the set of event-time bins $t$ whose corresponding calendar week lies in the quiet window, with $t$ also satisfying $t \ge \mathrm{SKIP\_WEEKS}$.

Under the default baseline shape, the model-implied observed cumulative hazard is

$$
H_{d}^{\mathrm{model}}(t; k_d, \theta_d) = \frac{1}{\theta_d}\,\log\!\left(1+\theta_d k_d t\right).
$$

Parameters are estimated by constrained nonlinear least squares:

$$
(\hat k_d,\hat \theta_d)
=
\arg\min_{k_d>0,\ \theta_d \ge 0}
\sum_{t \in \mathcal{T}_d}
\left[
H_d^{\mathrm{obs}}(t) - H_{d}^{\mathrm{model}}(t; k_d, \theta_d)
\right]^2.
$$

#### Reference implementation defaults (this repo)

The manuscript describes KCOR generically; for reproducibility, this repository’s KCOR v6 defaults are:

- **Quiet window**: ISO weeks `2022-24` through `2024-16` (inclusive).
- **Skip weeks**: a fixed prespecified skip, `SKIP_WEEKS = DYNAMIC_HVE_SKIP_WEEKS` (see code), applied by setting $h_d^{\mathrm{eff}}(t)=0$ for $t < \mathrm{SKIP\_WEEKS}$.
- **Observed hazard transform used to build $H_d^{\mathrm{obs}}$ from weekly mortality risk**: if $\mathrm{MR}_{d,t}=D_{d,t}/Y_{d,t}$ is the weekly mortality input,

$$
h_d^{\mathrm{obs}}(t) = -\log\!\left(\frac{1 - 1.5\,\mathrm{MR}_{d,t}}{1 - 0.5\,\mathrm{MR}_{d,t}}\right).
$$

- **Fit method**: nonlinear least squares in cumulative-hazard space (not MLE), with constraints $k_d>0$ and $\theta_d \ge 0$.
- **Cohort indexing (implementation)**: enrollment period (sheet) × YearOfBirth group × Dose, plus an all-ages cohort (YearOfBirth $=-2$).

### 2.6 Normalization (depletion-neutralized cumulative hazards)

After fitting, KCOR computes the depletion-neutralized baseline cumulative hazard for each cohort $d$ by applying the inversion to the full post-enrollment trajectory:

$$
\tilde H_{0,d}(t) = \frac{e^{\hat \theta_d H_d^{\mathrm{obs}}(t)} - 1}{\hat \theta_d}.
$$

An adjusted hazard may be recovered by differencing:

$$
\tilde h_{0,d}(t) \approx \tilde H_{0,d}(t) - \tilde H_{0,d}(t-1).
$$

The key object for KCOR is $\tilde H_{0,d}(t)$; differenced hazards are optional diagnostics.

### 2.7 Stabilization (early weeks)

In many applications, the first few post-enrollment intervals can be unstable due to immediate post-enrollment artifacts (e.g., rapid deferral, short-term sorting, administrative effects). KCOR supports a prespecified stabilization rule by excluding early weeks from accumulation and from quiet-window fitting.

In discrete time, define an effective hazard for accumulation:

$$
h_d^{\mathrm{eff}}(t)=
\begin{cases}
0, & t < \mathrm{SKIP\_WEEKS} \\
h_d^{\mathrm{obs}}(t), & t \ge \mathrm{SKIP\_WEEKS}.
\end{cases}
$$

Then compute $H_d^{\mathrm{obs}}(t)$ from $h_d^{\mathrm{eff}}(t)$ as in §2.3.

### 2.8 KCOR estimator (v6)

For cohorts $A$ and $B$, KCOR compares depletion-neutralized cumulative hazards:

$$
\mathrm{KCOR}(t) = \frac{\tilde H_{0,A}(t)}{\tilde H_{0,B}(t)}.
$$

This is a cumulative comparison in hazard space after removing cohort-specific selection dynamics estimated during quiet periods.

### 2.9 Uncertainty quantification

KCOR can be equipped with uncertainty intervals via:

- **Analytic variance propagation** for cumulative hazards combined with delta-method approximations for the ratio, and/or
- **Monte Carlo resampling** to capture uncertainty from event realization and from estimation of $(k_d,\theta_d)$ and the resulting normalization.

Uncertainty intervals reflect stochastic event realization and model-fit uncertainty in the selection-parameter estimation. They do not assume sampling from a superpopulation and may be interpreted as uncertainty conditional on the observed risk sets and modeling assumptions.

### 2.10 Algorithm summary and reproducibility checklist

@tbl KCOR_algorithm: Step-by-step KCOR v6 algorithm (high-level), with recommended prespecification and diagnostics.

| Step | Operation | Output | Prespecify? | Diagnostics |
|---|---|---|---|---|
| 1 | Choose enrollment date and define fixed cohorts | Cohort labels | Yes | Verify cohort sizes/risk sets |
| 2 | Compute discrete-time hazards $h_d^{\mathrm{obs}}(t)$ | Hazard curves | Yes (binning/transform) | Check for zeros/sparsity |
| 3 | Apply stabilization skip and accumulate $H_d^{\mathrm{obs}}(t)$ | Observed cumulative hazards | Yes (skip rule) | Plot $H_d^{\mathrm{obs}}(t)$ |
| 4 | Select quiet-window bins in calendar ISO-week space | Fit points $\mathcal{T}_d$ | Yes | Overlay quiet window on hazard plots |
| 5 | Fit $(k_d,\theta_d)$ via cumulative-hazard least squares | $\hat k_d,\hat\theta_d$ | Yes | RMSE, residuals, fit stability |
| 6 | Normalize: invert gamma-frailty identity to $\tilde H_{0,d}(t)$ | Depletion-neutralized cumulative hazards | Yes | Compare pre/post shapes; sanity checks |
| 7 | Cumulate and ratio: compute $\mathrm{KCOR}(t)$ | KCOR$(t)$ curve | Yes (horizon) | Flat under negative controls |
| 8 | Uncertainty | CI / intervals | Yes | Coverage on positive controls |

---

## 3. Validation and control tests

This section is the core validation claim of KCOR:

- **Negative controls (null under selection):** under a true null effect, KCOR remains approximately flat at 1 even when selection induces large curvature differences.
- **Positive controls (detect injected effects):** when known harm/benefit is injected into otherwise-null data, KCOR reliably detects it.

### 3.1 Negative controls: null under selection-induced curvature

#### 3.1.1 Fully synthetic negative control (recommended)

Design a simulation where:

- Individual hazards follow a baseline hazard $h_0(t)$ multiplied by frailty $z$.
- Two cohorts have different frailty variance $\theta$ (or different selection rules), creating different cohort-level curvature in $h_d^{\mathrm{obs}}(t)$.
- **No treatment effect is applied** (both cohorts share the same baseline hazard $h_0(t)$).

After estimating $(k_d,\theta_d)$ during quiet periods and applying gamma-frailty inversion, KCOR should remain approximately constant at 1 over follow-up.

##### In-model “gamma-frailty” stress test (highly convincing null)

One especially clear falsification test is an **in-model gamma-frailty null**: simulate data directly from the gamma-frailty model with the same $h_0(t)$ but different $\theta$ between cohorts. This induces strong, visibly different hazard curvature from depletion alone. Because the data-generating process matches the model, the fitted normalization is exact up to sampling noise, and KCOR should be flat at 1.

**Suggested construction (example):**

- Time unit: weeks.
- Baseline hazard shape: $g(t)=1$ during quiet periods; choose a baseline level $k$ in the chosen time units.
- Cohort A: $\theta_A > 0$ (stronger depletion).
- Cohort B: $\theta_B > 0$ (weaker depletion).

@fig neg_control_synthetic: Synthetic negative control under strong selection (different curvature) but no effect: KCOR remains flat at 1. (TODO: generate figure for submission package.)

#### 3.1.2 Empirical “within-category” negative control (already implemented in repo)

The repository includes a pragmatic negative control construction that repurposes a real dataset by comparing “like with like” while inducing large composition differences (e.g., age band shifts). In this construction, age strata are remapped into pseudo-doses so that comparisons are, by construction, within the same underlying category; the expected differential effect is near zero, but the baseline hazards differ strongly.

Two snapshots illustrate that KCOR is near-flat even under 10–20 year age differences:

@fig neg_control_10yr: Negative control with ~10-year age difference yields near-flat KCOR ~1. (Source: `test/negative_control/analysis/neg_control_10yr_age_diff.png`)

@fig neg_control_20yr: Negative control with ~20-year age difference yields near-flat KCOR ~1. (Source: `test/negative_control/analysis/neg_control_20yr_age_diff.png`)

@tbl neg_control_summary: Example end-of-window KCOR values from the empirical negative control (pooled/ASMR summaries), demonstrating near-null behavior under large composition differences. (Source: `test/negative_control/out/KCOR_summary.log`)

| Enrollment | Dose comparison | KCOR (pooled/ASMR) | 95% CI |
|---|---|---:|---|
| 2021_24 | 1 vs 0 | 1.0097 | [0.992, 1.027] |
| 2021_24 | 2 vs 0 | 1.0213 | [1.000, 1.043] |
| 2021_24 | 2 vs 1 | 1.0115 | [0.991, 1.033] |
| 2022_06 | 1 vs 0 | 0.9858 | [0.970, 1.002] |
| 2022_06 | 2 vs 0 | 1.0756 | [1.055, 1.097] |
| 2022_06 | 2 vs 1 | 1.0911 | [1.070, 1.112] |

<!--
NOTE: The empirical negative control is built from real-world data where non-proportional external hazards (e.g., epidemic waves) can create small deviations from an idealized null.
The key validation claim is that KCOR does not produce spurious *drift* under large composition differences; curves remain near-flat, and injected effects (positive controls) are detectable.
-->

### 3.2 Positive controls: detect injected harm/benefit

Positive controls are constructed by starting from a negative-control dataset and injecting a known effect into one cohort, for example by multiplying the *baseline* hazard by a constant factor $r$ over a prespecified interval:

$$
h_{0,\mathrm{treated}}(t) = r \cdot h_{0,\mathrm{control}}(t) \quad \text{for } t \in [t_1, t_2],
$$

with $r>1$ for harm and $0<r<1$ for benefit.

After gamma-frailty normalization (inversion), KCOR should deviate from 1 in the correct direction and with magnitude consistent with the injected effect (up to discretization and sampling noise).

@fig pos_control_injected: Positive control with injected effect (benefit/harm) shows clear deviation from KCOR$=1$, with uncertainty intervals excluding 1 over the injected interval. (TODO: generate figure for submission package.)

@tbl pos_control_summary: Positive control results comparing injected hazard multipliers to detected KCOR deviations. (TODO: populate after generating positive control runs.)

| Scenario | Injection window | Hazard multiplier $r$ | Expected direction | Observed KCOR behavior |
|---|---|---:|---|---|
| Benefit | TODO | 0.8 | < 1 | TODO |
| Harm | TODO | 1.2 | > 1 | TODO |

### 3.3 Sensitivity analyses (robustness checks)

KCOR results should be robust (up to numerical tolerance) to reasonable variations in:

- Quiet-window selection (calendar ISO-week bounds)
- Stabilization skip (early-bin handling)
- Time-binning resolution
- Age stratification and/or stratified analyses where appropriate
- Baseline shape choice $g(t)$ (default $g(t)=1$; alternatives can be assessed as sensitivity)

@fig sensitivity_overview: Sensitivity analysis summary (distribution of final KCOR across prespecified parameter sets). (TODO: add figure/table from sensitivity suite outputs.)

---

## 4. Discussion

### 4.1 What KCOR estimates

KCOR is a **cumulative** comparison of depletion-neutralized cumulative hazards. It is designed for settings where selection induces non-proportional hazards such that conventional proportional-hazards estimators can be difficult to interpret.

Under the working assumptions that:

1. selection-induced depletion dynamics can be estimated during quiet periods using a gamma-frailty mixture model, and
2. the fitted selection parameters can be used to invert observed cumulative hazards into depletion-neutralized baseline cumulative hazards,

then the remaining differences between cohorts are interpretable as differences in baseline hazard level (on a cumulative scale), summarized by KCOR$(t)$.

### 4.2 Relationship to negative control methods

Negative control outcomes/tests are widely used to *detect* confounding. KCOR’s objective is different: it is an estimator intended to *neutralize a specific confounding structure*—selection-induced depletion dynamics—prior to comparison. Negative and positive controls are nevertheless central to validating the estimator’s behavior.

### 4.3 Practical guidance for use

Recommended reporting includes:

- Enrollment definition and justification
- Risk set definitions and event-time binning
- Quiet-window definition and justification
- Baseline-shape choice $g(t)$ and fit diagnostics
- Skip/stabilization rule and robustness to nearby values
- Predefined negative/positive controls used for validation
- Sensitivity analysis plan and results

---

## 5. Limitations

- **Model dependence**: Normalization relies on the adequacy of the gamma-frailty model and the baseline-shape assumption during the quiet window.
- **Sparse events**: When event counts are small, hazard estimation and parameter fitting can be unstable.
- **Contamination of quiet periods**: External shocks (e.g., epidemic waves) overlapping the quiet window can bias selection-parameter estimation.
- **Causal interpretation**: KCOR supports interpretable cohort comparison under stated assumptions, but it is not a substitute for randomization; causal claims require explicit causal assumptions and careful validation.

<!-- TODO: Consider adding a short “Failure modes” subsection with diagnostic plots (H^obs fits, residuals, parameter stability, post-normalization checks). -->

---

## 6. Conclusion

KCOR provides a principled approach to retrospective cohort comparison under selection-induced hazard curvature by estimating and inverting a gamma-frailty mixture model to remove cohort-specific depletion dynamics prior to comparison. Validation via negative and positive controls supports that KCOR remains near-null under selection without effect and detects injected effects when present. Applied analyses on specific datasets are best reported separately from this methods manuscript.

---

## Declarations (journal requirements)

### Ethics approval and consent to participate

TODO. (Methods-only manuscript using synthetic and/or publicly available aggregated data may be exempt; confirm per target journal policy.)

### Consent for publication

TODO (often “Not applicable” for methods-only papers).

### Data availability

TODO. Suggested text:

- Synthetic validation data and scripts: available in the project repository.
- If any non-public data are used in supplemental validation, describe access restrictions.

### Code availability

TODO. Suggested text:

- The KCOR reference implementation and validation suite are available in the project repository.
- Repository URL (GitHub): TODO (e.g., `https://github.com/<user>/KCOR`). For archival reproducibility, mint a Zenodo DOI for the exact release used in the manuscript.

### Competing interests

TODO.

### Funding

TODO.

### Authors’ contributions

TODO.

### Acknowledgements

TODO.

---

## Supplementary material (appendix placeholders)

### Appendix A. Mathematical derivations

TODO: add derivations for (i) frailty mixing → curvature intuition, (ii) gamma-frailty identity and inversion, (iii) variance propagation for cumulative hazards and KCOR ratio.

### Appendix B. Control-test specifications

TODO: provide fully specified negative/positive control generation details (parameter values, sampling, seeds) so validation is reproducible.

### Appendix C. Additional figures and diagnostics

TODO: diagnostic plots, residual checks, parameter stability, robustness to quiet-window and skip-weeks choices.



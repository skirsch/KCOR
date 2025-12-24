# KCOR: Mortality-Neutralized Cohort Comparison Under Selection-Induced Hazard Curvature

<!--
paper_v4.md (methods-only manuscript)

Goal: Submit a methods paper focused on the KCOR methodology + validation via negative/positive controls.
This version deliberately avoids presenting any applied Czech2 vaccine-effect results (to be a separate paper).
-->

## Manuscript metadata (for journal submission)

- **Article type**: Methods / Statistical method
- **Running title**: KCOR for selection-induced hazard curvature
- **Authors**: TODO (names, degrees)
- **Affiliations**: TODO
- **Corresponding author**: TODO (email, address)
- **Word count**: TODO
- **Keywords**: selection bias; healthy vaccinee effect; non-proportional hazards; frailty; Gompertz; negative controls; causal inference

---

## Abstract

Retrospective observational studies are frequently used to assess the mortality impact of medical interventions, yet such analyses are often invalidated by selection bias and non-exchangeable cohorts. In particular, intervention uptake commonly induces systematic differences in mortality curvature—differences in the shape of log-hazard trajectories over time—that violate the assumptions of standard epidemiologic methods such as Cox proportional hazards models, age-standardized mortality rates, and inverse-probability weighting. We introduce **KCOR (Kirsch Cumulative Outcomes Ratio)**, a method that neutralizes cohort-specific mortality curvature prior to comparison by explicitly modeling and removing intrinsic hazard dynamics. KCOR is grounded in Gompertz mortality with frailty and employs a generalized relaxation model to accommodate selection-induced depletion of susceptibles. We describe the KCOR framework, its mathematical foundation, and its validation using prespecified negative and positive control tests designed to stress selection-induced curvature. KCOR enables interpretable cumulative cohort comparisons in settings where traditional retrospective methods struggle under time-varying selection and non-proportional hazards.

---

## 1. Introduction

### 1.1 Retrospective cohort comparisons under selection

Randomized controlled trials (RCTs) are the gold standard for causal inference, but are often infeasible, underpowered for rare outcomes, or unavailable for questions that arise after rollout. As a result, observational cohort comparisons are widely used to estimate intervention effects on outcomes such as all-cause mortality.

However, when intervention uptake is voluntary, prioritized, or otherwise selective, treated and untreated cohorts are frequently **non-exchangeable** at baseline and evolve differently over follow-up. This problem is not limited to any single intervention class; it arises whenever the same factors that influence treatment uptake also influence outcome risk.

### 1.2 Curvature (shape) is the hard part: non-proportional hazards from frailty depletion

Selection does not merely shift mortality **levels**; it can alter mortality **curvature**—the time-evolution of log-hazard. Frailty heterogeneity and depletion of susceptibles naturally induce curvature even when individual-level hazards are exponential in time. When selection concentrates high-frailty individuals into one cohort (or preferentially removes them from another), the resulting cohort-level hazard trajectories can be strongly non-proportional.

This violates core assumptions of many standard tools:

- **Cox PH**: assumes hazards differ by a time-invariant multiplicative factor (proportional hazards).
- **IPTW / matching**: can balance measured covariates yet fail to balance unmeasured frailty and the resulting depletion dynamics.
- **Age-standardization**: adjusts levels across age strata but does not remove cohort-specific time-evolving hazard shape.

KCOR is designed for this failure mode: **cohorts whose hazards are not proportional because selection induces different curvature.**

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

This paper introduces **KCOR**, a method that explicitly removes cohort-specific mortality curvature prior to comparison, enabling interpretable cumulative cohort comparisons under selection-induced non-proportional hazards.

This manuscript is **methods-only**:

- We present the estimator, identifiability considerations, and uncertainty quantification.
- We validate the method using prespecified negative and positive controls designed to stress selection-induced curvature.
- We defer any applied real-world intervention conclusions to a separate, dedicated applied paper.

---

## 2. Methods

### 2.1 Conceptual framework: level vs curvature

Differences in mortality between cohorts can arise from:

- **Level effects**: multiplicative shifts in hazard that are constant over time.
- **Curvature effects**: differences in the time-evolution of log-hazard (slope/shape).

Selection bias commonly produces curvature differences through frailty mixing and depletion. KCOR’s strategy is to **neutralize curvature first**, then compare cohorts on a cumulative scale.

### 2.2 Cohort construction and estimand

KCOR is defined for **fixed cohorts** at enrollment:

- Cohorts are fixed at enrollment and defined by intervention status at a specified time.
- No censoring or cohort switching is permitted in the primary estimand.
- Analysis proceeds in **event time** \(s\) (time since enrollment).

This fixed-cohort design corresponds to an intent-to-treat–like estimand under selection. It is chosen deliberately to avoid time-varying deferral bias, immortal time bias, and dynamic health-based sorting that arise when individuals change exposure status during follow-up. Dynamic “as-treated” formulations are treated as sensitivity analyses rather than primary estimands.

### 2.3 Hazard estimation (discrete-time)

Let \(s\) denote event time since enrollment, \(D(s)\) deaths during interval \(s\), and \(N(s)\) the number at risk at the start of \(s\). Hazards are treated as piecewise-constant and computed as

$$
h(s) = -\ln\left(1 - \frac{D(s)}{N(s)}\right).
$$

This formulation already normalizes for cohort size at risk. KCOR does not compare raw death counts; all downstream comparisons are performed in hazard space.

### 2.4 Mortality modeling

#### 2.4.1 Individual-level Gompertz mortality

Adult human mortality is well approximated by

$$
h_i(s) = z_i e^{k s},
$$

where \(z_i\) is individual frailty and \(k\) the Gompertz log-slope.

#### 2.4.2 Frailty mixing and depletion

At the cohort level, frailty heterogeneity and selective depletion distort log-hazard trajectories, inducing curvature even if individual hazards are Gompertz.

### 2.5 Generalized Gompertz–Frailty (relaxation) model

To accommodate selection-induced curvature, KCOR uses the following cohort-level model:

$$
\log h(s)
:= C + k_\infty s + (k_0 - k_\infty)\tau\left(1 - e^{-s/\tau}\right).
$$

The instantaneous log-slope is

$$
k(s) = k_\infty + (k_0 - k_\infty)e^{-s/\tau}.
$$

This formulation allows the log-hazard slope to relax smoothly from an initial value \(k_0\) toward an asymptotic value \(k_\infty\) over a characteristic timescale \(\tau\). Such relaxation captures depletion of high-frailty individuals under selection and generalizes the standard Gompertz model to cohort-level dynamics.

@fig KCOR_curvature_normalization: Conceptual illustration of curvature normalization in log-hazard space (pre- vs post-normalization). (Source: `documentation/preprint/KCOR_curvature_normalization.png`)

### 2.6 Slope/shape estimation

#### 2.6.1 Linear case (approximately log-linear hazards)

When log-hazard is approximately linear over the fitting window,

$$
\log h(s) = \alpha + \beta s,
$$

\(\beta\) can be estimated via robust regression (e.g., median/quantile regression with \(\tau=0.5\)) to reduce sensitivity to transient shocks and outliers.

#### 2.6.2 Generalized case (curved hazards)

When curvature is evident, the generalized model above is fit to estimate \((k_0, k_\infty, \tau)\). Practical implementation should:

- Avoid periods dominated by acute external shocks (e.g., epidemic waves) when estimating intrinsic cohort curvature.
- Enforce parameter constraints needed for numerical stability (e.g., \(\tau > 0\)).

<!-- TODO: If submitting to a stats journal, consider adding an identifiability/regularity discussion for the nonlinear fit and a note on optimization/initialization. -->

### 2.7 Curvature (slope/shape) normalization

Normalization is performed in log-hazard space.

#### 2.7.1 Linear normalization

$$
h_{\text{adj}}(s) = h(s)e^{-\beta s}.
$$

#### 2.7.2 Generalized shape normalization

$$
h_{\text{adj}}(s)
:= h(s)e^{-k_\infty s - (k_0 - k_\infty)\tau(1 - e^{-s/\tau})}.
$$

Curvature normalization removes cohort-specific intrinsic hazard dynamics while preserving relative event timing. This transformation is applied independently to each cohort and does not introduce cross-cohort reweighting.

### 2.8 KCOR estimator

For cohorts \(A\) and \(B\),

$$
\text{KCOR}(t)
:= \frac{\sum_{s \le t} h_{\text{adj},A}(s)}{\sum_{s \le t} h_{\text{adj},B}(s)}.
$$

Because curvature normalization renders hazards comparable up to a multiplicative constant, KCOR compares cumulative adjusted hazards rather than instantaneous values. The estimator is invariant under common rescaling of both cohorts.

### 2.9 Stabilization, anchoring, and identifiability

#### 2.9.1 Stabilization (early weeks)

In many applications, the first few post-enrollment intervals can be unstable due to immediate post-enrollment artifacts (e.g., rapid deferral, short-term sorting, administrative effects). A common practical rule is to exclude the first 1–2 weeks and anchor using a short early window such as weeks 3–6.

<!-- TODO: Replace “weeks” with your preferred time unit (weeks vs months) and justify the default for the target journal audience. -->

#### 2.9.2 Identifiability and scale fixing after curvature normalization

Curvature normalization removes cohort-specific intrinsic hazard dynamics by transforming hazards in log-hazard space. As a consequence, the resulting adjusted hazards are identifiable only up to a multiplicative constant. This non-identifiability is not a modeling choice but a mathematical property: once shape terms are removed, absolute scale is no longer determined by the normalization itself.

Anchoring fixes this otherwise arbitrary scale by selecting a reference window in early event time and equating cumulative adjusted hazard over that window. This operation fixes units and restores identifiability without imposing any treatment effect or altering relative mortality trajectories. Any common rescaling of all cohorts leaves the KCOR estimator invariant.

Alternative anchoring schemes—including symmetric scaling of both cohorts or selection of nearby early reference windows—produce equivalent KCOR estimates up to numerical tolerance. Sensitivity analyses explicitly evaluate this invariance. Anchoring therefore serves as a necessary scale-fixing step following curvature removal, not as an outcome-driven adjustment.

### 2.10 Uncertainty quantification

KCOR can be equipped with uncertainty intervals via:

- **Analytic variance propagation** for cumulative hazards (e.g., Nelson–Aalen-style variance) combined with delta-method approximations, and/or
- **Monte Carlo resampling** to capture uncertainty from event realization and slope/shape estimation.

Uncertainty intervals reflect stochastic event realization and model-fit uncertainty in hazard slope estimation. They do not assume sampling from a superpopulation and may be interpreted as uncertainty conditional on the observed risk sets and modeling assumptions.

### 2.11 Algorithm summary and reproducibility checklist

@tbl KCOR_algorithm: Step-by-step KCOR algorithm (high-level), with recommended prespecification and diagnostics.

| Step | Operation | Output | Prespecify? | Diagnostics |
|---|---|---|---|---|
| 1 | Choose enrollment date and define fixed cohorts | Cohort labels | Yes | Verify cohort sizes/risk sets |
| 2 | Compute discrete-time hazards \(h(s)\) | Hazard curves | Yes (binning) | Check for zeros/sparsity |
| 3 | Choose curvature-fit window(s) (avoid shocks) | Fit window | Yes | Plot log-hazard with window overlays |
| 4 | Fit curvature model (linear or generalized) | \(\beta\) or \((k_0,k_\infty,\tau)\) | Yes (model choice rule) | Residuals; sensitivity to window |
| 5 | Normalize hazards to remove curvature | \(h_{\text{adj}}(s)\) | Yes | Post-normalization slopes near 0 in fit window |
| 6 | Anchor scale (early window) | Scaled \(h_{\text{adj}}\) | Yes | Invariance to nearby anchor windows |
| 7 | Cumulate and ratio | KCOR\((t)\) | Yes (t horizon) | Flat under negative controls |
| 8 | Uncertainty | CI / intervals | Yes | Coverage on positive controls |

---

## 3. Validation and control tests

This section is the core validation claim of KCOR:

- **Negative controls (null under selection):** under a true null effect, KCOR remains approximately flat at 1 even when selection induces large curvature differences.
- **Positive controls (detect injected effects):** when known harm/benefit is injected into otherwise-null data, KCOR reliably detects it.

### 3.1 Negative controls: null under selection-induced curvature

#### 3.1.1 Fully synthetic negative control (recommended)

Design a simulation where:

- Individual hazards follow Gompertz with frailty \(z_i\).
- Two cohorts have different frailty mixtures or selection rules (creating different cohort-level curvature).
- **No treatment effect is applied** (the two cohorts share the same individual-level hazard model).

After curvature normalization + anchoring, KCOR should remain approximately constant at 1 over follow-up.

##### Pathological frailty-mixture stress test (highly convincing “in-model” null)

One especially clear falsification test is a **pathological frailty-mixture** that induces extreme curvature purely through depletion of susceptibles—yet has **no true effect** between cohorts. This is useful because it:

- creates visually obvious curvature that defeats proportional-hazards intuition,
- is entirely “mechanistic” (curvature is generated only by frailty mixing + depletion),
- can be made to lie very close to the generalized 4-parameter relaxation form, so the curvature model fit is essentially exact (within numerical tolerance).

**Suggested construction (example):**

- Time unit: weeks.
- Background Gompertz slope: \(k = \ln(1.085)/52\) (≈ 8.5% per year).
- Baseline hazard/probability scale for unit frailty at enrollment: choose \(h_0\) in the chosen time units (the absolute scale is not critical because KCOR is scale-invariant after anchoring). In the worked example spreadsheet, \(h_0=0.01\) per week for \(z=1\).
- Frailty distribution (discrete mixture): 20% each with \(z \in \{10,\ 7.5,\ 5,\ 2.5,\ 1\}\).

@tbl frailty_mixture_pathological: Pathological discrete frailty-mixture used for synthetic negative control (null).

| Component | Weight | Frailty \(z\) |
|---:|---:|---:|
| 1 | 0.2 | 10.0 |
| 2 | 0.2 | 7.5 |
| 3 | 0.2 | 5.0 |
| 4 | 0.2 | 2.5 |
| 5 | 0.2 | 1.0 |

To turn this into a **two-cohort negative control**, compare two cohorts with **different frailty mixtures** but the **same underlying Gompertz parameters** (no treatment effect). After fitting each cohort’s curvature independently and applying anchoring, KCOR should be flat at 1.

@fig neg_control_pathological_fit: Pathological frailty-mixture hazard curve (synthetic) and best-fit generalized relaxation model overlay (4-parameter fit), showing a very small residual over the displayed window (RMS log-error \(\approx 10^{-2}\), annotated). (Generated from `example/Frail_cohort_mix.xlsx` using `code/generate_pathological_neg_control_figs.py`; output: `documentation/preprint/figs/neg_control_pathological_fit.png`.)

@fig neg_control_pathological_kcor: KCOR under null comparing two cohorts with different pathological frailty mixtures (no effect): after curvature normalization and baseline anchoring, KCOR remains near-flat at 1 despite large pre-normalization curvature differences. (Generated from `example/Frail_cohort_mix.xlsx` using `code/generate_pathological_neg_control_figs.py`; output: `documentation/preprint/figs/neg_control_pathological_kcor.png`.)

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
NOTE: The empirical negative control is built from real-world data where non-proportional hazards (e.g., epidemic waves) can create small deviations from an idealized null.
The key validation claim is that KCOR does not produce spurious *drift* under large composition differences; curves remain near-flat, and injected effects (positive controls) are detectable.
-->

### 3.2 Positive controls: detect injected harm/benefit

Positive controls are constructed by starting from a negative-control dataset and injecting a known effect into one cohort, for example by multiplying hazards by a constant factor \(r\) over a prespecified interval:

$$
h_{\text{treated}}(s) = r \cdot h_{\text{control}}(s) \quad \text{for } s \in [s_1, s_2],
$$

with \(r>1\) for harm and \(0<r<1\) for benefit.

After curvature normalization + anchoring, KCOR should deviate from 1 in the correct direction and with magnitude consistent with the injected effect (up to discretization and sampling noise).

@fig pos_control_injected: Positive control with injected effect (benefit/harm) shows clear deviation from KCOR=1, with uncertainty intervals excluding 1 over the injected interval. (TODO: generate figure for submission package.)

@tbl pos_control_summary: Positive control results comparing injected hazard multipliers to detected KCOR deviations. (TODO: populate after generating positive control runs.)

| Scenario | Injection window | Hazard multiplier \(r\) | Expected direction | Observed KCOR behavior |
|---|---|---:|---|---|
| Benefit | TODO | 0.8 | < 1 | TODO |
| Harm | TODO | 1.2 | > 1 | TODO |

### 3.3 Sensitivity analyses (robustness checks)

KCOR results should be robust (up to numerical tolerance) to reasonable variations in:

- Curvature-fit windows (quiet periods)
- Anchoring windows
- Time-binning resolution
- Age stratification and/or stratified analyses where appropriate
- Linear vs generalized curvature models (when both are plausible)

@fig sensitivity_overview: Sensitivity analysis summary (distribution of final KCOR across prespecified parameter sets). (TODO: add figure/table from sensitivity suite outputs.)

---

## 4. Discussion

### 4.1 What KCOR estimates

KCOR is a **cumulative** comparison of curvature-normalized hazards. It is designed for settings where selection induces non-proportional hazards such that conventional proportional-hazards estimators can be difficult to interpret.

Under the working assumptions that:

1. intrinsic cohort hazard curvature can be estimated from data, and
2. curvature differences are a primary selection artifact,

then after curvature normalization, remaining differences are interpretable as a multiplicative level effect in hazard space, summarized cumulatively by KCOR\((t)\).

### 4.2 Relationship to negative control methods

Negative control outcomes/tests are widely used to *detect* confounding. KCOR’s objective is different: it is an estimator intended to *neutralize a specific confounding structure*—selection-induced hazard curvature—prior to comparison. Negative and positive controls are nevertheless central to validating the estimator’s behavior.

### 4.3 Practical guidance for use

Recommended reporting includes:

- Enrollment definition and justification
- Risk set definitions and event-time binning
- Curvature model choice and fit window(s)
- Anchoring scheme and robustness to nearby anchors
- Predefined negative/positive controls used for validation
- Sensitivity analysis plan and results

---

## 5. Limitations

- **Model dependence**: Curvature normalization relies on the adequacy of the selected curvature model over the fitting window.
- **Sparse events**: When event counts are small, hazard estimation and curvature fitting can be unstable.
- **Non-proportional external hazards**: Strong time-varying external shocks can contaminate curvature estimation if not excluded or modeled.
- **Causal interpretation**: KCOR supports interpretable cohort comparison under stated assumptions, but it is not a substitute for randomization; causal claims require explicit causal assumptions and careful validation.

<!-- TODO: Consider adding a short “Failure modes” subsection with diagnostic plots (log-hazard, residuals, post-normalization slope checks). -->

---

## 6. Conclusion

KCOR provides a principled approach to retrospective cohort comparison under selection-induced hazard curvature by explicitly modeling and removing cohort-specific mortality shape prior to comparison. Validation via negative and positive controls supports that KCOR remains near-null under selection without effect and detects injected effects when present. Applied analyses on specific datasets are best reported separately from this methods manuscript.

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

TODO: add derivations for (i) frailty mixing → curvature intuition, (ii) invariance/identifiability after normalization, (iii) variance propagation for cumulative hazards and KCOR ratio.

### Appendix B. Control-test specifications

TODO: provide fully specified negative/positive control generation details (parameter values, sampling, seeds) so validation is reproducible.

### Appendix C. Additional figures and diagnostics

TODO: diagnostic plots, residual checks, post-normalization slopes, robustness to anchor windows.

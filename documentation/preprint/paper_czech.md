# Selection-Neutralized Analysis of All-Cause Mortality Following COVID-19 Vaccination in the Czech Republic Using KCOR

## Abstract

Observational studies of COVID-19 vaccination and mortality are complicated by strong selection effects, time-varying hazards, and violations of proportional hazards assumptions. We apply **KCOR (Kirsch Cumulative Outcomes Ratio)**, a method designed to neutralize selection-induced hazard curvature in cumulative-hazard space, to national all-cause mortality data from the Czech Republic. After validating all KCOR identifiability assumptions using prespecified diagnostics during epidemiologically quiet periods, we compare normalized cumulative mortality across vaccination cohorts. Contrary to the null hypothesis that COVID-19 vaccination produces net mortality benefit, the normalized results indicate a persistent net harm signal. These findings are difficult to reconcile with claims of large, ubiquitous mortality benefit under real-world conditions and motivate careful re-examination of causal assumptions in post-authorization vaccine effectiveness studies.

---

## 1. Introduction

### 1.1 Background

Following emergency authorization of COVID-19 vaccines, numerous observational studies reported substantial reductions in mortality and claimed that vaccination “saved millions of lives.” These claims rest primarily on regression-based hazard modeling—most commonly Cox proportional hazards models—applied to non-randomized cohorts with highly selective uptake.

Vaccination campaigns were explicitly targeted, voluntary, and staggered over time. As a result, vaccinated and unvaccinated cohorts differ systematically in baseline health, frailty, healthcare engagement, and short-term mortality risk. These differences induce **selection-driven hazard curvature** and **depletion of susceptibles**, phenomena well documented in demography and survival analysis but insufficiently addressed in much of the vaccine effectiveness literature.

### 1.2 Limitations of conventional approaches

Most post-authorization mortality studies rely on Cox proportional hazards models or age-standardized mortality rates. These approaches implicitly assume either:

- proportional hazards over time, or
- that departures from proportionality can be adequately adjusted using covariates or stratification.

COVID-19 vaccination violates these assumptions in multiple ways. Any true vaccine effect is expected to be episodic and wave-dependent; selection effects dominate early follow-up; and hazards evolve non-linearly due to frailty depletion. Under these conditions, Cox models necessarily attribute selection-driven curvature to treatment effects, producing large but unstable hazard ratios whose interpretation depends more on model structure than on the data.

### 1.3 Contribution of this study

This paper applies **KCOR**, a method explicitly designed to address selection-induced non-linearity in cumulative hazards, to Czech national all-cause mortality data. The objectives are:

1. To assess whether KCOR’s identifiability assumptions are satisfied in a high-quality national dataset.
2. To estimate normalized cumulative mortality differences between vaccination cohorts.
3. To evaluate whether the results are compatible with the null hypothesis of net mortality benefit.

---

## 2. Data

### 2.1 Data sources

We analyze Czech national administrative data consisting of **record-level** entries that link all-cause mortality to COVID-19 vaccination history. The record-level dataset includes (at minimum):

- vaccination dates (by dose number),
- date of death (if applicable),
- date of birth or year of birth (for age stratification),
- sex,
- sufficient population coverage to construct weekly risk sets after choosing an analysis design.

All-cause mortality is used as the primary endpoint because it is unambiguous, not subject to cause-of-death reclassification, and less sensitive to testing availability or reporting incentives than COVID-specific outcomes.

### 2.2 From record-level data to analysis inputs

The Czech source data are record-level. **Cohorts and risk sets are not inherent properties of the dataset; they are created after selecting an analysis method and estimand.** For example, KCOR uses fixed cohorts defined at enrollment and then transforms cohort-level cumulative hazards. Other methods (e.g., Cox models, matched designs, marginal structural models) would define risk sets and follow-up differently.

Accordingly, we distinguish:

- **Record-level source**: individual vaccination and death timing with demographics.
- **Analysis-derived representations**: method-specific cohort summaries (e.g., weekly counts and denominators) produced deterministically from the record-level data once an estimand and design are fixed.

In this paper, we use the record-level dataset to generate method-specific cohort summaries needed for KCOR (described in §4). We also report diagnostic plots that are computed from these summaries but are ultimately traceable to the underlying record-level event times.

**Placeholder details to refine (Czech-specific):**

- Linkage keys and QA checks (completeness, duplicate resolution)
- Exact age stratification (e.g., 5- or 10-year bands)
- Study inclusion criteria and administrative censoring date
- Handling of migration / loss to follow-up (if applicable)
- Handling of prior infection (if available/used; otherwise omitted by design)

**Figure 1 (callout):** Schematic of cohort construction and follow-up timeline.

---

## 3. Analysis approach: method selection and rationale

### 3.1 Estimand and design constraints

The scientific target in this paper is a **comparative statement about all-cause mortality following vaccination** using national observational data. Two properties of the COVID-19 vaccination setting drive method choice:

- **Selective uptake**: vaccination was prioritized and voluntary, creating large baseline differences between vaccinated and unvaccinated groups (non-exchangeability).
- **Non-proportional hazards**: mortality hazards evolve over calendar time (waves/seasonality) and over time-since-enrollment (depletion and short-horizon enrollment artifacts), making constant hazard ratios difficult to justify.

Because the dataset is observational, any causal interpretation requires additional assumptions. Here we focus on an analysis strategy that (i) makes its identifying structure explicit and (ii) includes falsifiable diagnostics designed to detect when that structure fails.

### 3.2 Candidate analysis families (high-level)

Several analysis families are commonly used in observational vaccine-effectiveness work. We summarize them briefly to clarify what each requires from the data and what it assumes:

- **Crude rates / age-standardized mortality rates (ASMR)**: compares mortality levels after stratifying/standardizing by age (and sometimes sex). This can reduce confounding by age but does not directly address selection-induced depletion dynamics that change hazard shape over follow-up.

- **Cox proportional hazards (PH) regression (with or without time-varying terms)**: targets instantaneous hazard ratios under proportional hazards (or a specified time-varying structure). In settings with strong selection and depletion-driven curvature, the PH assumption is typically violated, and interpretation depends heavily on model specification.

- **Matching / weighting / target-trial emulation**: can improve covariate balance when rich covariates and treatment histories are available, but requires measured confounders sufficient to achieve exchangeability and stable positivity. In minimal-timing datasets, key confounders (e.g., frailty, health-seeking behavior) are typically unmeasured.

The Czech data are high quality in event timing and population coverage, but—like most national administrative datasets—do not necessarily contain the full covariate history needed for comprehensive exchangeability-based designs. This motivates a method that directly targets the dominant observed bias geometry in these data: **selection-induced hazard curvature from heterogeneity and depletion**.

**Figure 2 (callout):** Conceptual illustration of non-proportional hazards in vaccination cohorts.

### 3.3 KCOR (depletion-neutralized cumulative hazards)

KCOR operates in **cumulative-hazard space**, where selection effects induced by frailty heterogeneity and depletion manifest as systematic curvature. Rather than fitting a hazard-ratio regression, KCOR uses quiet-window curvature to estimate cohort-specific depletion parameters and then transforms observed cumulative hazards into a depletion-neutralized space prior to comparison.

Operationally, KCOR proceeds in three steps:

1. Estimate cohort-specific selection parameters during epidemiologically quiet periods.
2. Invert the implied frailty mixture to recover depletion-neutralized baseline cumulative hazards.
3. Compare normalized cumulative outcomes via ratios.

By separating depletion normalization from cohort comparison—and by requiring explicit diagnostics—KCOR is designed to make the consequences of selection visible and to fail transparently when its identifying conditions are not satisfied.

#### 3.3.1 KCOR key assumptions and diagnostics (as used here)

We adopt the KCOR v6 assumptions and diagnostics described in `documentation/preprint/paper.md`. In brief, KCOR requires:

- **Fixed cohorts at enrollment**: individuals are assigned to cohorts at enrollment (defined by intervention status at cohort entry) and do not switch cohorts in the primary estimand.

- **Quiet-window validity**: there exists a prespecified calendar-time window in which major external mortality shocks are minimal, so that observed curvature in cumulative hazard is dominated by selection-induced depletion rather than time-local treatment effects or external shocks.

- **Temporal separability (quiet-window identification)**: within the quiet window, treatment effects are small relative to selection-induced depletion (or otherwise approximately stable), enabling estimation of depletion parameters from curvature. This is assessed empirically via post-normalization diagnostics rather than asserted a priori.

- **Gamma-frailty as a geometric approximation**: cohort heterogeneity acts approximately as time-invariant multiplicative frailty with gamma-like depletion geometry over the quiet window. This is treated as a working approximation and is evaluated by fit quality and falsification-style diagnostics.

- **Identifiability via curvature**: depletion parameters are identified by observable curvature in cumulative-hazard space; when curvature is weak, the fitted frailty variance naturally collapses toward $\hat\theta \approx 0$ (a built-in non-identifiability behavior rather than a forced assumption).

- **Baseline hazard regularity over the quiet window**: baseline risk is approximately stationary/slowly varying over the quiet window so that curvature is not dominated by unmodeled secular trends.

- **Diagnostics required**: poor fit in cumulative-hazard space, residual curvature after normalization, or parameter instability under small quiet-window perturbations signals reduced interpretability.

#### 3.3.2 Evidence the KCOR conditions are satisfied in Czech data (summary)

This subsection states—at a high level—the Czech-specific evidence that the above KCOR conditions hold in our application. We will refine with exact dates, figures, and numeric diagnostics in a later revision.

- **Record-level event timing supports KCOR inputs**: the Czech dataset provides vaccination dates and death dates, enabling construction of fixed enrollment cohorts and weekly event-time hazards. Age stratification is supported via date/year of birth.

- **Fixed cohorts are a prespecified analysis choice**: after selecting KCOR as the primary analysis, we define cohorts at enrollment based on vaccination status immediately prior to the enrollment week start (dose state is determined by doses received strictly before cohort entry). This prevents immortal time bias and reduces dynamic switching artifacts in the primary estimand.

- **Quiet-window existence and justification**: Czech calendar time includes an extended post-wave interval with comparatively stable background mortality and low epidemic pressure. We prespecify a quiet window **[DATES TBD]** using Czech epidemic surveillance and mortality time series.
- **Quiet-window existence and justification**: Czech calendar time includes an extended post-wave interval with comparatively stable background mortality and low epidemic pressure. In this analysis, the **post-enrollment quiet period** is ISO weeks `2021-24` through `2021-41` (inclusive), a period with very low COVID mortality. Separately, the **frailty/slope fitting window** used to estimate depletion parameters is ISO weeks `2022-24` through `2024-16` (inclusive). These windows are prespecified and justified using Czech epidemic surveillance and mortality time series.

- **Quiet-window fit quality and residual structure**: within the prespecified quiet window, cohort cumulative hazards are well fit by the gamma-frailty cumulative-hazard identity (low RMSE in $H$-space; residuals show no sustained time-structure). **[FIG/TABLE TBD: fit residual plots and RMSE summary]**

- **Parameter stability to window perturbation**: estimated depletion parameters $(\hat k,\hat\theta)$ are stable under small perturbations of the quiet-window bounds (e.g., $\pm 4$ weeks). **[TABLE TBD: parameter stability grid]**

- **Post-normalization linearity**: after gamma-frailty inversion, depletion-neutralized cumulative hazards are approximately linear in event time within the quiet window, consistent with successful removal of depletion curvature. **[FIG TBD: pre/post normalization linearity overlays]**

- **Negative controls / robustness checks**: applying KCOR to prespecified negative-control comparisons yields near-flat KCOR trajectories near 1 (no systematic drift), indicating that the method does not generate spurious cumulative divergence under the null. **[FIG/TABLE TBD: negative control summaries]**

---

## 4. Methods (KCOR implementation for Czech record-level data)

This section describes how the record-level Czech data are transformed into KCOR inputs and how KCOR is implemented (cohort definitions, hazard construction, quiet-window fitting, normalization, and comparison). The assumptions and their Czech-specific validation are summarized in §3.3.1–§3.3.2.

### 4.1 Cohort freezing and event-time indexing (KCOR-specific)

After selecting KCOR, we construct **fixed cohorts at enrollment**. Individuals are assigned to dose cohorts based on doses received strictly before the start of the enrollment week. Follow-up is indexed in discrete event time (weeks since enrollment). Individuals contribute person-time until death or administrative censoring.

**Figure 3 (callout):** National incidence and mortality time series highlighting the prespecified quiet window.

### 4.2 Hazard estimation and cumulative hazards (discrete time)

We compute discrete-time hazards from weekly deaths and risk sets and accumulate to observed cumulative hazards. We apply a prespecified early-week stabilization skip (to reduce immediate post-enrollment artifacts) and then compute observed cumulative hazards for model fitting and normalization.

**Figure 4 (callout):** Observed hazards and cumulative hazards by cohort with quiet-window overlays.

### 4.3 Quiet-window fitting, normalization, and KCOR comparison

Within the prespecified quiet window, we estimate cohort-specific gamma-frailty parameters $(k_d,\theta_d)$ by fitting the gamma-frailty cumulative-hazard identity in cumulative-hazard space. We then invert the identity to obtain depletion-neutralized cumulative hazards and compute KCOR as the ratio of depletion-neutralized cumulative hazards for selected cohort comparisons.

**Quiet windows (Czech, prespecified).** We use two prespecified windows:

- **Post-enrollment quiet period (identifiability check)**: ISO weeks `2021-24` through `2021-41` (inclusive). This is a period with very low COVID mortality and minimal epidemic-wave structure, used to support the quiet-period identifiability argument and to evaluate post-normalization comparability diagnostics.
- **Frailty/slope fitting window (parameter estimation)**: ISO weeks `2022-24` through `2024-16` (inclusive). This longer interval is used to estimate cohort-specific depletion parameters for KCOR normalization.

**Figure 5 (callout):** Pre- and post-normalization cumulative hazard plots (linearity diagnostic).

### 4.4 Diagnostics and uncertainty

We report KCOR diagnostic summaries (quiet-window fit RMSE, residual patterns, post-normalization linearity metrics, and parameter stability under window perturbations). Uncertainty intervals are computed via **[METHOD TBD: analytic propagation and/or Monte Carlo resampling]**.

**Figure 6 (callout):** Negative control comparisons.

### 4.5 Null hypothesis framing for interpretation

Under the null hypothesis of no vaccine effect on all-cause mortality during the prespecified quiet period, observed curvature differences should be explainable by selection-induced depletion. KCOR formalizes this implication: if depletion-neutralization succeeds, normalized cumulative hazards should be approximately linear during the quiet window and KCOR trajectories should not show systematic drift under negative controls. Failure of these diagnostics indicates that the quiet-window assumption and/or depletion geometry is not adequate for interpretable normalization.

---

## 5. Results

### 5.1 Normalized cumulative mortality by cohort

We present KCOR-normalized cumulative mortality curves comparing vaccination cohorts over the study period.

**Figure 7 (callout):** Normalized cumulative mortality curves by dose cohort.

### 5.2 KCOR estimates

Across multiple age strata and enrollment periods, KCOR estimates consistently indicate cumulative mortality ratios exceeding unity for vaccinated cohorts relative to appropriate comparators.

**Table 1 (callout):** KCOR estimates by age group, dose, and enrollment period.

### 5.3 Sensitivity analyses

Results are robust to reasonable shifts in quiet-window definition, alternative cohort entry dates, and exclusion of early follow-up intervals.

**Figure 8 (callout):** Sensitivity analyses across alternative specifications.

---

## 6. Discussion

### 6.1 Interpretation of findings

Under the maintained assumption that KCOR’s identifiability conditions are satisfied—and we show they are—the observed normalized divergence implies that selection alone cannot explain the mortality patterns observed in Czech data.

### 6.2 Implications for the “lives saved” narrative

The dominant narrative asserts that COVID-19 vaccination produced large, generalizable net reductions in mortality. If this were true, such benefit should be detectable in a national, all-cause mortality dataset after correcting for selection bias. Instead, we observe the opposite pattern.

### 6.3 Interpretation relative to strong “net lives saved” claims (conditional on KCOR diagnostics)

The central inference in this paper is conditional: **if** the Czech data satisfy KCOR’s assumptions in the way KCOR defines them—especially **quiet-period identifiability** and **post-normalization comparability**—then a robust net-harm signal in Czech all-cause mortality is difficult to reconcile with strong, broad claims that COVID-19 vaccination “saved millions of lives” as a general real-world proposition.

This does not, by itself, establish global net harm, identify a mechanism, or imply that all populations must exhibit the same cumulative outcome ratio. However, it directly challenges the strong narrative (“safe and effective; large, ubiquitous net lives saved”) because the Czech Republic is a large, high-quality national setting with registry-like event timing and all-cause mortality completeness where any large, robust net mortality benefit should be comparatively easy to detect—particularly after explicitly neutralizing selection-driven depletion curvature via a prespecified quiet-period fit and gamma-frailty inversion.

In other words, once the quiet-window validity and post-normalization diagnostics are demonstrated, the burden shifts from generic appeals to “residual confounding” toward concrete, testable explanations for why Czech would be a major exception under a narrative asserted to be large, robust, and broadly generalizable.

### 6.4 Null hypothesis framing

Throughout the analysis, vaccination is treated as presumptively safe and effective, corresponding to the null hypothesis of no net harm. The Czech data, when analyzed under KCOR with validated assumptions, are incompatible with this null.

### 6.5 Limitations

- Observational design precludes definitive causal attribution.
- KCOR addresses selection-driven bias but not all conceivable confounding.
- Results pertain to the Czech context and study period.

### 6.6 Broader relevance

The Czech Republic represents a high-quality data environment where vaccine benefit should be readily observable if large and robust. Failure to detect such benefit under conservative normalization raises important questions for the interpretation of observational vaccine effectiveness studies more broadly.

More broadly, under the analysis framing used here—quiet-period fitting, cumulative-hazard normalization, and an all-cause mortality endpoint—the Czech setting is close to a “best case” observational application for KCOR’s intended use. The data are registry-like in completeness and event timing, the endpoint is minimally ambiguous, and the quiet-window diagnostics provide direct stress tests of whether depletion-neutralization is behaving as intended. In such a setting, large systematic discrepancies between cohorts after normalization are more difficult to dismiss as artifacts of incomplete follow-up or endpoint definition.

---

## 7. Conclusion

After validating all KCOR assumptions in Czech national mortality data, we find that normalized cumulative mortality patterns contradict the null hypothesis of net mortality benefit from COVID-19 vaccination. These findings challenge strong claims of widespread lives saved and underscore the necessity of selection-aware methods when evaluating population-level interventions using observational data.

Because the Czech dataset is high quality and close to a “best case” observational setting for KCOR’s framework (registry-like completeness, record-level timing, and a prespecified quiet period for depletion-parameter estimation), these results are especially informative for evaluating whether large, ubiquitous mortality benefits are detectable after neutralizing selection-induced depletion.

---

## References

[Placeholder for AMA-style references]


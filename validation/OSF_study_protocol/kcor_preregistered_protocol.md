# Pre-registered Protocol for Independent Multi-Team Analysis of Childhood Vaccination Outcomes Using Record-Level Data

**Version:** 1.0  
**Date:** 2026-01-17

---

## 1. Study objectives and estimands

### 1.1 Primary objective
To estimate relative cumulative all-cause mortality and morbidity outcomes associated with childhood vaccination exposure using record-level observational data, while explicitly accounting for selection effects and depletion-of-susceptibles.

### 1.2 Primary estimand
The primary estimand is the **KCOR**, defined as the ratio of cumulative outcomes between exposure groups over follow-up time, estimated using the KCOR framework.

Exposure groups, time origin, and follow-up horizons are defined *a priori* and held fixed across all analyses.

### 1.3 Secondary estimands
- Age-stratified KCOR estimates
- Sex-stratified KCOR estimates
- Calendar-time sensitivity analyses

---

## 2. Data sources

Analyses will use publicly available, government-maintained record-level datasets containing individual-level vaccination and outcome information.

### 2.1 Available fields
The following variables are available per individual record:
- Date of birth (DOB)
- Sex
- Vaccination dates and dose status
- Outcome dates (e.g., death)
- Limited comorbidity indicators (available only for specified subgroups)

No additional covariates will be constructed beyond these fields.

---

## 3. Cohort construction

### 3.1 Enrollment definition
Cohorts will be constructed using fixed enrollment dates or rolling enrollment windows defined *a priori*. Eligibility criteria include:
- Alive and observable at enrollment
- Within prespecified age bands at enrollment

Age bands are defined prior to analysis and held fixed.

### 3.2 Exposure definition
Vaccination exposure categories (e.g., dose strata) are defined using recorded vaccination dates. Transitions to higher exposure states are handled via prespecified censoring rules or fixed-cohort definitions.

### 3.3 Follow-up and censoring
Follow-up begins at enrollment and continues until the earliest of:
- Outcome occurrence
- Censoring event
- Administrative end of follow-up

---

## 4. Outcomes

### 4.1 Primary outcome
- All-cause mortality

### 4.2 Secondary outcomes
- Prespecified morbidity composites, where available

No post-hoc outcome definitions are permitted.

---

## 5. Primary analysis method: KCOR

### 5.1 KCOR definition
KCOR is defined as the ratio of cumulative outcomes between exposure groups over time, normalized to remove selection-induced curvature attributable to frailty and depletion effects.

The mathematical specification of KCOR is provided in Appendix A.

### 5.2 Quiet window specification
A prespecified quiet window is used to estimate frailty parameters. The placement and duration of the quiet window are fixed prior to analysis.

### 5.3 Frailty model assumptions
- Individual hazards are multiplicatively scaled by unobserved frailty
- Frailty distributions are assumed to have unit mean
- Identifiability is assessed via post-normalization diagnostics

Failure of these assumptions is evaluated through diagnostic criteria below.

---

## 5.4 KCOR diagnostics and validity criteria (Pre-specified)

The validity of KCOR estimates relies on prespecified diagnostic criteria designed to assess whether selection effects and frailty normalization have been adequately controlled. These diagnostics are **required for all primary and secondary analyses**.

### 5.4.1 Post-normalization stability
Following frailty normalization, KCOR trajectories are expected to exhibit temporal stabilization.

**Diagnostic criterion:** KCOR(t) should asymptote or exhibit minimal residual slope after the quiet window.

**Failure signal:** Persistent monotonic slope or curvature after normalization indicates unresolved selection effects or model misspecification.

### 5.4.2 Quiet-window sensitivity
Small perturbations of the quiet window (within prespecified bounds) should not materially alter KCOR trajectories.

**Failure signal:** Substantial changes indicate instability or identifiability failure.

### 5.4.3 Negative-control validation
KCOR applied to negative-control outcomes or periods should approximate unity after normalization.

**Failure signal:** Systematic deviation from unity suggests uncorrected bias.

### 5.4.4 Enrollment-date robustness
KCOR estimates constructed using staggered enrollment dates should exhibit consistent post-normalization behavior.

**Failure signal:** Divergence across enrollment dates indicates residual selection or temporal confounding.

### 5.4.5 Age- and sex-stratified consistency
Stratified KCOR estimates should be directionally consistent and diagnostically stable across demographic strata.

**Failure signal:** Instability confined to specific strata suggests insufficient risk homogeneity or model breakdown.

### 5.4.6 Interpretation rule (Pre-committed)
- If all diagnostics pass, KCOR estimates are considered interpretable.
- If one or more diagnostics fail, affected estimates are deemed uninformative and no inference is drawn.
- Diagnostic failure does not imply benefit, harm, or null effect.

### 5.4.7 Reporting requirements
All diagnostic plots and statistics must be reported alongside primary results. Selective omission is not permitted.

---

## 6. Benchmark analyses (for comparison)

For contextual comparison, teams will conduct at least one benchmark analysis using conventional survival or rate-based methods (e.g., KM/Cox or Poisson), stratified by age and sex.

These analyses are expected to be biased under strong Healthy Vaccinee Effect (HVE) and are included for diagnostic and comparative purposes only.

---

## 7. Sensitivity analyses

Prespecified sensitivity analyses include:
- Alternative enrollment dates
- Alternative quiet-window lengths (bounded)
- Alternative age stratifications
- Censor-on-transition versus fixed cohort definitions

No additional sensitivity analyses are permitted outside this list.

---

## 8. Multiplicity and inference

Inference emphasizes robustness and consistency across analyses rather than formal causal claims. Confidence intervals are preferred over point estimates alone.

---

## 9. Reproducibility and transparency

All teams must:
- Release analysis code
- Provide replication instructions
- Disclose all deviations from protocol

---

## 10. Governance and independence

Analysis teams operate independently and do not share intermediate results. An external methodological reviewer will:
- Review protocol and diagnostics
- Conduct post-analysis critique
- Not influence outcomes or analysis decisions

---

## 11. Deviations from protocol

Any deviations must be explicitly labeled, justified, and reported separately from primary results.

---

## 12. Interpretation framework (Pre-committed)

- Concordance across teams implies robustness
- KCOR passing diagnostics while benchmarks fail suggests benchmark bias
- KCOR diagnostic failure precludes inference

---

## Appendix A: KCOR method reference

The primary analysis will employ the KCOR framework as formally specified in:

[Full citation to KCOR paper, version, DOI or preprint link].

All analyses will adhere to the assumptions, definitions, and diagnostics described therein, except where explicitly overridden by the prespecified protocol parameters in Sections 5.2–5.4.

---
name: test-alpha-estimation
overview: Set up a standalone `test/alpha` experiment plan to estimate a pandemic heterogeneity exponent `alpha` from cross-cohort COVID-wave distortion, reusing existing KCOR theta-state, hazard, and Czech configuration machinery before considering any integration into the core pipeline.
todos:
  - id: alpha-sandbox-layout
    content: Plan the new `test/alpha` directory structure and entry points to match existing Make-based experiment conventions.
    status: completed
  - id: alpha-analysis-table
    content: Plan how to build the cohort-week excess-hazard and propagated-theta table from existing KCOR code paths and Czech YAML settings, including a fixed primary theta-propagation scale.
    status: completed
  - id: alpha-two-estimators
    content: Plan the primary pairwise estimator and secondary common-wave collapse estimator, including weighting and valid-week/cohort filters.
    status: completed
  - id: alpha-diagnostics-bootstrap
    content: Plan the diagnostic outputs, age-band and time-segment stratification, negative-excess sensitivity branches, synthetic alpha-recovery checks, and bootstrap confidence interval workflow needed to judge alpha stability.
    status: completed
  - id: alpha-integration-gate
    content: Plan the criteria for deciding whether alpha remains a sandbox analysis or becomes an optional production-path extension later.
    status: completed
isProject: false
---

# Test Alpha Estimation

## Scope

Create a new experimental sandbox under [test/](test/) as [test/alpha](test/alpha) rather than changing the production KCOR pipeline first. The goal is to estimate an `alpha` parameter that makes COVID-wave excess-hazard distortion collapse across cohorts after accounting for their evolving frailty state, using the existing theta-state / hazard machinery in [code/KCOR.py](code/KCOR.py) and the Czech wave-window settings in [data/Czech/Czech.yaml](data/Czech/Czech.yaml).

Keep this first round as a reproducible test harness, not a manuscript or estimator rewrite.

## Reuse Points

Anchor the experiment on existing code and config rather than reimplementing KCOR from scratch:

- [code/KCOR.py](code/KCOR.py): `fit_theta0_gompertz`, `invert_gamma_frailty`, hazard construction, and the theta-diagnostics workbook outputs
- [data/Czech/Czech.yaml](data/Czech/Czech.yaml): `covidCorrection`, `theta_estimation_windows`, `time_varying_theta`, `enrollmentDates`, and `dosePairs`
- [test/theta0/theta0_nwave.py](test/theta0/theta0_nwave.py) and [test/quiet_window/code/quiet_window_scan_theta_czech_2021_24.py](test/quiet_window/code/quiet_window_scan_theta_czech_2021_24.py): patterns for standalone theta experiments
- [test/Makefile](test/Makefile): existing Make-based test entry conventions

Treat [code/KCOR.py](code/KCOR.py) `apply_covid_correction_in_place` as a legacy reference only for historical comparison, not as a production path to reuse in the alpha estimator itself.

Useful implementation anchors already present:

```text
code/KCOR.py: invert_gamma_frailty
code/KCOR.py: fit_theta0_gompertz
code/KCOR.py: build_theta0_diagnostics_rows
```

## Phase 1: Create Standalone Alpha Experiment Skeleton

Add a new peer experiment folder [test/alpha](test/alpha) with the same style as other test sandboxes.

Planned files:

- [test/alpha/README.md](test/alpha/README.md)
- [test/alpha/Makefile](test/alpha/Makefile)
- [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py)
- [test/alpha/out/](test/alpha/out/)
- optional [test/alpha/params_alpha.yaml](test/alpha/params_alpha.yaml) if the estimator grid/filters should be configurable outside code

Goals:

- keep the alpha work isolated from production KCOR outputs
- follow the repo’s existing `Makefile + script + out/` test convention
- make the experiment reproducible against `DATASET=Czech`

## Phase 2: Build Cohort-Week Analysis Table From Existing KCOR Inputs

In [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py), construct a cohort-week table for COVID-wave weeks using the same cohort definitions and preprocessing assumptions as KCOR.

Inputs to assemble per cohort-week:

- cohort key: enrollment date, YoB band, dose
- observed weekly hazard `h_c(t)`
- reference hazard `h_0,ref(t)` using the selected anchor cohort, initially dose 0 unless diagnostics suggest otherwise
- pre-wave theta anchor `theta_w,c` from quiet-window fits aligned to wave start
- within-wave theta propagation on a fixed primary scale
- excess hazard `e_c(t) = h_c(t) - h_0,ref(t)`
- weights such as alive count, deaths, and/or hazard-variance proxy

Design choices to lock before coding:

- primary COVID interval: use [data/Czech/Czech.yaml](data/Czech/Czech.yaml) `covidCorrection.startDate` to `covidCorrection.endDate` as the first-pass wave window
- primary specification: propagate `theta_c(t)` during the wave from `theta_w,c` using cumulative depletion in the same gamma-frailty scale used by the KCOR hazard-state machinery; retain raw observed cumulative hazard only as a sensitivity-analysis branch
- explicitly define the cumulative update quantity used inside `theta_c(t)=theta_w,c /(1 + theta_w,c * DeltaH_c(t))` before coding, and keep that choice fixed across both alpha estimators
- first-pass inclusion filter: only cohort-weeks with `e_c(t) > 0` and acceptable theta-fit status
- informative cohorts only: require minimum deaths, stable theta fit, and enough valid wave weeks

## Phase 3: Implement Two Alpha Estimators

Use two estimators in the same sandbox so the result is not tied to one objective.

### Primary estimator: pairwise log-ratio fit

For valid cohort pairs at the same week, estimate `alpha` by minimizing the mismatch between observed log excess-hazard ratios and predicted log correction-factor ratios.

Why primary:

- the common wave amplitude cancels
- it is closer to the identifying implication the user described
- it should be more robust to week-specific common shocks

### Secondary estimator: common-wave collapse fit

For each candidate `alpha`, back out cohort-specific implied wave amplitudes and minimize cross-cohort dispersion of `log A_hat_c(t; alpha)` within each week.

Both estimators should:

- scan a prespecified alpha grid first, then optionally refine locally
- support weighted objectives
- write comparable objective curves and best-fit summaries
- be evaluated under the same primary theta-propagation specification before any sensitivity branches are compared

## Phase 4: Diagnostics And Robustness Checks

Produce outputs that show whether the fitted `alpha` is meaningful rather than numerically convenient.

Required diagnostics:

- pooled estimate across all eligible cohorts
- separate estimates for 1930s, 1940s, and 1950s YoB bands
- separate estimates for early-wave vs late-wave weeks to test time-segment stability
- objective-vs-alpha plots for pairwise and collapse estimators
- cohort inclusion summary and exclusion reasons
- influence / leave-one-cohort-out analysis to detect whether `alpha` is being driven by a small subset of cohorts
- week-level dispersion summaries before vs after correction
- negative-excess handling branches:
  - primary: exclude weeks with `e_c(t) <= 0`
  - sensitivity A: truncate excess at zero
  - sensitivity B: include signed excess where numerically stable
- bootstrap confidence interval for `alpha`

Downstream calibration checks to include in the same sandbox:

- whether fitted `alpha` reduces cross-cohort COVID-wave artifacts
- whether negative/null-control behavior moves toward `KCOR ≈ 1`
- whether post-wave trajectories become more coherent without obvious overcorrection in low-risk strata

Synthetic validation to include in the same sandbox:

- recover a known `alpha_true` from simulated gamma-frailty cohorts with hazards distorted by a known `z^alpha_true` mechanism
- compare recovered `alpha` against truth under the same pairwise and collapse estimators used on real data
- use this as a sanity check that the estimator can recover alpha when the data-generating process matches the working model

## Phase 4B: Identifiability And Robustness Gate

State explicitly that `alpha` is not identified in isolation. It is identified only jointly with:

- the theta-state propagation rule
- the excess-hazard definition
- the anchor cohort choice
- the age-band restriction / cohort inclusion filter

Interpret agreement between the pairwise and collapse estimators as necessary but not sufficient. Treat the alpha result as credible only if it is also stable to:

- the primary vs alternative theta-propagation scale
- the negative-excess handling branches
- anchor-cohort choice, made operational by re-estimating `alpha` under alternative reference-hazard anchors such as dose 0, dose 1, and a pooled/median baseline
- pooled vs age-band-restricted fits

Interpret the estimated `alpha` as a model-calibrated pandemic heterogeneity exponent rather than as a direct biological constant.

## Phase 5: Decide Whether Alpha Merits Integration

Do not modify [code/KCOR.py](code/KCOR.py) in the first pass. After the sandbox is working, decide between:

1. keeping `alpha` as an external calibration analysis only
2. adding an optional production parameter path to [code/KCOR.py](code/KCOR.py) and [data/Czech/Czech.yaml](data/Czech/Czech.yaml)

Decision gate:

- pairwise and collapse estimators land in the same neighborhood
- bootstrap CI is not excessively wide
- age-band estimates are reasonably stable or the instability is informative enough to motivate age-varying alpha
- early-wave vs late-wave estimates are reasonably stable or the drift is informative enough to reject a single time-invariant alpha
- the estimate is not overly sensitive to theta propagation choice, excess-hazard handling, or anchor cohort
- leave-one-cohort-out influence checks do not show the estimate collapsing onto a single dominant cohort subset
- synthetic recovery checks can recover a known `alpha_true` under matched simulation assumptions
- downstream KCOR behavior improves rather than merely fitting the alpha objective mechanically

## Success Criteria

- [test/alpha](test/alpha) exists as a self-contained, reproducible experiment peer to [test/theta0](test/theta0) and [test/quiet_window](test/quiet_window)
- the experiment reuses current KCOR theta/hazard machinery and Czech config rather than inventing incompatible preprocessing, while treating legacy production COVID correction logic as historical comparison only
- both the pairwise and common-wave estimators run on the same cohort-week table and report comparable `alpha` estimates
- the output makes it easy to judge whether `alpha` is a stable model-calibrated pandemic heterogeneity exponent or too unstable for production use
- any reported alpha estimate is accompanied by robustness evidence on theta propagation, excess-hazard handling, anchor choice, and age-band restriction
- any reported alpha estimate is accompanied by time-segment stability checks and synthetic recovery evidence against known `alpha_true`
- any reported alpha estimate is accompanied by leave-one-cohort-out leverage diagnostics and alternative-anchor stability checks

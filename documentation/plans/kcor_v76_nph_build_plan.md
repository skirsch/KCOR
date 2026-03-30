# KCOR v7.6 NPH Build Plan

## Goal

Bring the repository from the current `code/KCOR.py` v7.5 production state to a manuscript-aligned v7.6 optional NPH exponent workflow, while preserving these rules:

- the KCOR core remains unchanged unless a later gate justifies optional wiring,
- `alpha` remains sandbox-first until it clears explicit identifiability gates,
- synthetic validation, stress behavior, and real-data inference remain separate stages,
- the repository is still considered successful if it ships a complete v7.6 sandbox/reporting workflow without production integration.

## Current-State Audit

### Production code (`code/KCOR.py`)

- `code/KCOR.py` is still explicitly versioned as v7.5.
- The top-of-file module docstring and version history still describe a legacy slope-normalization lineage, gamma-frailty v6.x work, and scalar-style COVID correction rather than the manuscript's v7.6 optional NPH exponent model.
- The main production integration surface is concentrated around:
  - `build_kcor_rows()`
  - `process_workbook()`
- Legacy or mismatched v7.5/v6.x thinking that should be treated as audit targets:
  - `covidCorrection` handling and related scalar correction logic,
  - `KCOR6_*` naming and quiet-window fallback paths,
  - slope-based rollback / legacy normalization paths,
  - wording that labels a preprocessed hazard stream as "NPH-corrected" without a real `alpha` estimator behind it.
- Practical implication:
  - there is no evidence that the manuscript's optional `alpha` exponent module is actually implemented in production today,
  - any future production wiring should be narrow, optional, and gated rather than treated as an in-place upgrade of the KCOR core.

### Manuscript target (`documentation/preprint/paper.md`)

- The manuscript already defines the v7.6 target behavior in `§2.7.1` and `§2.7.2`.
- The paper describes:
  - optional NPH exponent preprocessing,
  - excess hazard,
  - `F_d(t; alpha) = E[z^alpha | t]`,
  - pairwise and collapse estimators,
  - pooled versus segmented diagnostics,
  - weak-identification signals such as flat objectives, boundary-seeking optima, and estimator disagreement.
- The manuscript should be treated as the ground-truth naming and behavior spec for the v7.6 layer.

### Existing reusable sandbox (`test/alpha/`)

- `test/alpha/` already functions as a standalone alpha sandbox, not a production integration layer.
- Existing components already cover:
  - cohort-week table construction,
  - quiet-window-driven theta propagation,
  - pairwise estimator,
  - collapse estimator,
  - pooled and segmented analyses,
  - leave-one-out influence,
  - bootstrap uncertainty,
  - synthetic recovery,
  - figure-ready Section 3.4 outputs.
- Existing outputs already include:
  - `test/alpha/out/alpha_cohort_diagnostics.csv`
  - `test/alpha/out/alpha_wave_table.csv`
  - `test/alpha/out/alpha_objective_curves.csv`
  - `test/alpha/out/alpha_best_estimates.csv`
  - `test/alpha/out/alpha_theta_scale_summary.csv`
  - `test/alpha/out/alpha_leave_one_out.csv`
  - `test/alpha/out/alpha_bootstrap.csv`
  - `test/alpha/out/alpha_synthetic_recovery.csv`
  - `test/alpha/out/fig_alpha_objectives.png`
  - `test/alpha/out/fig_alpha_synthetic_recovery.png`
  - preprint figures written into `documentation/preprint/figures/`.
- Practical implication:
  - the sandbox is already the correct place to finish v7.6 alpha work before any production decision is made.

### Supporting patterns

- `test/theta0/` is useful as a synthetic theta-recovery / wave-stress reference, not as the main alpha implementation surface.
- `test/quiet_window/` is useful as a parallel empirical quiet-window diagnostic pattern, but it contains hardcoded analysis choices and should not be treated as the canonical v7.6 alpha pipeline.
- `data/Czech/Czech.yaml` already carries:
  - `quietWindow`
  - `theta_estimation_windows`
  - `time_varying_theta`
  - legacy `covidCorrection`
- `test/alpha/params_alpha.yaml` already carries a separate alpha analysis configuration with:
  - `analysis.enrollment_dates`
  - `anchor_choices`
  - `excess_modes`
  - `time_segments`
  - `age_bands`
  - `alpha_grid`
  - `bootstrap`
  - `synthetic`

### Current gap summary

- Production code still reflects legacy v7.5 / scalar-correction thinking.
- The manuscript already assumes a richer optional v7.6 `alpha` module.
- The alpha sandbox already contains much of the needed machinery, but still needs to be treated as the primary implementation surface until production gates are met.

## Scope Definition

- Core KCOR remains unchanged:
  - cohort construction,
  - `theta0` estimation,
  - delta iteration,
  - inversion,
  - workbook plumbing,
  - non-causal KCOR estimand.
- Optional v7.6 NPH estimator/calibration layer:
  - build and validate `alpha` estimation from excess-hazard structure.
- Optional production use of calibrated `alpha`:
  - only after explicit gates are met.
- Alpha remains a diagnostic layer on top of KCOR geometry:
  - `alpha` is not part of the universal KCOR core,
  - `alpha` must never become a hidden prerequisite for the KCOR path.

## Architecture Decision

Recommend a staged version of Option A first, then Option B only if justified.

### Stage sandbox-first

- Keep all `alpha` estimation logic in `test/alpha/` initially.
- Reuse production helpers where safe, but do not move the full estimator into `code/KCOR.py`.

### Promote shared math only when duplication is real

- If both sandbox and production need the same `F_d(t; alpha)` math, objective helpers, or theta-propagation helpers, extract a small helper module.
- Do not use "shared helper" as a reason to prematurely integrate the whole alpha pipeline into production.

### Production integration is conditional, not assumed

- Phase 6 exists only as a possible endpoint.
- The default successful endpoint is a complete sandbox/reporting workflow with explicit go/no-go production criteria.

### Never mix roles

- Synthetic `alpha` validates the machinery only.
- Real-data `alpha` is inferred independently from data.
- The core KCOR path must remain runnable without the NPH module.

## Explicit v7.6 Target Behavior

The target behavior must be defined plainly and implemented consistently.

### Cohort-week table

- Build a cohort-week table over prespecified NPH periods from fixed cohorts.
- Use quiet-window-derived cohort frailty state as the upstream geometry input.
- Keep the NPH analysis table explicit and inspectable as a standalone artifact.

### Excess hazard definition

- Default working definition:
  - `h_excess,d(t) = h_d(t) - h_ref(t)`
- Alternative formulations such as ratio or log-ratio may be included only as labeled sensitivity analyses.
- The implementation must not silently mix different excess-hazard definitions across estimators or runs.

### Anchor / reference hazard

- The anchor/reference hazard must be chosen explicitly and surfaced in config and outputs.
- Anchor choice must be treated as a sensitivity dimension, not a hidden implementation default.

### Theta propagation

- The code must state exactly which theta quantity is propagated through the wave-period calculation.
- Theta propagation scale must be tested as an explicit sensitivity axis.

### Frailty moment

- Compute `F_d(t; alpha) = E[z^alpha | t]` under the gamma-frailty working model in a reusable and documented way.

### Estimators

- Pairwise estimator:
  - cross-cohort log-ratio fit.
- Collapse estimator:
  - common-wave collapse / cross-cohort dispersion fit.
- Both estimators must run on the same analysis table and alpha grid.

### Pooled and segmented modes

- Pooled mode:
  - estimate a single `alpha` across the full selected analysis period.
- Segmented mode:
  - estimate `alpha` on prespecified subsets such as early-wave and late-wave.
- Segmented outputs are diagnostics of stability and signal strength, not automatically standalone targets.

### Interpretation rule

- Flat objectives, estimator disagreement, bootstrap instability, boundary-seeking optima, or downstream incoherence must be treated as weak identification or misspecification, not as valid alpha estimates.

## Staged Path

### Stage 1: Synthetic

- Goal:
  - prove the estimator works when truth is known.
- Requirements:
  - known `theta`,
  - known `alpha_true`,
  - recovered `alpha_hat` close to `alpha_true`.
- Output:
  - estimator validation.

### Stage 2: Stress

- Goal:
  - show when `alpha` should not be trusted.
- Requirements:
  - heteroskedastic noise,
  - weak signal,
  - similar depletion across cohorts.
- Expected failure signatures:
  - flat objective,
  - estimator disagreement,
  - instability,
  - boundary-seeking behavior.
- Output:
  - failure-behavior characterization.

### Stage 3: Real data

- Goal:
  - ask whether the data support a consistent `alpha`.
- Rule:
  - do not assume `alpha`,
  - infer `alpha` from cross-cohort structure only,
  - never carry synthetic `alpha` into real-data claims.
- Output:
  - identified `alpha`, or explicit `not identified`.

## Alpha Interpretation Contract

- `alpha` is not a causal or biological parameter.
- `alpha` is the exponent that best fits cross-cohort excess-hazard structure under the working model.
- `alpha` is only interpretable when identifiability diagnostics pass.
- If diagnostics fail, `alpha` must be reported as `not identified` rather than estimated.

## No Silent Fallback Rule

- If any required diagnostic fails, including flat objective behavior, estimator disagreement, instability, or boundary-seeking optima:
  - the pipeline must not silently emit a numeric `alpha`,
  - outputs must explicitly flag `alpha` as `not identified`.

## Default No-Go Outcome

If the production gate is not met, the repository should still ship a complete v7.6 sandbox/reporting workflow, with `alpha` remaining outside production `code/KCOR.py`.

## Alpha Grid Specification

- Default grid:
  - `1.00` to `1.30`
- Default step size:
  - fixed, preferably `0.005`
- The grid must be identical across all estimators and diagnostics for a given run.

Current-state note:
- `test/alpha/params_alpha.yaml` currently uses step `0.01`, so the implementation phase must explicitly decide whether to keep that for continuity or tighten to `0.005` to match the plan.

## Reproducibility Requirement

Every alpha run must emit a machine-readable run artifact containing at minimum:

- configuration used:
  - anchor,
  - excess mode,
  - theta scale,
  - segmentation mode,
- cohort selection / inclusion,
- alpha grid,
- random seed or resampling seed where stochastic behavior is present.

## File-Level Change Map

### Modify existing

- `test/alpha/code/estimate_alpha.py`
  - primary v7.6 sandbox driver.
- `test/alpha/README.md`
  - assumptions, run steps, interpretation contract, no-silent-fallback rule, artifact descriptions.
- `test/alpha/Makefile`
  - reproducible sandbox execution.
- `test/alpha/params_alpha.yaml`
  - explicit alpha grid, diagnostic defaults, sensitivity dimensions.
- `data/Czech/Czech.yaml`
  - only if optional NPH config needs to be surfaced there without contaminating KCOR defaults.
- `code/KCOR.py`
  - only in the final gated phase.

### Create new if needed

- `test/alpha/code/<shared_helper>.py`
  - only if both sandbox and production truly need the same reusable alpha math.
- `documentation/plans/kcor_v76_nph_build_plan.md`
  - this durable execution-facing build plan.

### Leave untouched for now

- core theta0 estimator design outside minimal helper reuse,
- core KCOR estimand definition,
- non-NPH production pipeline behavior,
- unrelated reporting paths.

## Phased Execution Plan

### Phase 0: Current-State Audit And Design Lock

- Purpose:
  - map manuscript v7.6 behavior against production v7.5 code and existing alpha sandbox.
- Concrete outputs:
  - documented gap audit,
  - locked architecture decision.
- Files touched:
  - planning docs only.
- Risks:
  - underestimating legacy NPH remnants,
  - overpromising production integration.
- Criteria to proceed:
  - the gap and the staged architecture are explicit enough to implement.

### Phase 1: Standalone Alpha Sandbox

- Purpose:
  - establish the sandbox as the canonical v7.6 alpha surface.
- Concrete outputs:
  - reproducible entrypoint,
  - explicit config,
  - README,
  - synthetic recovery path,
  - baseline real-data run path.
- Files touched:
  - `test/alpha/code/estimate_alpha.py`
  - `test/alpha/README.md`
  - `test/alpha/Makefile`
  - `test/alpha/params_alpha.yaml`
- Risks:
  - drift from manuscript naming,
  - hidden dependencies on production shortcuts.
- Criteria to proceed:
  - sandbox reproduces synthetic recovery and all expected core artifacts.

### Phase 2: Cohort-Week Analysis Table And Estimators

- Purpose:
  - formalize excess hazard, theta propagation, `F_d(t; alpha)`, and both estimators.
- Concrete outputs:
  - cohort-week table,
  - objective curves,
  - best-estimate summaries,
  - explicit anchor/excess/theta-scale dimensions.
- Files touched:
  - mostly `test/alpha/code/estimate_alpha.py`,
  - optional helper module if justified.
- Risks:
  - hidden assumptions about excess-hazard definition, anchor choice, or theta propagation.
- Criteria to proceed:
  - both estimators run over the same table,
  - outputs are inspectable,
  - excess-hazard semantics are explicit in code and outputs.

### Phase 3: Diagnostics And Robustness Gates

- Purpose:
  - decide whether alpha is interpretable.
- Concrete outputs:
  - pooled estimate,
  - age-band restricted estimates,
  - early-wave vs late-wave estimates,
  - pairwise objective curve,
  - collapse objective curve,
  - objective-curvature metric,
  - leave-one-out CSV,
  - bootstrap CSV,
  - theta-scale sensitivity,
  - anchor sensitivity,
  - negative-excess handling sensitivity.
- Files touched:
  - `test/alpha/code/estimate_alpha.py`
  - `test/alpha/README.md`
- Risks:
  - treating agreement as sufficient,
  - visually eyeballing identifiability without numeric criteria.
- Criteria to proceed:
  - diagnostics are reproducible and operationalized numerically,
  - failure is surfaced as `not identified`.

### Phase 4: Downstream Calibration Checks

- Purpose:
  - show that fitted alpha improves coherence rather than only objective fit.
- Concrete outputs:
  - downstream coherence artifacts for manuscript-facing interpretation.
- Files touched:
  - sandbox outputs and figure-generation code only.
- Risks:
  - mistaking objective optimization for substantive improvement.
- Criteria to proceed:
  - improvement is visible in coherence diagnostics and consistent with `§3.4`.

### Phase 5: Production Integration Decision Gate

- Purpose:
  - decide whether alpha stays sandbox/reporting-only or can enter production.
- Concrete outputs:
  - explicit go/no-go decision.
- Files touched:
  - planning docs and README only, unless the gate passes.
- Risks:
  - integrating because the machinery exists rather than because the diagnostics justify it.
- Criteria to proceed:
  - all production-gate conditions below are met.

### Phase 6: Production KCOR.py Wiring, If Justified

- Purpose:
  - optionally wire calibrated alpha into hazard preprocessing as an optional module.
- Concrete outputs:
  - config-controlled optional path,
  - preserved default behavior when NPH is omitted,
  - optional workbook/reporting plumbing as needed.
- Files touched:
  - `code/KCOR.py`
  - optional shared helper module
  - `data/Czech/Czech.yaml`
- Risks:
  - contaminating the core path,
  - silently changing defaults,
  - overclaiming production readiness.
- Criteria to proceed:
  - production path remains optional, assumption-bearing, and fully diagnosable.

## Required Diagnostics And Decision Gates

Before alpha is treated as interpretable, require all of the following:

- pooled estimate,
- age-band restricted estimates,
- early-wave vs late-wave segmented estimates,
- pairwise objective curve,
- collapse objective curve,
- objective-curvature metric at the optimum,
- leave-one-cohort-out influence check,
- bootstrap uncertainty or comparable resampling check,
- theta-propagation-scale sensitivity,
- anchor/reference-hazard sensitivity,
- negative-excess handling sensitivity,
- downstream coherence check showing that fitted alpha improves coherence rather than merely optimizing the objective.

Agreement between pairwise and collapse estimators is necessary but not sufficient.

## Objective Curvature Requirement

- Define an operational curvature metric at the optimum, such as a second-derivative approximation or local slope-change measure on the fixed alpha grid.
- Alpha is considered identifiable only if:
  - curvature exceeds a predefined threshold,
  - the optimum is interior and not grid-pinned.
- A visually non-flat curve without passing this metric is insufficient.

## Boundary Behavior Rule

If the optimum occurs at `alpha = 1.00` or `alpha = 1.30`, the result must be treated as boundary-seeking unless independent evidence shows the search grid is too narrow and a justified wider rerun is performed.

## When Alpha Is Allowed Into Production KCOR.py

Only allow `alpha` into `code/KCOR.py` if all of the following hold:

- pairwise and collapse estimates are in the same neighborhood,
- both objectives are non-flat by the explicit curvature metric and not merely boundary-seeking,
- bootstrap uncertainty is not absurdly wide,
- the estimate is not driven by one tiny cohort subset or one influential omission,
- pooled results are reasonably stable to anchor, excess-handling, and theta-scale choices,
- downstream KCOR behavior improves in a way consistent with the manuscript’s diagnostic framing.

If this bar is not met, keep alpha as a sandbox/reporting analysis only.

## Output And Reporting Plan

After eventual execution, the repo should contain at least:

- cohort inclusion / cohort diagnostics table,
- cohort-week analysis table,
- objective-curve CSVs,
- best-estimate summary CSV,
- bootstrap CSV,
- leave-one-out CSV,
- run-level reproducibility artifact containing configuration, cohort selection, alpha grid, and seeds,
- figure-ready outputs for manuscript `§3.4`,
- synthetic recovery outputs,
- human-readable sandbox README with assumptions and run instructions,
- any decision summary documenting whether alpha cleared the production gate.

## Non-Goals

- do not rewrite core theta estimation in this pass,
- do not change the core KCOR estimand,
- do not overclaim alpha as a biological constant,
- do not silently replace the core pipeline with NPH assumptions,
- do not collapse sandbox experimentation and production integration into one step,
- do not reuse synthetic alpha as an input to real-data inference.

## Recommended Next Action

Immediately after approving this plan:

1. treat this file as the durable execution-facing plan,
2. execute Phase 0 by auditing:
   - `code/KCOR.py`
   - `documentation/preprint/paper.md`
   - `data/Czech/Czech.yaml`
   - `test/alpha/`
3. only then begin implementation in the sandbox, not production.

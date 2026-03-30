---
name: kcor-v76-build
overview: Create a staged build plan to bring the repository from the current `code/KCOR.py` v7.5 state to a manuscript-aligned v7.6 optional NPH exponent implementation, keeping alpha sandbox-first until explicit integration gates are met.
todos:
  - id: audit-v75-gap
    content: Audit `code/KCOR.py`, manuscript v7.6 NPH sections, Czech YAML, and alpha sandbox to document the concrete gap between current production code and the manuscript spec.
    status: completed
  - id: lock-sandbox-architecture
    content: Lock the staged sandbox-first architecture and identify which alpha components remain sandbox-only, which may become shared helpers, and what must stay out of the KCOR core.
    status: completed
  - id: spec-v76-behavior
    content: Define the target v7.6 alpha behavior in plain English, including cohort-week construction, excess hazard, theta propagation, pairwise/collapse estimators, pooled/segmented modes, and required diagnostics.
    status: completed
  - id: map-file-changes
    content: Produce the file-level change map for sandbox, optional shared helpers, Czech YAML additions, and later production touchpoints.
    status: completed
  - id: phase-build-work
    content: Break execution into phases 0 through 6 with outputs, touched files, risks, and exit criteria, embedding the synthetic → stress → real staged discipline.
    status: completed
  - id: define-gates-and-artifacts
    content: Specify the alpha identifiability gate, the production-integration bar, the final artifact set, and explicit non-goals.
    status: completed
isProject: false
---

# KCOR v7.6 NPH Build Plan

## Goal

Produce an engineer-ready staged build plan for the v7.6 optional NPH exponent upgrade so that, once the plan is executed, the repository has:

- a validated standalone `alpha` estimation workflow,
- the required diagnostics and figure-ready outputs for manuscript `§3.4`,
- a clear decision gate for whether `alpha` ever enters production `KCOR.py`, and
- no accidental blending of the KCOR core with assumption-bearing NPH logic.

## Current-State Audit

- Production is still explicitly v7.5 in [code/KCOR.py](code/KCOR.py):

```79:79:code/KCOR.py
VERSION = "v7.5"                # KCOR version number
```

- The production integration surface is concentrated in [code/KCOR.py](code/KCOR.py) around `build_kcor_rows()` and `process_workbook()`, which means any later NPH wiring should be narrow and optional rather than a core rewrite.
- The manuscript is already written to the v7.6 conceptual target in [documentation/preprint/paper.md](documentation/preprint/paper.md), so the paper should be treated as the behavior/naming spec.
- The alpha work already exists as a standalone sandbox in [test/alpha/README.md](test/alpha/README.md):

```1:20:test/alpha/README.md
# Alpha Estimation Sandbox
...
- estimate `alpha` with two estimators:
  - pairwise log-ratio fit
  - common-wave collapse fit
...
- validate recovery against synthetic data with known `alpha_true`
```

- Existing Czech configuration in [data/Czech/Czech.yaml](data/Czech/Czech.yaml) already supplies quiet-window and theta-fit structure that a future v7.6 layer can reuse without redefining the KCOR core:

```23:39:data/Czech/Czech.yaml
quietWindow:
  startDate: '2023-01'
  endDate: '2023-52'
...
theta_estimation_windows:
    - ['2023-19', '2023-40', '2023 post COVID']
...
time_varying_theta:
  gompertz_gamma: 0.085
```

## Scope Definition

- **Core KCOR remains unchanged**
  - cohort construction, `theta0` fitting, delta-iteration, inversion, workbook plumbing, and the non-causal estimand stay intact.
- **Optional v7.6 NPH estimator/calibration layer**
  - build and validate `alpha` estimation from cohort-week structure, excess hazard, pairwise/collapse estimators, and diagnostics.
- **Optional production use of calibrated alpha**
  - only after explicit gates are met; never folded silently into the universal KCOR path.
- **Alpha remains a diagnostic layer on top of KCOR geometry**
  - `alpha` is not part of the universal KCOR estimand or core depletion-normalization machinery; it is an optional, assumption-bearing layer used to summarize frailty-dependent excess-hazard structure when diagnostics support identification.

## Architecture Decision

Recommend a staged version of **Option A first, Option B only later if justified**.

1. **Stage sandbox-first**

- Keep `alpha` estimation in [test/alpha/](test/alpha/) while the estimator, stress behavior, and real-data identifiability gates are being validated.
- Reuse production helpers where safe, but keep the full `alpha` workflow outside `code/KCOR.py` until decision gates are passed.

1. **Promote shared math only if duplication becomes real**

- If both sandbox and production need the same `F_d(t; alpha)` math, excess-hazard transforms, or objective helpers, extract a small helper module rather than pushing the whole estimator into production.

1. **Production integration is conditional, not assumed**

- `alpha` should enter [code/KCOR.py](code/KCOR.py) only if the pooled estimate is diagnostically interpretable, reasonably stable, and improves downstream coherence in the manuscript’s diagnostic sense.

1. **Never mix these roles**

- Synthetic `alpha` is for estimator validation only.
- Real-data `alpha` is inferred independently from data.
- The core KCOR path must remain runnable without the NPH module.
- `alpha` must never become a hidden prerequisite for the KCOR core path.

## Explicit v7.6 Target Behavior

The final build should make the following behavior concrete and testable:

- Build an NPH cohort-week table from fixed cohorts over prespecified wave periods, using quiet-window-derived cohort frailty state as input.
- Define excess hazard explicitly and consistently. Default working definition:
  - `h_excess,d(t) = h_d(t) - h_ref(t)`
  - Alternative formulations such as ratio or log-ratio may be evaluated as labeled sensitivity analyses, but must not be mixed silently across estimators or runs.
- Surface the anchor/reference hazard choice as a sensitivity dimension rather than hiding it in code.
- Propagate the chosen theta quantity through the wave-period calculation explicitly and consistently.
- Compute `F_d(t; alpha) = E[z^alpha | t]` under the gamma-frailty working model in a reusable, documented way.
- Implement two estimators:
  - pairwise cross-cohort log-ratio estimator,
  - common-wave collapse estimator.
- Support pooled estimation and segmented estimation (e.g. early-wave vs late-wave) as distinct outputs.
- Treat flat objectives, estimator disagreement, bootstrap instability, boundary-seeking estimates, and downstream incoherence as weak-identification signals rather than valid `alpha` results.

## Staged Path

The execution order should follow the guidance’s clean staged logic and be embedded into the phases below:

- **Stage 1: Synthetic**
  - prove the estimator recovers known `alpha_true` when truth is known.
- **Stage 2: Stress**
  - show failure behavior under heteroskedasticity, weak signal, or similar depletion geometry.
- **Stage 3: Real data**
  - infer `alpha` from observed cross-cohort structure only; never carry synthetic `alpha` into real-data claims.

## Alpha Interpretation Contract

- `alpha` is not a causal or biological parameter.
- `alpha` is the exponent that best fits cross-cohort excess-hazard structure under the working model.
- `alpha` is only interpretable when identifiability diagnostics pass.
- If diagnostics fail, `alpha` must be reported as `not identified` rather than estimated.

## No Silent Fallback Rule

- If any required diagnostic fails, including flat objective behavior, estimator disagreement, instability, or boundary-seeking optima, the pipeline must **not** silently emit a numeric `alpha`.
- The run must explicitly flag `alpha` as `not identified` in outputs, summaries, and any downstream reporting artifact.

## Default No-Go Outcome

If the production gate is not met, the repository should still ship a complete v7.6 sandbox/reporting workflow, with `alpha` remaining outside production `KCOR.py`.

## Alpha Grid Specification

- Default alpha grid: `1.00` to `1.30`.
- Default step size: fixed, preferably `0.005` unless a different fixed step is justified and documented.
- The grid must be identical across pairwise estimation, collapse estimation, pooled/segmented runs, and all diagnostic plots for a given run.

## Reproducibility Requirement

Every alpha run must emit a machine-readable run artifact recording at minimum:

- configuration used, including anchor choice, excess mode, theta propagation scale, and segmentation mode,
- cohort selection / inclusion,
- alpha grid,
- random seed or resampling seed whenever stochastic components are used.

## File-Level Change Map

- **Modify existing**
  - [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py)
    - primary sandbox driver for cohort-week table construction, estimators, diagnostics, and paper `§3.4` outputs.
  - [test/alpha/README.md](test/alpha/README.md)
    - document assumptions, run steps, staged interpretation, and decision gates.
  - [test/alpha/Makefile](test/alpha/Makefile)
    - keep sandbox execution simple and reproducible.
  - [data/Czech/Czech.yaml](data/Czech/Czech.yaml)
    - only add optional NPH config knobs if they are necessary and do not contaminate the KCOR core defaults.
  - [code/KCOR.py](code/KCOR.py)
    - production touchpoint only in the final gated phase.
- **Create new if needed**
  - `test/alpha/code/<shared_helper>.py`
    - only if sandbox/production truly need common `alpha` math.
  - [documentation/plans/kcor_v76_nph_build_plan.md](documentation/plans/kcor_v76_nph_build_plan.md)
    - durable repository-facing build plan artifact to hand back for execution.
- **Leave untouched for now**
  - core theta0 estimator design outside minimal helper reuse,
  - KCOR estimand definition and non-NPH production pipeline behavior,
  - unrelated production reporting paths.

## Phased Execution Plan

### Phase 0: Current-State Audit And Design Lock

- Purpose
  - map manuscript v7.6 behavior against production v7.5 code and existing alpha sandbox.
- Concrete outputs
  - documented gap audit,
  - locked architecture decision: sandbox-first, gated production later.
- Files touched
  - planning docs only; likely [documentation/plans/kcor_v76_nph_build_plan.md](documentation/plans/kcor_v76_nph_build_plan.md).
- Risks
  - underestimating existing legacy NPH pieces in [code/KCOR.py](code/KCOR.py), or overpromising production integration too early.
- Exit criteria
  - file map, phases, and decision gates are explicit enough for implementation.

### Phase 1: Standalone Alpha Sandbox

- Purpose
  - ensure the sandbox is the canonical place for v7.6 `alpha` estimation and synthetic validation.
- Concrete outputs
  - reproducible sandbox entrypoint,
  - README describing assumptions, inputs, and outputs,
  - synthetic-recovery and baseline real-data run path.
- Files touched
  - [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py), [test/alpha/README.md](test/alpha/README.md), [test/alpha/Makefile](test/alpha/Makefile), [test/alpha/params_alpha.yaml](test/alpha/params_alpha.yaml).
- Risks
  - sandbox logic drifting from manuscript naming or silently reusing production shortcuts.
- Criteria to proceed
  - sandbox builds cohort-week data, runs both estimators, and reproduces synthetic recovery artifacts.

### Phase 2: Cohort-Week Analysis Table And Estimators

- Purpose
  - formalize the v7.6 working behavior: excess hazard, theta propagation, `F_d(t; alpha)`, pairwise and collapse estimators.
- Concrete outputs
  - cohort-week analysis table,
  - objective-curve CSVs,
  - best-estimate summary CSVs,
  - clear anchor/excess/theta-scale dimensions.
- Files touched
  - primarily [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py); optional helper module if justified.
- Risks
  - hidden assumptions about anchor hazard, excess handling, or theta propagation.
- Criteria to proceed
  - both estimators run over the same cohort-week table and their outputs are inspectable, not just final point estimates,
  - the excess-hazard definition is explicit in code and outputs rather than implicit in implementation details.

### Phase 3: Diagnostics And Robustness Gates

- Purpose
  - define when `alpha` is interpretable and when it should be rejected.
- Concrete outputs
  - pooled estimate,
  - age-band restricted estimates,
  - early-wave vs late-wave estimates,
  - pairwise and collapse objective curves,
  - objective-curvature metric at the optimum,
  - leave-one-cohort-out CSV,
  - bootstrap CSV,
  - theta-scale sensitivity,
  - anchor sensitivity,
  - negative-excess handling sensitivity.
- Files touched
  - [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py), [test/alpha/README.md](test/alpha/README.md).
- Risks
  - treating estimator agreement as sufficient, or reporting unstable pooled estimates as if identified.
- Criteria to proceed
  - diagnostics are explicit, reproducible, and surfaced as gates rather than afterthoughts,
  - identifiability is operationalized numerically rather than judged only by eye.

### Phase 4: Downstream Calibration Checks

- Purpose
  - confirm that fitted `alpha` improves coherence in the manuscript’s diagnostic sense rather than merely optimizing the alpha objective mechanically.
- Concrete outputs
  - downstream comparison artifacts showing whether calibrated `alpha` improves cross-cohort coherence or interpretability.
- Files touched
  - sandbox outputs and figure-generation code only.
- Risks
  - mistaking objective optimization for substantive improvement.
- Criteria to proceed
  - improvement is visible in coherence diagnostics and consistent with `§3.4` framing.

### Phase 5: Production Integration Decision Gate

- Purpose
  - decide whether `alpha` should remain sandbox/reporting-only or be allowed into production.
- Concrete outputs
  - explicit go/no-go decision recorded in the build plan and README.
- Files touched
  - planning docs and README; no production wiring unless gate passes.
- Risks
  - integrating because the code exists, not because the diagnostics justify it.
- Criteria to proceed
  - all conditions in the production gate below are met.

### Phase 6: Production KCOR.py Wiring, If Justified

- Purpose
  - optionally wire calibrated `alpha` into hazard preprocessing as an optional module in [code/KCOR.py](code/KCOR.py).
- Concrete outputs
  - optional config-controlled production path,
  - preserved default behavior when NPH is omitted,
  - workbook/reporting plumbing for NPH-aware outputs if needed.
- Files touched
  - [code/KCOR.py](code/KCOR.py), optional shared helper module, [data/Czech/Czech.yaml](data/Czech/Czech.yaml).
- Risks
  - contaminating the core path, silently changing defaults, or overclaiming production-readiness.
- Criteria to proceed
  - production path remains optional, assumption-bearing, and fully diagnosable.

## Required Diagnostics And Decision Gates

The build must require all of the following before `alpha` is treated as interpretable:

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
- downstream coherence check showing the fitted `alpha` improves interpretation rather than just objective fit.

Agreement between pairwise and collapse estimators is **necessary but not sufficient**.

## Objective Curvature Requirement

- Define an operational curvature metric at the optimum, such as a second-derivative approximation or local slope-change measure computed on the fixed alpha grid.
- `alpha` is considered identifiable only if:
  - the curvature metric exceeds a predefined threshold, and
  - the optimum is interior rather than pinned to the grid boundary.
- A visually non-flat curve without passing this metric is insufficient for claiming identification.

## Boundary Behavior Rule

If the optimum occurs at `alpha = 1.00` or `alpha = 1.30`, the result must be treated as boundary-seeking unless independent evidence shows the search grid is too narrow and a justified wider rerun is performed.

## When Alpha Is Allowed Into Production KCOR.py

Only allow `alpha` into [code/KCOR.py](code/KCOR.py) if all of the following hold:

- pairwise and collapse estimates are in the same neighborhood,
- both objectives are non-flat by the explicit curvature metric and not merely boundary-seeking,
- bootstrap uncertainty is not absurdly wide,
- the estimate is not driven by one tiny cohort subset or one influential omission,
- pooled results are reasonably stable to anchor, excess-handling, and theta-scale choices,
- downstream KCOR behavior improves in a way consistent with the manuscript’s diagnostic framing.

If this bar is not met, keep `alpha` as a sandbox/reporting analysis only.

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
- any decision summary documenting whether `alpha` cleared the production gate.

## Non-Goals

- do not rewrite core theta estimation in this pass,
- do not change the core KCOR estimand,
- do not overclaim `alpha` as a biological constant,
- do not silently replace the core pipeline with NPH assumptions,
- do not collapse sandbox experimentation and production integration into one step,
- do not reuse synthetic `alpha` as an input to real-data inference.

## Recommended Next Action

After this plan is approved, create the durable repo plan artifact at [documentation/plans/kcor_v76_nph_build_plan.md](documentation/plans/kcor_v76_nph_build_plan.md), then execute Phase 0 by auditing [code/KCOR.py](code/KCOR.py), [documentation/preprint/paper.md](documentation/preprint/paper.md), [data/Czech/Czech.yaml](data/Czech/Czech.yaml), and [test/alpha/](test/alpha/) against the v7.6 manuscript behavior before changing any production code.

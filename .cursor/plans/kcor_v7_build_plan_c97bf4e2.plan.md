---
name: KCOR V7 Build Plan
overview: Implement KCOR V7 by starting from the frozen V6 pipeline and making a surgical theta-estimation upgrade driven by multi-window quiet periods in dataset YAML, while preserving V6 correction mechanics and outputs.
todos:
  - id: rebase-kcor-py
    content: Rebase code/KCOR.py to a clean KCORv6-equivalent baseline while preserving current CLI/runtime entry behavior.
    status: completed
  - id: add-fit-theta0-global
    content: Implement global quiet-window theta0 regression function with bounded least-squares and diagnostics.
    status: completed
  - id: parse-theta-estimation-windows
    content: Add YAML parsing and mask-construction logic for theta_estimation_windows with quietWindow fallback.
    status: completed
  - id: wire-v7-theta-in-process
    content: Replace V6 theta estimation callsite in process_workbook to use theta0_global per cohort and keep invert_gamma_frailty unchanged.
    status: completed
  - id: extend-logging-summary
    content: Add V7 theta-source/diagnostic fields to logs and summary outputs without changing workbook contracts.
    status: completed
  - id: doc-v7-update
    content: Update README V7 narrative to match spec (unchanged inversion machinery, upgraded theta input).
    status: completed
isProject: false
---

# KCOR V7 Build Plan

## Goal

Upgrade runtime `KCOR.py` to V7 by re-basing on V6 logic and changing only how cohort theta is estimated: use global quiet-window regression for `theta0` (post-enrollment anchor), then keep the existing gamma-frailty inversion path unchanged.

## Key Inputs

- Spec: [documentation/specs/KCORv7/KCORv7_spec.md](documentation/specs/KCORv7/KCORv7_spec.md)
- Baseline implementation: [code/KCORv6.py](code/KCORv6.py)
- Dataset config with new windows: [data/Czech/Czech.yaml](data/Czech/Czech.yaml)

## Build Scope

- Runtime code target: [code/KCOR.py](code/KCOR.py)
- Keep output sheet contracts and downstream Makefile/test entrypoints unchanged.
- Use `theta_estimation_windows` where present; preserve `quietWindow` as fallback for backward compatibility.

## Implementation Steps

1. **Re-base runtime on V6 core**
  - Replace current `KCOR.py` body with a clean V6-equivalent baseline from [code/KCORv6.py](code/KCORv6.py).
  - Preserve only required environment plumbing/CLI compatibility already expected by current Make targets.
2. **Add V7 theta estimator (new function)**
  - Introduce `fit_theta0_global(h_arr, H_cum, quiet_mask, theta0_init=1.0)` in `KCOR.py` near existing frailty fitting helpers.
  - Implement model: `h_q ~ k / (1 + theta0 * H_q)` using bounded nonlinear least squares (`k>=1e-12`, `theta0>=0`).
  - Treat the boundary solution `theta0=0` as valid and expected when data show no curvature (degenerate case `h_q = k`, i.e., no detectable frailty signal and no correction). Do not special-case this path.
  - Return `theta0_hat` plus diagnostics (success, n_obs, rmse, status/message) to match V6-style reporting robustness.
3. **Add quiet-window parsing for V7 config**
  - Extend YAML load path in `KCOR.py` to parse `theta_estimation_windows` from dataset YAML (same source currently used for `dosePairs`, `covidCorrection`, `quietWindow`).
  - Build reusable mask helper from ISO windows (`YYYY-WW`) against per-cohort timeline indices.
  - Fallback logic:
    - If `theta_estimation_windows` exists and yields enough points, use it.
    - Else fall back to legacy single `quietWindow` behavior.
4. **Swap theta estimation callsite in workbook processing**
  - In `process_workbook(...)`, where V6 currently derives `(k_hat, theta_hat)` from quiet-window cumulative-hazard fitting, replace theta source with `theta0_hat` from global quiet-window fit.
  - Apply the same estimator independently to each cohort key `(EnrollmentDate, YoB, Dose)` and reuse existing `invert_gamma_frailty(H_obs, theta)` pipeline unchanged.
  - Keep existing skip-week/cumulative hazard conventions intact to avoid behavioral drift.
5. **Diagnostics and logging updates**
  - Extend parameter maps and summary output to distinguish `theta_source` (v7-global vs fallback-v6) and include `theta0_hat` fit diagnostics.
  - Keep log formatting stable, only append V7-specific lines to aid validation.
6. **Documentation alignment**
  - Update V7 description in repo docs (at minimum in [README.md](README.md), if present) to reflect: “V6 machinery unchanged, theta input upgraded via multi-window global fit.”

## Execution/Data Flow (Target)

```mermaid
flowchart TD
  yamlConfig[DatasetYamlConfig] --> windowBuilder[BuildQuietWindowMask]
  cohortSeries[CohortSeries h(t), H(t)] --> thetaFit[fitTheta0Global]
  windowBuilder --> thetaFit
  thetaFit --> theta0[theta0Hat]
  theta0 --> frailtyInvert[invertGammaFrailty]
  cohortSeries --> frailtyInvert
  frailtyInvert --> outputs[KCOROutputsAndSummary]
```



## Validation Plan

- **Config validation**: Confirm [data/Czech/Czech.yaml](data/Czech/Czech.yaml) `theta_estimation_windows` is consumed and logged.
- **Numerical sanity**:
  - `theta0_hat >= 0` always.
  - If optimizer lands at `theta0_hat == 0`, accept it as the correct degenerate solution for flat-hazard cohorts.
  - Near-zero frailty cohorts produce near-zero correction naturally (no hardcoded cohort exclusion).
- **Regression checks**:
  - Output workbook schema unchanged (sheet names/columns expected by downstream tooling).
  - Existing Make/test paths still run against `KCOR.py` without interface break.
- **Behavioral checks**:
  - Compare V6 vs V7 on one cohort: V7 differs primarily through theta input; inversion formula path remains identical.

## Risks & Mitigations

- **Risk**: Sparse points in specified windows.
  - **Mitigation**: enforce minimum-point threshold and automatic fallback to legacy `quietWindow` fit.
- **Risk**: ISO-week window parsing mismatch.
  - **Mitigation**: reuse existing ISO conversion helpers and centralize mask construction.
- **Risk**: unintended drift from V6.
  - **Mitigation**: strict “V6-first rebase,” then isolated theta-estimation diff only.


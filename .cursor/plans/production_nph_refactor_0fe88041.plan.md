---
name: production nph refactor
overview: Refactor the production NPH correction path in `code/KCOR.py` to use an optional symmetric all-cohort neutralization design with `identified_only` default behavior, `forced_alpha` sensitivity support, and no legacy `factor` branch, while keeping behavior unchanged when NPH is disabled.
todos:
  - id: normalize-production-config
    content: Redesign `NPH_correction` parsing in `code/KCOR.py` around `mode`, `forced_alpha`, and `neutralization`, and update dataset YAML accordingly.
    status: completed
  - id: remove-legacy-factor
    content: Remove the legacy `factor`-based NPH branch from the production correction path.
    status: completed
  - id: implement-symmetric-production-path
    content: Replace the asymmetric dose-0-anchored production adjustment with a symmetric all-cohort neutralization path using the existing theta/frailty machinery.
    status: completed
  - id: add-mode-gating
    content: Ensure `off`, `identified_only`, and `forced_alpha` behave distinctly and never silently promote a non-identified alpha into primary use.
    status: completed
  - id: audit-and-verify
    content: "Update production logging and define verification checks proving disabled mode is unchanged, `identified_only` skips cleanly without an identified alpha, `forced_alpha: 1.0` is identity, and symmetric reconstruction does not reintroduce asymmetry."
    status: completed
isProject: false
---

# Production Symmetric NPH Plan

## Goal

- Update the production NPH correction architecture in [C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py](C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py) so the correction form is symmetric across cohorts, remains optional, defaults to `identified_only`, and supports `forced_alpha` sensitivity runs.
- Do not execute this plan until explicitly requested.

## Current Production Seams

- [C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py](C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py) loads `NPH_correction` from dataset YAML in `_load_dataset_dose_pairs_config()` and exposes it through `get_covid_correction_config()`.
- The current production correction is applied in `apply_covid_correction_in_place()`, where `external_default_alpha` adjusts only `Dose != 0` against a dose-0 reference hazard and `legacy_factor` is a separate asymmetric branch.
- [C:/Users/stk/Documents/GitHub/KCOR/data/Czech/Czech.yaml](C:/Users/stk/Documents/GitHub/KCOR/data/Czech/Czech.yaml) currently uses `NPH_correction.startDate`, `endDate`, and `default_alpha`, so the production-facing config surface will need to change.

## Planned Changes

- Replace the current production config semantics under [C:/Users/stk/Documents/GitHub/KCOR/data/Czech/Czech.yaml](C:/Users/stk/Documents/GitHub/KCOR/data/Czech/Czech.yaml) with a mode-driven structure such as:
  - `mode: identified_only`
  - `forced_alpha: null`
  - `neutralization: symmetric_all_cohorts`
  - keep `startDate` and `endDate`
- Refactor [C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py](C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py) config parsing so `get_covid_correction_config()` returns a normalized production contract:
  - `mode`: `off | identified_only | forced_alpha`
  - `forced_alpha`
  - `neutralization`
  - parsed wave window
  - any resolved alpha source metadata needed for logging and auditability
- Remove the legacy `factor` branch from the production implementation rather than carrying it forward as a compatibility mode.
- Introduce a single symmetric NPH hazard-adjustment path in `apply_covid_correction_in_place()` that:
  - computes the same frailty-moment factor `F_d(t; alpha)` from the same theta-based depletion model already used by the KCOR inversion machinery
  - applies neutralization cohort-by-cohort to all cohorts, including dose 0
  - avoids reintroducing asymmetry during hazard reconstruction
- Keep NPH optional:
  - `off`: no production hazard changes
  - `identified_only`: apply only when a production-readable identified alpha is available
  - `forced_alpha`: apply the symmetric correction with an explicitly supplied numeric alpha for sensitivity runs
- Add explicit gating so production behavior is unchanged when NPH is disabled, and so `identified_only` does not silently fall back to an external default alpha and pretend the result is identified.
- Add an explicit no-alpha-source rule for `identified_only`:
  - if no identified production alpha source is available, no NPH correction is applied
  - `hazard_raw` remains unchanged
  - logs must state that correction was skipped because no identified alpha was available
- Update startup logging in [C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py](C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py) to print the new NPH mode, resolved alpha source, and whether the symmetric path is active.
- Add explicit sensitivity-only logging for `forced_alpha` runs so supplied alpha values, including boundary or extreme values, are never labeled as identified.
- Align the production plan with the sandbox conclusion:
  - treat symmetry as an architectural cleanup and sensitivity-analysis capability
  - do not treat it as evidence that Czech primary analysis now supports NPH

## Guardrails

### Alpha = 1 Identity Requirement

- When `mode: forced_alpha` and `forced_alpha: 1.0`, the symmetric NPH correction must reduce to the identity transformation on `hazard_raw` within a defined numerical tolerance.
- Verification must compare pre-correction and post-correction `hazard_raw` values and fail if any difference exceeds tolerance.

### Numerical Tolerance

- All identity and invariance checks must use an explicit numerical tolerance to avoid false failures from floating-point noise.
- Use a defined absolute or relative tolerance such as `1e-12`, and record that tolerance in the implementation or audit output used for verification.

### No Alpha Source Rule

- When `mode: identified_only` and no identified production alpha source is available, no NPH correction is applied.
- In that case `hazard_raw` must remain unchanged and logs must explicitly report that correction was skipped due to the missing identified alpha source.

### Frailty Moment Consistency

- The production symmetric path must compute `F_d(t; alpha)` using the same theta-based depletion model already used by the KCOR inversion machinery.
- Do not introduce a second or approximate frailty-moment implementation in production.

### Reconstruction Consistency

- After symmetric neutralization and hazard reconstruction, no cohort may be implicitly protected from correction.
- Dose 0 must be adjusted under the same rule as other cohorts.
- The reconstruction step must not reintroduce asymmetry after the neutralization step.

## Verification

- Verify that `mode: off` leaves `hazard_raw` and downstream KCOR behavior unchanged.
- Verify that `forced_alpha` applies the symmetric all-cohort path, is labeled as sensitivity-only in logs, and never labels a supplied alpha as identified.
- Verify that `mode: forced_alpha` with `forced_alpha: 1.0` leaves `hazard_raw` unchanged within the defined numerical tolerance.
- Verify that `identified_only` does nothing unless an identified production alpha source is explicitly available.
- Verify that when `identified_only` has no alpha source, correction is skipped, `hazard_raw` is unchanged, and logs say why.
- Verify that no code path silently uses the old `factor` semantics or relabels a forced/supplied alpha as identified.
- Verify that the symmetric production path uses the existing theta/frailty machinery rather than a second approximation for `F_d(t; alpha)`.
- Verify that hazard reconstruction treats dose 0 and nonzero-dose cohorts under the same rule and does not reintroduce asymmetry.
- Confirm that the Czech dataset config loads under the new schema and that the production NPH summary line reflects the new mode semantics.


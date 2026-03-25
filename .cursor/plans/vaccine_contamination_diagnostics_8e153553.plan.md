---
name: Vaccine contamination diagnostics
overview: Add read-only vaccine-contamination diagnostics to `fit_theta0_gompertz`, propagate them into `theta0_diagnostics`, and add run-log/validation checks without changing estimation behavior.
todos:
  - id: add-fitter-contamination-diags
    content: Compute and return seven pre-wave contamination diagnostics in fit_theta0_gompertz before delta branching.
    status: completed
  - id: propagate-diags-to-params
    content: Copy new diagnostics into params_dict for per-age and all-ages paths.
    status: completed
  - id: update-theta0-diagnostics-builder
    content: Add new columns and compute k_hat_vs_dose0_ratio with dose-0 join (mc-aware).
    status: completed
  - id: add-log-and-validations
    content: Add dual_print summary and validation checks for ratio/richness conditions.
    status: completed
  - id: verify-lints-and-smoke
    content: Run py_compile/lints and a targeted smoke validation for new diagnostics.
    status: completed
isProject: false
---

# Add Vaccine-Contamination Diagnostics

## Scope

- Implement only diagnostics in `[code/KCOR.py](code/KCOR.py)`; do not change k/theta fitting, delta iteration, fallback branching behavior, or KCOR computation.
- Add new per-cohort scalar fields to `fit_theta0_gompertz` return payload and expose them in `theta0_diagnostics`.
- Add one cross-cohort field (`k_hat_vs_dose0_ratio`) in diagnostics builder and run-log summary.

## 1) Extend `fit_theta0_gompertz` payload (pre-branch, read-only)

- In `[fit_theta0_gompertz()](code/KCOR.py)`, compute from the pre-wave quiet segment (`first_idx`) immediately before the delta-iteration loop:
  - `h_obs_quiet_slope` via `np.polyfit(t_prewave, h_obs_prewave, 1)[0]`
  - `h_obs_first4_mean`, `h_obs_last4_mean`
  - `h_obs_first4_vs_last4_ratio` (NaN if either side has <4 bins)
  - `quiet_window_duration_weeks = t_prewave[-1] - t_prewave[0] + 1`
  - `k_hat_first4`, `k_hat_last4` using same `h/exp(gamma*t)` transform as `k_hat`
- Persist these fields into **all** return dictionaries:
  - normal converged return,
  - `DELTA_INAPPLICABLE` early fallback return,
  - iteration-exception return,
  - `_fail` returns as NaN placeholders where pre-wave bins are unavailable.
- Keep these fields computed/populated before any delta-sign branch so they always exist for fallback cohorts.

## 2) Propagate fields into cohort params used by sheet builders

- In both params construction paths in `[process_workbook()](code/KCOR.py)` (per-age and all-ages), copy the new diagnostics from `diag` into `params_dict`.
- Preserve current status semantics already implemented (`NOT_IDENTIFIED`, `INSUFFICIENT_DEATHS`, `DELTA_INAPPLICABLE`, `OK`).

## 3) Extend `theta0_diagnostics` builder

- In `[build_theta0_diagnostics_rows()](code/KCOR.py)`:
  - Add the 7 new scalar columns into row construction and output ordering.
  - Compute `k_hat_vs_dose0_ratio` after DataFrame creation by joining dose-0 reference `k_hat` on `(EnrollmentDate, YearOfBirth)` (and `mc_id` when present).
  - Emit NaN when no dose-0 reference exists; ensure ratio is NaN for dose-0 rows.
  - Place new columns after residual columns in output order per spec.

## 4) Add run-log contamination summary + validation checks

- In `[create_summary_file()](code/KCOR.py)`, add a compact log summary (`dual_print`) for mean `k_hat_vs_dose0_ratio` by dose.
- Add lightweight validation checks during diagnostics build/logging:
  - dose-0 ratio is NaN,
  - at least some dose>0 rows with `h_obs_first4_vs_last4_ratio > 1` (informational warning if none),
  - all 7 new scalar fields populated for `DELTA_INAPPLICABLE` rows (warn if missing).

## 5) Verification

- Run syntax check and lints for `[code/KCOR.py](code/KCOR.py)`.
- Run a targeted smoke run to confirm:
  - `theta0_diagnostics` has new columns,
  - `k_hat_vs_dose0_ratio` behavior matches rules,
  - run log prints mean ratio by dose.


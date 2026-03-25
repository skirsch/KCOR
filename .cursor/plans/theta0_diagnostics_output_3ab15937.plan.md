---
name: Theta0 diagnostics output
overview: Add a cohort-level `theta0_diagnostics` sheet to the primary KCOR output workbook and propagate theta fit diagnostics/status into both this new sheet and `dose_pairs` (`theta0_status_num/den`).
todos:
  - id: extend-fit-diagnostics
    content: Add missing delta/residual/window diagnostics to fit_theta0_gompertz return payload, including theta0_init and prewave-bin count.
    status: completed
  - id: persist-status-and-fields
    content: Propagate new fields into params_dict and compute canonical theta0_status.
    status: completed
  - id: update-dose-pairs-columns
    content: Add theta0_status_num/theta0_status_den to build_kcor_rows and output column order.
    status: completed
  - id: add-theta0-diagnostics-sheet
    content: Create and write theta0_diagnostics DataFrame in primary workbook writer.
    status: completed
  - id: verify-output-shape
    content: Run targeted validation and lints, including delta_raw population checks for DELTA_INAPPLICABLE cohorts.
    status: completed
isProject: false
---

# Implement theta0 diagnostics output

## Scope and target outputs

- Implement in the primary workbook writer in `[code/KCOR.py](code/KCOR.py)`, not summary-only files.
- Add a new worksheet `theta0_diagnostics` with one row per unique cohort key `(EnrollmentDate, YearOfBirth, Dose)` (plus `mc_id` when present in Monte Carlo mode).
- Extend `dose_pairs` rows with `theta0_status_num` and `theta0_status_den`.

## 1) Extend theta fitter diagnostics payload

- Update `fit_theta0_gompertz` in `[code/KCOR.py](code/KCOR.py)` to return missing fields needed by the spec:
  - `theta0_init` (pre-wave single-window seed theta before iteration),
  - per-wave `delta_raw_*` and `delta_applied_*` (from existing internal delta loop),
  - `delta_negative` boolean (any raw delta < 0),
  - `n_iter` (`n_iterations` already tracked),
  - `n_quiet_bins_prewave` (count in first/pre-wave quiet segment) in addition to total `n_quiet_bins`,
  - `fit_residual_max` (max absolute hazard residual), while preserving current RMS,
  - quiet-window index bounds (`quiet_window_start`, `quiet_window_end`) based on fit mask indices used for theta fitting.
- Enforce fallback-safe sequencing in `fit_theta0_gompertz`:
  - compute per-wave raw deltas first,
  - set `delta_negative = any(delta_raw_i < 0)`,
  - populate diagnostics payload fields (`delta_raw_*`, `delta_applied_*`, `delta_negative`, `theta0_init`) **before** any early-return branch,
  - then branch to `DELTA_INAPPLICABLE` fallback if negative, otherwise continue iteration.
- Ensure `theta0_init` is always persisted for all statuses, including `DELTA_INAPPLICABLE`, so `theta0_init` vs `theta0_hat` remains analyzable.
- Keep existing behavior for `DELTA_INAPPLICABLE` fallback and convergence logic unchanged.

## 2) Persist normalized cohort diagnostics in `kcor6_params_map`

- In both per-age and all-ages fit branches (the two `params_dict` construction blocks in `[code/KCOR.py](code/KCOR.py)`), add the new diagnostics fields from `diag`.
- Add a derived canonical status enum field for output (`theta0_status`) mapped from fit/guard outcomes in priority order:
  - `NOT_IDENTIFIED`
  - `INSUFFICIENT_DEATHS`
  - `DELTA_INAPPLICABLE`
  - else `OK`
- Implement `NOT_IDENTIFIED` from identifiability failure conditions (instead of collapsing into `OK`/`weak_identifiability`) so status semantics match the diagnostic spec.
- Preserve existing fields (`theta_fit_status`, `note`) for backward compatibility.

## 3) Add statuses to `dose_pairs`

- In `build_kcor_rows` (`[code/KCOR.py](code/KCOR.py)`), when reading numerator/denominator params from `kcor6_params_map`, also read `theta0_status`.
- Append columns `theta0_status_num` and `theta0_status_den` to the output, and include them in standard ordering near `theta_num/theta_den`.

## 4) Build and write `theta0_diagnostics` sheet

- Add a helper in `[code/KCOR.py](code/KCOR.py)` to convert `kcor6_params_map` into a cohort-level diagnostics DataFrame with columns:
  - `EnrollmentDate`, `YearOfBirth`, `Dose`, `k_hat`, `theta0_init`, `theta0_hat`, `n_quiet_bins`, `n_quiet_bins_prewave`, `n_iter`,
  - `delta_raw_`*, `delta_applied_*`, `delta_negative`, `status`,
  - `quiet_window_start`, `quiet_window_end`, `fit_residual_max`, `fit_residual_rms`,
  - include `mc_id` in MC mode.
- Ensure deterministic column ordering with dynamic wave columns sorted by wave index (e.g. `delta_raw_1`, `delta_raw_2`), and gracefully handle cohorts with fewer waves by emitting `NaN` for missing wave columns.
- Write this DataFrame into the primary workbook writer block (same block that writes `dose_pairs`, `About`, `by_dose`, `dose_pair_deaths`) as sheet `theta0_diagnostics`.

## 5) Validation

- Run a representative KCOR execution to confirm:
  - workbook contains `theta0_diagnostics`,
  - row uniqueness per `(EnrollmentDate, YearOfBirth, Dose[, mc_id])`,
  - `dose_pairs` contains `theta0_status_num/den`,
  - expected statuses appear (`NOT_IDENTIFIED`, `INSUFFICIENT_DEATHS`, `DELTA_INAPPLICABLE`, `OK`) with non-empty delta fields where applicable,
  - `delta_raw_*` remains populated for `DELTA_INAPPLICABLE` cohorts (even when fallback is used), so negative raw values are visible in `theta0_diagnostics`.
- Run lints on edited file(s) and fix introduced diagnostics issues only.


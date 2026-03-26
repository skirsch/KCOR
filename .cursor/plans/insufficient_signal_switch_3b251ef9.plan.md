---
name: Insufficient signal switch
overview: Replace delta-inapplicable fallback with an insufficient-signal zero-theta rule, propagate the new status/diagnostic fields through outputs, and verify behavior without changing core delta/quiet-window logic.
todos:
  - id: switch-fallback-to-zero-theta
    content: Replace first-pass DELTA_INAPPLICABLE fallback with INSUFFICIENT_SIGNAL zero-theta return and remove single-window refit.
    status: completed
  - id: propagate-new-status-and-flag
    content: Update params/status ladder/log fields and add theta_hit_bound propagation in per-age and all-ages paths.
    status: completed
  - id: update-diagnostics-builder
    content: Add theta_hit_bound to theta0_diagnostics builder and keep column ordering consistent.
    status: completed
  - id: update-summary-counts
    content: Replace DELTA_INAPPLICABLE with INSUFFICIENT_SIGNAL in summary helper/count sheet and non-OK aggregation.
    status: completed
  - id: validate-and-regression-check
    content: Run py_compile/lints and targeted checks for status strings, theta_applied behavior, and unaffected older dose-0 cohorts.
    status: completed
isProject: false
---

# Replace DELTA_INAPPLICABLE With INSUFFICIENT_SIGNAL

## Scope

- Implement only in `[code/KCOR.py](code/KCOR.py)`.
- Keep delta computation, quiet-window masking, `INSUFFICIENT_DEATHS`, and `NOT_IDENTIFIED` logic intact.
- Remove single-window fallback refit and replace with zero-theta insufficient-signal response.

## 1) Update `fit_theta0_gompertz` trigger and fallback behavior

- In `[fit_theta0_gompertz](code/KCOR.py)`, replace first-pass negative-delta fallback block with an `INSUFFICIENT_SIGNAL` zero-theta path using this explicit order:
  1. Run first reconstruction pass.
  2. If first-pass `delta_negative` is true, return immediately with `INSUFFICIENT_SIGNAL` and `theta_applied=0`.
  3. Otherwise run normal iteration to obtain final `theta0_raw`.
  4. If `theta0_raw >= theta0_max * 0.999`, return `INSUFFICIENT_SIGNAL` with `theta_applied=0`.
  5. Else return normal `OK` result.
  - This preserves early exit for negative-delta cohorts and applies the bound-hit check only when `theta0_raw` is available.
  - In all cases, keep full diagnostics payload populated (`delta_raw_values`, contamination fields, etc.).
- Remove the code path that re-runs `_fit_theta(...)` as single-window fallback for delta inapplicability.

## 2) Carry new status/flags downstream

- In both params-building paths in `[process_workbook](code/KCOR.py)` (per-age and all-ages):
  - Replace `delta_inapplicable` usage with `insufficient_signal` equivalents.
  - Add `theta_hit_bound` persisted from fitter diagnostics.
  - Keep `theta0_status` priority ladder as:
    1. `NOT_IDENTIFIED`
    2. `INSUFFICIENT_DEATHS`
    3. `INSUFFICIENT_SIGNAL`
    4. `OK`
- Update log tags/messages to eliminate `DELTA_INAPPLICABLE` naming.

## 3) Ensure theta=0 identity correction behavior

- Verify explicitly in `[process_workbook](code/KCOR.py)` that `theta=0` yields identity correction:
  - `h_adj = h0 / (1 + theta * H0_eff)` reduces to `h_adj = h0` at `theta=0`.
  - No divide-by-theta path is executed when theta is zero.
  - Any conditional branch (`if theta > 0 ... else raw`) is acceptable only if the `else` path is exact passthrough.
- Add only minimal defensive checks if needed; no formula changes.

## 4) Update diagnostics sheet builder

- In `[build_theta0_diagnostics_rows](code/KCOR.py)`:
  - Add `theta_hit_bound` column from `params_dict`.
  - Keep current dynamic wave columns and existing ordering, inserting `theta_hit_bound` with core scalar diagnostics.
  - Keep `k_hat_vs_dose0_ratio` behavior unchanged where already implemented.

## 5) Update summary status counts and helpers

- In `[create_summary_file](code/KCOR.py)`:
  - Replace `DELTA_INAPPLICABLE` count column with `INSUFFICIENT_SIGNAL` in `theta0_status_summary` and any `any_non_ok_status` logic.
  - Update `_theta0_status_from_params` helper to map new flag names.
  - Update `dual_print` text so no `DELTA_INAPPLICABLE` label remains.

## 6) Clean strings/comments/docstrings

- Replace user-facing/internal status strings/comments in `[code/KCOR.py](code/KCOR.py)`:
  - `delta_inapplicable` → `insufficient_signal`
  - `DELTA_INAPPLICABLE` → `INSUFFICIENT_SIGNAL`
- Update explanatory comments to reflect “weak/insufficient frailty signal (negative delta or bound-hit), theta_applied=0”.

## 7) Verification

- Run syntax and lint checks on `[code/KCOR.py](code/KCOR.py)`.
- Run targeted checks to confirm:
  - no `delta_inapplicable` string remains in outputs/log/status fields,
  - previously affected cohorts now have `theta_applied=0` and `theta_fit_status='insufficient_signal'`,
  - `theta_hit_bound=True` appears on bound-hit cohorts,
  - older dose-0 cohorts’ KCOR values remain unchanged.


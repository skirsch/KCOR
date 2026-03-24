---
name: KCOR v7.4 theta0 upgrade
overview: Replace the current Gompertz single-pass theta fit with the v7.4 delta-iteration estimator using NPH-consistent `k_anchor = h_nph(skip_week)`, YAML-configurable `k_anchor_tolerance` bounds for first-window `(k,theta)` fit, explicit gap-defined wave boundaries from `theta_estimation_windows`, first-window seeding, and discrete `H_gom`.
todos:
  - id: upgrade-estimator-core
    content: Implement v7.4 delta-iteration theta estimator and helper functions in code/KCOR.py.
    status: completed
  - id: wire-existing-call-sites
    content: Keep existing call sites working with compatible diagnostics for sensitivity/per-age/all-ages fits.
    status: completed
  - id: version-and-docs
    content: Update version marker/history and README notes to describe v7.4 method change.
    status: completed
  - id: validate-paths
    content: Run theta-related tests/checks and verify no lints/regressions in touched files.
    status: completed
isProject: false
---

# KCOR v7.4 Theta0 Delta-Iteration Plan

## Scope and intent

Implement the v7.4 theta0 estimator in the production KCOR pipeline by replacing the current `fit_theta0_gompertz` flow with an iterative method that:

- uses a two-stage `k` procedure on the first quiet window:
  - anchor read `k_anchor = h_nph(skip_week)` from the same NPH-corrected hazard series used by theta fitting,
  - bounded joint fit `(k, theta)` on pre-first-wave quiet points with `k` constrained to `[k_anchor*(1-k_anchor_tolerance), k_anchor*(1+k_anchor_tolerance)]` and `theta` to `[0, theta0_max]` (`k_anchor_tolerance` from YAML; e.g., `0.20` gives `[0.8, 1.2] * k_anchor`),
- fixes this fitted `k` for all subsequent delta-iteration steps (no re-estimation of `k` in iterations),
- uses the first-window fitted `theta` as the explicit iteration seed (`theta_0`),
- reconstructs `H0_eff` from observed hazard using current `theta`,
- computes cumulative/decomposed deltas at each gap-end wave boundary,
- refits one `theta` over all quiet windows with accumulated deltas,
- repeats until convergence.

## Target code locations

- Core estimator + helpers in [code/KCOR.py](code/KCOR.py)
- YAML parsing defaults (if needed for iteration controls) in [code/KCOR.py](code/KCOR.py)
- Dataset config (only if adding optional knobs) in [data/Czech/Czech.yaml](data/Czech/Czech.yaml)
- User-facing method/version notes in [README.md](README.md)

## Implementation details

- Replace/upgrade `fit_theta0_gompertz` to a v7.4 iterative estimator while keeping its call signature compatible for existing call sites.
- Add internal helper logic in `KCOR.py` to:
  - build sorted quiet intervals from `theta_estimation_windows` (ISO windows),
  - define wave intervals as the calendar gaps between consecutive quiet windows,
  - define each wave boundary using explicit gap indexing: if `t_next_quiet_start` is the first week of the next quiet window, set `t_gap_end = t_next_quiet_start - 1` and compute `delta_i = H0_eff_recon[t_gap_end] - H_gom[t_gap_end]`,
  - compute initial `(k, theta)` from the first quiet window only (pre-first-gap seed), with bounded `k` around `k_anchor`,
  - keep `k` fixed after seed fitting; update only `theta` and `delta_i` during iteration,
  - build `H_gom` as a discrete running sum from `k` (`sum_{u < t} k*exp(gamma*u)`), not the analytic closed form,
  - reconstruct `H0_eff` via frailty inversion recurrence from `h_obs` and candidate `theta`,
  - compute cumulative excess and convert to incremental `delta_i`, clamping `delta_i < 0` to `0` with per-iteration warning logs,
  - add a post-convergence diagnostic warning if any wave remains persistently negative before clamping (possible model misspecification or quiet-window misclassification),
  - build accumulated `Delta(t)` and refit `theta` over all quiet points using `H_gom(t)+Delta(t)`.
- Keep downstream frailty application unchanged (`invert_gamma_frailty` remains the same), so only theta estimation behavior changes.
- Preserve existing guards/logging integration (`min_quiet_deaths`, degenerate guard, unidentifiable log) by returning diagnostics compatible with current call-site expectations (`success`, `n_fit`, `status`, `rmse_hazard`, `relRMSE_hazard_fit`, `fit_mask_theta`, `k_hat`).
- Add explicit iteration controls in config/defaults (`convergence_tol`, `max_iterations`) and surface values in diagnostics/logs.
- Add/propagate an identifiability diagnostic flag for low-information cohorts (low mortality / short quiet windows), where numerical convergence may still be geometrically weak.
- Add/propagate `k_hit_bound` in estimator diagnostics/return payload; call sites should log a warning when `True` (indicates weak first-window conditioning for `k` refinement).
- Ensure all existing v7 call sites use the upgraded estimator consistently:
  - sensitivity sweep path,
  - per-age cohort fits,
  - all-ages fits.
- Bump code version marker to `v7.4` (the `VERSION` constant in `code/KCOR.py`), add/update the v7.4 entry in in-file version history comments, and document v7.4 behavior in README release notes.

## Validation

- Run focused checks that currently exercise theta fitting paths (at minimum the theta test target and one representative KCOR run path).
- Run synthetic ground-truth validation using [test/theta0/theta0_nwave.py](test/theta0/theta0_nwave.py) against the upgraded estimator; require `< 0.5%` theta recovery error on reference cases (e.g., ~6%/yr mortality, theta=2, two-wave setup).
- Confirm no regression in output schema fields (`theta0_raw`, `theta_applied`, `theta_fit_status`, diagnostics columns).
- Verify lints for touched files.


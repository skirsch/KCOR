# Alpha Estimation Sandbox

This directory contains a standalone sandbox for estimating a pandemic heterogeneity exponent `alpha` from cross-cohort COVID-wave distortion after accounting for cohort frailty state.

The sandbox is intentionally separate from the production KCOR pipeline. It reuses KCOR theta fitting and gamma-frailty helpers, but does not reuse the legacy production COVID-correction path except as historical context.

## Goals

- build a cohort-week table during the COVID interval
- propagate cohort frailty state from a pre-wave theta anchor
- estimate `alpha` with two estimators:
  - pairwise log-ratio fit
  - common-wave collapse fit
- test robustness to:
  - age-band restriction
  - early-wave vs late-wave segmentation
  - excess-hazard handling
  - reference-hazard anchor choice
  - leave-one-cohort-out influence
- validate recovery against synthetic data with known `alpha_true`

## Usage

From the repository root:

```bash
make alpha
```

Or from this directory:

```bash
make all
```

## Outputs

Outputs are written to `test/alpha/out/` and include:

- `alpha_cohort_diagnostics.csv`
- `alpha_wave_table.csv`
- `alpha_objective_curves.csv`
- `alpha_best_estimates.csv`
- `alpha_theta_scale_summary.csv`
- `alpha_leave_one_out.csv`
- `alpha_leave_one_out_summary.csv`
- `alpha_bootstrap.csv`
- `alpha_bootstrap_summary.csv`
- `alpha_synthetic_recovery.csv`
- `alpha_synthetic_vaccine_effect_recovery.csv`
- `alpha_synthetic_vaccine_effect_summary.csv`
- `alpha_synthetic_vaccine_effect_report.md`
- `alpha_conditional_VE_estimates.csv`
- `alpha_conditional_VE_summary.csv`
- `alpha_conditional_VE_report.md`
- `alpha_run_artifact.json`
- `alpha_primary_sensitivity_slices.csv`
- `alpha_decision_summary.csv`
- `alpha_calibration_choice.csv`
- `alpha_identifiability_report.md`
- `alpha_neutralization_mode_comparison.csv`
- `alpha_neutralization_comparison_report.md`
- `alpha_neutralization_run_artifact.json`
- `fig_alpha_objectives.png`
- `fig_alpha_neutralization_mode_comparison.png`
- `fig_alpha_synthetic_recovery.png`
- `fig_alpha_synthetic_vaccine_effect.png`
- `fig_alpha_vs_assumed_VE.png`

## Interpretation

The estimated `alpha` should be interpreted as a model-calibrated pandemic heterogeneity exponent, not as a direct biological constant.

## v7.6 Sandbox Contract

- The default search grid is `alpha in [1.00, 1.30]` with step `0.005`.
- The current primary Czech identification branch is restricted to doses `0` and `2`.
- Excess hazard is defined as `h_excess,d(t) = h_d(t) - h_ref(t)`.
- The default sandbox run compares `reference_anchored` against `symmetric_all_cohorts` NPH neutralization without modifying production `code/KCOR.py`.
- The primary legacy-style sandbox outputs remain keyed to `primary_nph_neutralization_mode: reference_anchored`.
- The comparison run enforces identical cohort-week rows, alpha grid, anchor, excess mode, and theta propagation across neutralization modes.
- The sandbox performs an explicit `alpha = 1` invariance check and fails if neutralization is not identity at that point.
- `alpha_best_estimates.csv` now separates the raw optimum (`alpha_hat`) from the reportable optimum (`alpha_hat_reported`).
- If the optimum occurs at `1.00` or `1.30`, the curve is treated as boundary-seeking and not reportable without a justified wider rerun.
- If the objective minimum is too flat, or if primary pairwise/collapse estimates disagree beyond the configured tolerance, the sandbox reports `not_identified` rather than silently promoting a numeric alpha.
- If identification fails and the dataset YAML provides `NPH_correction.default_alpha`, the sandbox records a separate external-calibration choice rather than silently relabeling that value as identified.
- `alpha_run_artifact.json` records the config, seeds, cohort selection footprint, curve diagnostics, and primary identification status for auditability.
- `alpha_neutralization_comparison_report.md` is sandbox-only and should not be treated as a production or manuscript result.

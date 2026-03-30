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
- `alpha_run_artifact.json`
- `alpha_primary_sensitivity_slices.csv`
- `alpha_decision_summary.csv`
- `alpha_identifiability_report.md`
- `fig_alpha_objectives.png`
- `fig_alpha_synthetic_recovery.png`

## Interpretation

The estimated `alpha` should be interpreted as a model-calibrated pandemic heterogeneity exponent, not as a direct biological constant.

## v7.6 Sandbox Contract

- The default search grid is `alpha in [1.00, 1.30]` with step `0.005`.
- Excess hazard is defined as `h_excess,d(t) = h_d(t) - h_ref(t)`.
- `alpha_best_estimates.csv` now separates the raw optimum (`alpha_hat`) from the reportable optimum (`alpha_hat_reported`).
- If the optimum occurs at `1.00` or `1.30`, the curve is treated as boundary-seeking and not reportable without a justified wider rerun.
- If the objective minimum is too flat, or if primary pairwise/collapse estimates disagree beyond the configured tolerance, the sandbox reports `not_identified` rather than silently promoting a numeric alpha.
- `alpha_run_artifact.json` records the config, seeds, cohort selection footprint, curve diagnostics, and primary identification status for auditability.

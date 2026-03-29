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
- `alpha_leave_one_out.csv`
- `alpha_bootstrap.csv`
- `alpha_synthetic_recovery.csv`
- `fig_alpha_objectives.png`
- `fig_alpha_synthetic_recovery.png`

## Interpretation

The estimated `alpha` should be interpreted as a model-calibrated pandemic heterogeneity exponent, not as a direct biological constant.

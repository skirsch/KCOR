# KCOR v7 Specification: Time-Varying Theta

## Overview

KCOR v6 estimated a **fixed theta (θ)** for the unvaccinated cohort from a single 2023 quiet
window and applied that constant value throughout the entire follow-up period. This is
methodologically incorrect because θ is not constant — it declines continuously as frailty
depletion progressively removes high-frailty individuals from the risk set.

KCOR v7 replaces the constant-θ assumption with a **time-varying θ(t)** derived from the
closed-form gamma frailty depletion formula, anchored by a single estimated θ(0).

---

## Why Constant Theta Is Wrong

Under gamma frailty, the unvaccinated cohort has higher baseline mortality and therefore
faster frailty depletion than the vaccinated cohort. This means:

- The true effective θ is **high at early timepoints** (2021) when high-frailty individuals
  are still alive
- θ declines continuously as high-frailty individuals are selectively depleted
- By 2023, the surviving population is more homogeneous and θ is materially lower

Using the 2023 θ estimate for all timepoints **under-corrects at early timepoints**. This
causes the unvaccinated hazard to appear artificially low (frailty depletion not fully
accounted for), making the vaccinated cohort look relatively worse than it truly is — i.e.,
the correction is biased **against** the vaccine at early timepoints.

---

## Closed-Form Theta Trajectory

Under gamma frailty, the variance of the surviving frailty distribution at time t is:

```
θ(t) = θ(0) / (1 + θ(0)·H(t))²
```

Where:
- `θ(0)` — frailty variance at cohort inception (the single parameter to estimate)
- `H(t)` — cumulative baseline hazard at time t (observed directly from data)

The **entire θ trajectory is fully determined** by θ(0) and the observed H(t). No additional
parameters are needed.

---

## Inverting to Recover θ(0) from Any Timepoint

Since the closed form holds at any timepoint t*, we can recover θ(0) from an observation
at any t*. The inversion yields a quadratic with two roots — the **minus branch** is the
physically correct one, selected by the continuity argument: as H(t*) → 0, θ(0) must
approach θ(t*) (no depletion has occurred). Only the minus branch has this property.

The naive minus-branch form suffers catastrophic cancellation when θ(t*)H(t*) is small.
The numerically stable equivalent form is:

```
θ(0) = 2θ(t*) / [1 - 2θ(t*)·H(t*) + √(1 - 4θ(t*)·H(t*))]
```

This is the form implemented in `theta0_from_anchor()` in KCOR.py.

### Why not the plus root
The plus root diverges as H(t*) → 0 and produces large, unstable θ(0) estimates
inconsistent across anchor points. It is immediately detectable as the wrong branch via
the consistency diagnostic.

### Validity condition
The inversion requires:

```
1 - 4θ(t*)·H(t*) ≥ 0
i.e. H(t*) ≤ 1 / (4θ(t*))
```

Anchor points violating this condition are skipped and counted in the diagnostic report.
In practice this is satisfied for reasonable cumulative hazards and frailty variances.

---

## Estimation Strategy for θ(0)

### Key insight
Every timepoint t in every quiet window yields an **independent estimate of θ(0)** via the
inversion formula. All estimates should agree if the gamma frailty model is correctly
specified.

### Algorithm

```
For each quiet window w:
    1. Fit one nonlinear regression across all timepoints in w
       → yields θ_anchor for window w  [via fit_k_theta_cumhaz]
    2. For each timepoint t in w:
       a. Compute θ(t) from closed form using θ_anchor and H(t)
       b. Apply stable minus-branch inversion → θ(0) estimate at timepoint t
          [via theta0_from_anchor(theta_t, h_t)]
       c. Skip and record if discriminant check fails

Pool all θ(0) estimates across all windows and all timepoints:
    → Single MLE estimate of θ(0)
    → 95% CI from variance of estimates
    → Goodness-of-fit diagnostic from scatter/drift across windows
```

### Why multiple quiet windows are better than one
- Each window contributes independent estimates — more data, more precise θ(0)
- Windows are separated in calendar time — cross-window agreement is a strong
  consistency test
- Window boundaries are **arbitrary** — θ(0) is a structural parameter invariant to
  window choice
- No assumption that θ is constant within any window — the closed form handles
  within-window variation automatically

### What counts as a quiet window
A quiet window is a period where:
- COVID mortality is negligible
- Observed mortality reflects baseline frailty-driven deaths only
- H(t) accumulation is uncontaminated by COVID signal

---

## Computation Steps

1. **Estimate θ(0)** from all specified quiet windows using the pooling algorithm above.
   All windows should produce consistent estimates — disagreement signals model
   misspecification.

2. **Compute θ(t) at every timepoint** in the full follow-up period using:
   ```
   θ(t) = θ(0) / (1 + θ(0)·H(t))²
   ```
   This is implicit inside `invert_gamma_frailty(H_obs, theta0_hat)` which receives
   the already-determined θ(0) and applies the correction using the full H(t) trajectory.

3. **Compute adjusted H(t) at every timepoint** via `invert_gamma_frailty(H_obs, theta0_hat)`:
   ```
   H_adj(t) = (e^(θ(0)·H(t)) - 1) / θ(0)
   ```
   implemented as `np.expm1(theta * H_arr) / theta` for numerical stability at small θH.

---

## Code Architecture

Two distinct functions with distinct responsibilities:

### `theta0_from_anchor(theta_t, h_t)`
- Performs branch selection (minus branch)
- Uses numerically stable form: `θ(0) = 2θ(t) / [1 - 2θ(t)H(t) + √(1 - 4θ(t)H(t))]`
- Guards: non-finite input, discriminant check, `max(disc, 0.0)` for float near-zero
- Returns `(θ(0), success_bool, reason_string)`

### `invert_gamma_frailty(H_obs, theta)`
- Separate concern — takes θ(0) as already-determined input
- No branch selection needed here
- Applies `np.expm1(theta * H_arr) / theta` for numerical stability
- Guards near-zero θ via `KCOR6_THETA_EPS`

The separation is intentional: branch selection happens once during θ(0) estimation;
the correction application is a straightforward forward pass given θ(0).

---

## YAML Configuration

```yaml
time_varying_theta:
  enabled: true
  apply_to: both_cohorts           # unvaccinated_only | vaccinated_only | both_cohorts
  estimate_from: unvaccinated      # which cohort's quiet windows to use for θ(0) estimation

  theta_estimation_windows:
    - start: "2021-26"
      end: "2021-36"
      description: "pre-Delta quiet period"
    - start: "2022-22"
      end: "2022-25"
      description: "inter-wave quiet period"
    - start: "2023-22"
      end: "2023-37"
      description: "post-COVID quiet period"

  diagnostics:
    plot_theta0_estimates_by_window: true
    plot_theta_trajectory: true
    report_consistency_test: true
```

---

## Diagnostics

### 1. `plot_theta0_estimates_by_window`

Plots the individual θ(0) estimates derived from each timepoint via the inversion formula.

- **X axis**: calendar time
- **Y axis**: θ(0) estimate at that timepoint
- **One dot per timepoint**, color-coded by window
- **Horizontal line**: pooled MLE θ(0)
- **Shaded band**: 95% CI around pooled estimate

**What to look for:**
- All dots scattered randomly around the pooled line → model well specified
- Systematic drift across time → gamma frailty misspecified
- One window consistently offset from others → that window may not be truly quiet
- High scatter within a window → noisy data in that window

### 2. `plot_theta_trajectory`

Plots the full θ(t) curve from cohort inception to end of follow-up.

- **X axis**: calendar time
- **Y axis**: θ(t) value
- **Blue curve**: θ(t) = θ(0) / (1 + θ(0)·H(t))² evaluated at every timepoint
- **Shaded CI band**: propagated uncertainty from θ(0) CI
- **Red horizontal line**: constant θ used in v6 (for comparison)
- **Vertical markers**: quiet windows, vaccine rollout dates, COVID wave periods

**What to look for:**
- Monotonically declining curve — confirms frailty depletion is operating as expected
- Steeper decline during high-mortality periods (COVID waves)
- Gap between red v6 line and blue v7 curve quantifies the under-correction in v6
- The gap is largest at early timepoints where under-correction was most severe

### 3. `report_consistency_test`

Statistical report testing whether all quiet windows produce consistent θ(0) estimates.

**Output:**
```
θ(0) estimates by window:

  Window 2021-26:2021-36   mean=X.XX   sd=X.XX   n=10
  Window 2022-22:2022-25   mean=X.XX   sd=X.XX   n=4
  Window 2023-22:2023-37   mean=X.XX   sd=X.XX   n=16

  Pooled θ(0) MLE:         X.XX
  95% CI:                  [X.XX, X.XX]

  Cross-window variance:   X.XXX
  Within-window variance:  X.XXX
  F-statistic:             X.XX
  p-value (drift test):    X.XX  → [no significant drift | significant drift detected]

  Mann-Kendall trend test: tau=X.XX   p=X.XX  → [no trend | trend detected]
```

**Interpretation:**

| Result | Interpretation |
|---|---|
| All windows agree, no drift | Gamma frailty well specified, θ(0) trustworthy |
| High within-window scatter | Data quality issue or window not truly quiet |
| Cross-window disagreement | Structural change between periods or misspecification |
| Monotonic drift in θ(0) | Frailty distribution may not be truly gamma |

---

## Comparison to v6

| Property | KCOR v6 | KCOR v7 |
|---|---|---|
| θ assumption | Constant throughout follow-up | Time-varying via closed form |
| Estimation window | Single 2023 window | Multiple quiet windows, any period |
| Window boundary sensitivity | High — different windows give different θ | None — θ(0) is window-invariant |
| Correction at early timepoints | Under-corrected (θ too low) | Correctly corrected via θ(t) |
| Bias direction at early timepoints | Against vaccine | Removed |
| Goodness-of-fit test | None | Built-in via cross-window consistency |
| Additional parameters to estimate | 0 | 0 (θ(0) replaces constant θ) |

---

## Notes

- The number of parameters to estimate does **not increase** from v6 to v7 — both estimate
  a single scalar. v7 simply uses it more correctly via the closed-form trajectory.
- The closed-form inversion means no numerical optimization is needed for θ(0) estimation
  at each timepoint — it is a direct calculation.
- The pooled MLE across timepoints can be computed as a simple weighted mean or via
  full MLE depending on desired precision.
- If `apply_to: both_cohorts`, each cohort gets its own independent θ(0) estimation
  from its own quiet window data, reflecting the different depletion rates of the two cohorts.

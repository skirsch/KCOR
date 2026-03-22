# KCOR v7: Static HVE Normalization via Gompertz-Frailty Depletion

## What KCOR v7 does

KCOR v7 normalizes cohort cumulative hazards to remove selection-induced
depletion bias (static healthy vaccinee effect / static HVE) before computing
cumulative outcome ratios. The upgrade from v6 is a single targeted change:
**theta estimation now uses the correct Gompertz+frailty-depletion model with
k fixed from enrollment, rather than a flat-baseline constant-k model from a
single late quiet window.** The downstream gamma inversion and KCOR ratio
computation are unchanged.

---

## The depletion problem

Under gamma frailty, the observed marginal hazard for a fixed cohort is:

$$h_{\text{obs}}(t) = \frac{h_0(t)}{1 + \theta_d \, H_{0,d}(t)}$$

where $\theta_d > 0$ is the frailty variance at enrollment for dose group $d$
and $H_{0,d}(t)$ is the true cumulative baseline hazard. As time progresses,
high-frailty individuals die preferentially, the risk set becomes
progressively healthier, and $h_{\text{obs}}(t)$ decelerates relative to
$h_0(t)$. When $\theta_d$ differs across cohorts — as it does whenever
selection concentrates or depletes frail individuals at enrollment — the
cohorts exhibit **non-proportional hazards** and naive cumulative comparisons
are biased.

---

## The correct baseline model

Human all-cause mortality follows the Gompertz law at the population level:
$h_0(t) = k \, e^{\gamma t}$, with $\gamma \approx 0.085$ per year
($\approx 0.00163$ per week). For a fixed cohort observed over 2–3 years,
ignoring this aging trend causes the optimizer to absorb the Gompertz rise
into $\hat\theta$, systematically biasing the estimate. The correct marginal
hazard model for a fixed cohort under gamma frailty with Gompertz baseline is:

$$h_{\text{obs}}(t) = \frac{k \, e^{\gamma t}}{1 + \theta_d \, H_{\text{gom},d}(t)}$$

where the Gompertz cumulative hazard is:

$$H_{\text{gom},d}(t) = \frac{k_d}{\gamma}\!\left(e^{\gamma t} - 1\right)$$

---

## Estimating $\hat\theta_d$: the v7 procedure

**Step 1 — Fix $k_d$ from enrollment (no free parameter).**

The time axis is rebased so $t = 0$ corresponds to the first accumulating
week (i.e., $t_{\text{raw}}$ shifted by `DYNAMIC_HVE_SKIP_WEEKS`):

$$t_{\text{rebased}} = t_{\text{raw}} - t_{\text{skip}}$$

At $t_{\text{rebased}} = 0$, the denominator equals 1 and
$h_{\text{obs}}(0) = k_d$ exactly. The baseline hazard is therefore fixed
directly from observed data:

$$\hat k_d = \frac{1}{W}\sum_{j=0}^{W-1} h_{\text{eff}}(j)$$

where the sum runs over the $W$ weeks immediately following the skip period
(`k_anchor_weeks`, default $W = 4$). These anchor weeks are also included as
regression targets in the fit below.

**Step 2 — Estimate $\hat\theta_d$ by single-parameter nonlinear least squares.**

With $\hat k_d$ and $\gamma$ both fixed, $\theta_d$ is the only free
parameter. The fit uses all timepoints in the union of the quiet estimation
windows and the anchor window:

$$\text{fit\_mask} = (\text{quiet\_mask} \cup \text{anchor\_mask}) \cap \{t_{\text{rebased}} \ge 0\} \cap \{h_{\text{eff}} \text{ finite}\}$$

The estimator minimizes the sum of squared residuals over fit\_mask:

$$\hat\theta_d = \arg\min_{\theta \ge 0} \sum_{t \in \text{fit\_mask}} \left[ h_{\text{eff}}(t) - \frac{\hat k_d \, e^{\gamma t_{\text{rebased}}}}{1 + \theta \, \hat H_{\text{gom},d}(t_{\text{rebased}})} \right]^2$$

using bounded nonlinear least squares (`scipy.optimize.least_squares`,
method `trf`, $\theta \ge 0$).

**Why anchor weeks are included as targets.**
$\hat k_d$ is a summary statistic of the anchor weeks, not a held-out block.
Including them as residual points keeps the fit self-consistent at
$t_{\text{rebased}} \approx 0$ (where the model must satisfy
$h \approx \hat k_d$) and wastes no information.

**Quiet windows** are prespecified calendar intervals free of COVID wave
mortality (empirically $<1\%$ of all-cause deaths attributable to COVID).
Three summer windows are used in the Czech application:
2021-W26–W36, 2022-W22–W25, and 2023-W19–W40. H(t) accumulates
continuously through wave periods; only h(t) targets are restricted to
quiet intervals.

**Identifiability condition.**
For cohorts where $\hat k_d$ is very small (young birth years with low
baseline mortality), the Gompertz model is mathematically unidentifiable:
flattening the Gompertz rise requires $\theta \approx \gamma / \hat k_d$,
which can exceed the plausibility bound. If $\gamma / \hat k_d > \theta_{\max}$
(default $\theta_{\max} = 100$), the cohort is flagged
`KCOR7_GOMPERTZ_UNIDENTIFIABLE` and $\hat\theta_d$ is set to zero
(no correction applied). If the optimizer returns $\hat\theta_d > \theta_{\max}$
for other reasons, it is flagged `KCOR7_DEGENERATE` and similarly zeroed.

---

## Applying the correction: gamma inversion

Once $\hat\theta_d$ is obtained, the depletion-neutralized cumulative hazard
is computed by the gamma-frailty inversion (unchanged from v6):

$$\tilde H_{0,d}(t) = \frac{\exp\!\bigl(\hat\theta_d \, H_{\text{obs},d}(t)\bigr) - 1}{\hat\theta_d}$$

implemented as `np.expm1(theta * H_obs) / theta` for numerical stability.
Here $H_{\text{obs},d}(t)$ is the observed cumulative hazard accumulated over
the full follow-up including COVID wave periods — correctly incorporating
all real frailty depletion, including wave-accelerated depletion of frail
individuals, into the correction.

When $\hat\theta_d = 0$, the inversion reduces to
$\tilde H_{0,d}(t) = H_{\text{obs},d}(t)$ (identity, no correction).

---

## KCOR ratio

Cohorts are compared via the ratio of depletion-neutralized cumulative
hazards, normalized to unity at a prespecified reporting baseline:

$$\mathrm{KCOR}(t) = \frac{\tilde H_{0,A}(t)}{\tilde H_{0,B}(t)}$$

A flat trajectory ($\mathrm{KCOR}(t) \approx 1$) indicates no divergence
between cohorts after accounting for selection-induced depletion.
Deviations from unity represent cumulative outcome differences that persist
after frailty normalization.

---

## Key parameters (YAML)

```yaml
time_varying_theta:
  gompertz_gamma:   0.085     # Gompertz aging constant, per year (fixed biological constant)
  k_anchor_weeks:   4         # weeks post-skip used to fix k_d
  degenerate_fit:
    theta0_max:     100       # theta above this → set to 0, log warning
    action:         set_zero
  min_quiet_deaths: 30        # minimum deaths in quiet windows to attempt fit

theta_estimation_windows:     # quiet calendar intervals (ISO year-week)
  - [2021-26, 2021-36, "post primary"]
  - [2022-22, 2022-25, "post booster"]
  - [2023-19, 2023-40, "post COVID"]
```

---

## What does NOT change from v6

- `invert_gamma_frailty(H_obs, theta)` — the gamma inversion formula is identical
- KCOR ratio computation, normalization weeks, bootstrap CIs
- Dynamic HVE skip (`DYNAMIC_HVE_SKIP_WEEKS`)
- COVID wave correction (`COVID_CORRECTION`)
- All output sheet formats and downstream analysis

---

## Why v7 is better than v6

| Property | v6 | v7 |
|---|---|---|
| Baseline model | Flat $k$ (constant) | Gompertz $k e^{\gamma t}$ |
| Free parameters | 2 ($k$, $\theta$) | 1 ($\theta$ only) |
| $k$ estimation | Joint fit — confounded with $\theta$ | Fixed from first $W$ weeks — clean separation |
| Estimation windows | Single 2023 quiet window | All quiet windows simultaneously |
| $\theta$ estimate anchored at | Late depleted population | Enrollment (t=0) |
| Bias direction | Under-estimates $\theta$; under-corrects | Correct |
| Monotonicity of $\hat\theta$ by age | Violated for many enrollment dates | Holds for YoB 1920–1970 across all enrollment dates |
| Vaccinated $\hat\theta$ | Near zero by construction | Non-zero where frailty signal exists |
| Negative control (mid-2022) | Approximately flat | Flat near 1.0 across all dose comparisons |

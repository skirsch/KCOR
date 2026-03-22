# Box: KCOR v7 in two pages

KCOR (Kirsch Cumulative Outcomes Ratio) is a depletion-neutralized cohort
comparison framework that removes selection-induced frailty bias before
computing cumulative outcome ratios. It operates on aggregated registry
data — dates of birth, intervention, and death — without requiring
individual-level covariates. The method proceeds in seven steps.

---

## 1. Fixed cohorts

Individuals are assigned to cohorts at enrollment based on intervention
status (e.g., dose received). **No post-enrollment switching or censoring
is permitted.** Individuals remain in their enrollment cohort for the full
follow-up. This prevents immortal-time artifacts and keeps frailty
composition stable within each cohort.

---

## 2. Discrete-time hazard and cumulative hazard

For each cohort $d$ and weekly interval $t$, the discrete-time hazard is:

$$h_{\mathrm{obs},d}(t) = -\ln\!\left(1 - \frac{d_d(t)}{N_d(t)}\right)$$

where $d_d(t)$ is deaths in interval $t$ and $N_d(t)$ is the risk set.
An optional stabilization skip (step 4 below) zeros out early weeks to
remove immediate post-enrollment artifacts. Cumulative hazards are
accumulated from the **effective hazard** $h_d^{\mathrm{eff}}(t)$ —
meaning the observed hazard after applying the stabilization skip (zero
during the skip period, equal to $h_{\mathrm{obs},d}(t)$ thereafter):

$$H_{\mathrm{obs},d}(t) = \sum_{s \le t} h_d^{\mathrm{eff}}(s)$$

---

## 3. The frailty depletion problem

Under gamma frailty, individuals with latent frailty $z_{i,d}$ have
individual hazard $h_{i,d}(t) = z_{i,d}\,\tilde{h}_{0,d}(t)$. As
high-frailty individuals die first, the surviving risk set becomes
progressively healthier, causing the cohort-level hazard to decelerate
relative to the true baseline. The gamma-frailty model links the observed
and baseline cumulative hazards via:

$$H_{\mathrm{obs},d}(t) = \frac{1}{\theta_d}\ln\!\left(1 + \theta_d\,\tilde{H}_{0,d}(t)\right)$$

where $\theta_d \ge 0$ is the **frailty variance of cohort $d$ at enrollment
($t=0$)** — the variance of the initial gamma distribution before any
depletion has occurred — and $\tilde{H}_{0,d}(t)$ is the
depletion-neutralized baseline cumulative hazard. $\theta_d$ is a fixed
scalar: it characterizes the cohort's frailty heterogeneity at the moment
cohorts are defined. The time-varying narrowing of the surviving
population's frailty distribution as high-frailty individuals die is
captured entirely by $H_{\mathrm{obs},d}(t)$ in the inversion formula —
not by a time-varying $\theta_d$. When $\theta_d$ differs across cohorts —
as it does when selection concentrates or depletes frail individuals at
enrollment (static HVE) — naive cumulative comparisons are biased and
hazards are non-proportional.

---

## 4. Stabilization skip

The first `SKIP_WEEKS` intervals after enrollment are excluded from
accumulation to remove transient post-enrollment artifacts (dynamic HVE):

$$h_d^{\mathrm{eff}}(t) = \begin{cases} 0, & t < \mathrm{SKIP\_WEEKS} \\ h_{\mathrm{obs},d}(t), & t \ge \mathrm{SKIP\_WEEKS} \end{cases}$$

The time axis for frailty fitting is rebased so $t_{\mathrm{rebased}} = 0$
at the first accumulating week:

$$t_{\mathrm{rebased}} = t_{\mathrm{raw}} - \mathrm{SKIP\_WEEKS}$$

All frailty fitting uses $t_{\mathrm{rebased}}$ so that
$H_{\mathrm{gom},d}(0) = 0$ exactly at the first real data point.

---

## 5. Estimating $\hat{\theta}_d$: Gompertz-corrected quiet-window fit

The key insight of v7 is that $\theta_d$ — the **enrollment-time frailty
variance** — must be estimated from data anchored at $t_{\mathrm{rebased}}=0$,
not from a late quiet window that reflects an already-depleted population.
Estimating $\theta_d$ from late data systematically underestimates the
true enrollment-time frailty variance and produces an insufficient
correction.

Human all-cause mortality follows the Gompertz law
$\tilde{h}_{0,d}(t) = k_d\,e^{\gamma t}$ with $\gamma = 0.085$ per year
($\approx 0.00163$ per week), a fixed biological constant. Ignoring this
aging trend causes the optimizer to absorb the Gompertz rise into
$\hat{\theta}_d$, biasing the frailty estimate. The correct marginal hazard
for a fixed cohort under gamma frailty with Gompertz baseline is:

$$h_{\mathrm{obs},d}(t) = \frac{k_d\,e^{\gamma\,t_{\mathrm{rebased}}}}{1 + \theta_d\,H_{\mathrm{gom},d}(t_{\mathrm{rebased}})}$$

where the Gompertz cumulative hazard is:

$$H_{\mathrm{gom},d}(t_{\mathrm{rebased}}) = \frac{k_d}{\gamma}\!\left(e^{\gamma\,t_{\mathrm{rebased}}} - 1\right)$$

**Fixing $\hat{k}_d$ from enrollment.** At $t_{\mathrm{rebased}} = 0$ the
denominator equals 1 and $h_{\mathrm{obs},d}(0) = k_d$ exactly. The
baseline hazard is therefore read directly from the data as the mean of the
first $W$ post-skip weeks (`k_anchor_weeks`, default $W = 4$):

$$\hat{k}_d = \frac{1}{W}\sum_{j=0}^{W-1} h_d^{\mathrm{eff}}(j)$$

This eliminates $k_d$ as a free parameter entirely.

**Single-parameter fit.** With $\hat{k}_d$ and $\gamma$ both fixed,
$\theta_d$ is the only free parameter. The fit uses all timepoints in the
union of the quiet estimation windows and the anchor window:

$$\text{fit\_mask} = (\text{quiet\_mask} \cup \text{anchor\_mask}) \cap \{t_{\mathrm{rebased}} \ge 0\} \cap \{h_d^{\mathrm{eff}} \text{ finite}\}$$

The estimator minimizes squared residuals over fit\_mask using bounded
nonlinear least squares (`scipy.optimize.least_squares`, method `trf`,
$\theta_d \ge 0$):

$$\hat{\theta}_d = \arg\min_{\theta_d \ge 0} \sum_{t \in \text{fit\_mask}} \left[ h_d^{\mathrm{eff}}(t) - \frac{\hat{k}_d\,e^{\gamma\,t_{\mathrm{rebased}}}}{1 + \theta_d\,H_{\mathrm{gom},d}(t_{\mathrm{rebased}})} \right]^2$$

**Quiet windows** are prespecified calendar intervals free of epidemic wave
mortality (empirically $< 1\%$ of all-cause deaths attributable to the
external shock). $H_{\mathrm{obs},d}(t)$ accumulates continuously through
wave periods — correctly incorporating all real frailty depletion, including
wave-accelerated depletion — but $h_d^{\mathrm{eff}}(t)$ regression targets
are restricted to quiet intervals only. Multiple disjoint windows are used;
fitting them jointly with a single $\hat{\theta}_d$ also serves as an
internal consistency check.

**Identifiability.** For cohorts where $\hat{k}_d$ is very small, flattening
the Gompertz rise requires $\theta_d \approx \gamma/\hat{k}_d$, which can
exceed the plausibility bound $\theta_{\max}$ (default 100). If
$\gamma/\hat{k}_d > \theta_{\max}$, the cohort is flagged
`GOMPERTZ_UNIDENTIFIABLE` and $\hat{\theta}_d$ is set to zero (no
correction applied). If the optimizer returns $\hat{\theta}_d > \theta_{\max}$
for other reasons, it is flagged `DEGENERATE` and similarly zeroed. Both
conditions are logged explicitly.

---

## 6. Normalization (gamma inversion)

Given $\hat{\theta}_d$, the depletion-neutralized baseline cumulative
hazard is obtained by exact inversion of the gamma-frailty identity,
applied to the full post-enrollment trajectory of $H_{\mathrm{obs},d}(t)$:

$$\tilde{H}_{0,d}(t) = \frac{\exp\!\left(\hat{\theta}_d\,H_{\mathrm{obs},d}(t)\right) - 1}{\hat{\theta}_d}$$

implemented as `np.expm1(theta * H_obs) / theta` for numerical stability.
When $\hat{\theta}_d = 0$ this reduces to
$\tilde{H}_{0,d}(t) = H_{\mathrm{obs},d}(t)$ (identity; no correction).
Note that $H_{\mathrm{obs},d}(t)$ here is the full observed cumulative
hazard including wave periods — not restricted to quiet windows — so all
real depletion, including epidemic-wave-accelerated depletion of frail
individuals, is correctly incorporated into the correction.

---

## 7. KCOR ratio

Cohorts are compared via the ratio of depletion-neutralized cumulative
hazards, optionally anchored at a prespecified reference time $t_0$:

$$\mathrm{KCOR}(t) = \frac{\tilde{H}_{0,A}(t)}{\tilde{H}_{0,B}(t)}, \qquad \mathrm{KCOR}(t;\,t_0) = \frac{\mathrm{KCOR}(t)}{\mathrm{KCOR}(t_0)}$$

A flat trajectory ($\mathrm{KCOR}(t) \approx 1$) indicates no divergence
between cohorts after depletion normalization. Deviations from unity
represent cumulative outcome differences that persist after frailty
correction.

---

## Key parameters

| Parameter | Default | Role |
|---|---|---|
| `SKIP_WEEKS` | 2 | Dynamic HVE stabilization skip |
| `k_anchor_weeks` | 4 | Weeks post-skip used to fix $\hat{k}_d$ |
| `gompertz_gamma` | 0.085/yr | Gompertz aging constant (fixed biological constant) |
| `theta0_max` | 100 | Degenerate/unidentifiable fit threshold |
| `min_quiet_deaths` | 30 | Minimum deaths in quiet windows to attempt fit |
| `NORMALIZATION_WEEKS` | 4 | Reference window for KCOR anchoring |

---

## What changed from v6 to v7

The gamma inversion (step 6) and KCOR ratio (step 7) are **unchanged**.
The only change is step 5 — how $\hat{\theta}_d$ is estimated:

| Property | v6 | v7 |
|---|---|---|
| Baseline model | Flat $k_d$ (constant) | Gompertz $k_d e^{\gamma t}$ |
| Free parameters | 2 ($k_d$ and $\theta_d$) | 1 ($\theta_d$ only) |
| $\hat{k}_d$ source | Jointly fitted — confounded with $\hat{\theta}_d$ | Fixed from first $W$ post-skip weeks |
| Estimation windows | Single late quiet window | All quiet windows simultaneously |
| $\hat{\theta}_d$ anchored at | Late depleted population — underestimates true enrollment-time $\theta_d$ | Enrollment ($t_{\mathrm{rebased}} = 0$) — correct enrollment-time $\theta_d$ |
| Vaccinated $\hat{\theta}_d$ | Near zero by construction | Non-zero where frailty signal exists |
| Monotonicity of $\hat{\theta}_d$ by age | Frequently violated | Holds empirically (YoB 1920–1970) |

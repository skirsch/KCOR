# KCOR v7 Upgrade: Gompertz+Depletion Theta0 Estimation

## Overview

Replace the current v7 theta0 estimation method with a physically correct
Gompertz+frailty-depletion model. The H(t) adjustment formula (`invert_gamma_frailty`)
is **unchanged** — only the theta0 estimation changes.

### Implementation notes (this repository)

- **`theta_estimation_windows`** stays at the **dataset YAML root** as a list of triples `[start, end, optional_label]` (e.g. `data/Czech/Czech.yaml`). Do not nest a separate `theta_estimation_windows` block under `time_varying_theta` for Czech.
- **Time axis for the Gompertz fit:** at each call site, `t_rebased = t_vals - DYNAMIC_HVE_SKIP_WEEKS`. The fitter receives **`t_rebased`** so that **`H_gompertz(0) = 0`** at the first post–HVE-skip week (aligned with nonzero `hazard_eff`). Pipeline column **`t`** (weeks since enrollment) is otherwise unchanged.
- **Fit sample:** least-squares uses **`(quiet_mask | anchor_mask) & (t_rebased >= 0)`** with finite `h`, where **`anchor_mask = (t_rebased >= 0) & (t_rebased < k_anchor_weeks)`**. K-anchor weeks are **included** in the θ fit, not held out after fixing `k`.
- **`fit_theta0_gompertz`** in `code/KCOR.py` takes `(h_arr, t_rebased, quiet_mask, k_anchor_weeks, gamma_per_week)` — no separate `hve_skip` argument.

---

## Why the Current Method Is Wrong

The current v7 fits:
```
h_obs(t) = k / (1 + θ₀ · H(t))
```

with flat constant k. This is wrong because human mortality grows at ~8.5%/year
(Gompertz law). Over 2-3 years of follow-up, ignoring Gompertz causes the optimizer
to absorb the aging trend into theta0, producing biased estimates.

---

## The Correct Model

For a fixed cohort under gamma frailty with Gompertz baseline:

```
h_obs(t) = k · exp(γt) / (1 + θ₀ · H_gompertz(t))
```

Where the Gompertz cumulative hazard is:
```
H_gompertz(t) = (k/γ) · (exp(γt) - 1)
```

Parameters:
- `k` — baseline hazard at enrollment, **fixed from data** (not fitted)
- `γ` — Gompertz constant, **fixed at 0.085/year** (biological constant)
- `θ₀` — frailty variance at enrollment, the **single free parameter** to fit

This gives one free parameter (θ₀) instead of two (k, θ₀), making the
estimation more stable and correctly separating aging from frailty depletion.

---

## How k Is Fixed

`k` is pinned from the observed effective hazard immediately after the HVE skip:

```python
# Rebasing: t_rebased = t_enrollment - DYNAMIC_HVE_SKIP_WEEKS
anchor_mask = (t_rebased >= 0) & (t_rebased < k_anchor_weeks)
k = mean(h_arr[anchor_mask])   # h_arr is hazard_eff (same as post-skip policy in code)
```

The `k_anchor_weeks` parameter is configurable under `time_varying_theta` in the YAML (default: 4 weeks).

At **`t_rebased = 0`** (first post-skip week), **`H_gompertz(0) = 0`**, so under the model **`h ≈ k`** at that point. The anchor window averages `h` over the first few post-skip weeks before frailty depletion and Gompertz curvature dominate (e.g. four weeks at ~8.5%/year ≈ 0.65% effect, negligible).

---

## Implementation Instructions

### Step 1: Add YAML parameter

In the dataset YAML (e.g. `data/Czech/Czech.yaml`), add under
`time_varying_theta`:

```yaml
time_varying_theta:
  enabled: true
  apply_to: both_cohorts
  gompertz_gamma: 0.085          # per year — biological constant, do not change
  k_anchor_weeks: 4              # weeks after HVE skip to estimate k
  degenerate_theta_max: 100      # theta0 > this → set to 0
  theta_estimation_windows:
    - start: "2021-26"
      end:   "2021-36"
      description: "post primary quiet period"
    - start: "2022-22"
      end:   "2022-25"
      description: "post booster quiet period"
    - start: "2023-22"
      end:   "2023-37"
      description: "post COVID quiet period"
```

### Step 2: Parse new YAML parameters

In the YAML loading section of `KCOR.py`, parse:

```python
gompertz_gamma  = cfg.get("gompertz_gamma",  0.085)   # per year
k_anchor_weeks  = cfg.get("k_anchor_weeks",  4)        # weeks
degenerate_max  = cfg.get("degenerate_theta_max", 100)
```

Convert gamma to per-week for all calculations:
```python
gamma_per_week = gompertz_gamma / 52.0
```

### Step 3: Replace `fit_theta0_global` with new function

Replace the existing theta0 estimation function with:

```python
def fit_theta0_gompertz(h_arr, t_arr, quiet_mask, hve_skip, k_anchor_weeks,
                        gamma_per_week, theta0_init=1.0):
    """
    Estimate theta0 using Gompertz+frailty-depletion model.

    Model:  h_obs(t) = k * exp(gamma*t) / (1 + theta0 * H_gom(t))
    where:  H_gom(t) = (k/gamma) * (exp(gamma*t) - 1)

    k is fixed from mean h(t) in anchor window immediately after HVE skip.
    gamma is a fixed biological constant (0.085/year converted to per-week).
    theta0 is the single free parameter.

    Parameters
    ----------
    h_arr         : np.ndarray, weekly hazard h(t) for this cohort
    t_arr         : np.ndarray, time in weeks from enrollment (0, 1, 2, ...)
    quiet_mask    : np.ndarray bool, True for quiet window timepoints to fit
    hve_skip      : int, DYNAMIC_HVE_SKIP_WEEKS (already applied upstream)
    k_anchor_weeks: int, weeks after hve_skip to use for k estimation
    gamma_per_week: float, Gompertz constant per week (0.085/52)
    theta0_init   : float, initial guess for theta0

    Returns
    -------
    dict with keys:
        theta0_hat   : float, estimated theta0 (0.0 if degenerate)
        k_hat        : float, baseline hazard fixed from anchor
        relRMSE      : float, relative RMSE of fit on quiet points
        n_quiet      : int, number of quiet points used
        converged    : bool
        status       : str, 'ok' | 'degenerate' | 'insufficient_data'
    """
    # --- Fix k from anchor window ---
    anchor_start = hve_skip
    anchor_end   = hve_skip + k_anchor_weeks
    if anchor_end > len(h_arr):
        return {"theta0_hat": 0.0, "k_hat": np.nan, "relRMSE": np.nan,
                "n_quiet": 0, "converged": False, "status": "insufficient_data"}

    k = float(np.mean(h_arr[anchor_start:anchor_end]))
    if k < 1e-12:
        return {"theta0_hat": 0.0, "k_hat": k, "relRMSE": np.nan,
                "n_quiet": 0, "converged": False, "status": "insufficient_data"}

    gamma = gamma_per_week

    # --- Gompertz cumulative hazard at every timepoint ---
    # H_gom(t) = (k/gamma) * (exp(gamma*t) - 1)
    # Use np.expm1 for numerical stability at small gamma*t
    H_gompertz = (k / gamma) * np.expm1(gamma * t_arr)

    # --- Select quiet window points ---
    h_q = h_arr[quiet_mask]
    t_q = t_arr[quiet_mask]
    H_q = H_gompertz[quiet_mask]

    n_quiet = int(quiet_mask.sum())
    if n_quiet < 3:
        return {"theta0_hat": 0.0, "k_hat": k, "relRMSE": np.nan,
                "n_quiet": n_quiet, "converged": False,
                "status": "insufficient_data"}

    # --- Single-parameter fit ---
    def model_h(t, H, theta0):
        numerator   = k * np.exp(gamma * t)
        denominator = 1.0 + theta0 * H
        return numerator / denominator

    def residuals(params):
        theta0 = params[0]
        return h_q - model_h(t_q, H_q, theta0)

    try:
        result = least_squares(
            residuals,
            x0=[theta0_init],
            bounds=([0.0], [np.inf]),
            method='trf',
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
        )
        theta0_hat = float(result.x[0])
        h_pred     = model_h(t_q, H_q, theta0_hat)
        relRMSE    = float(np.sqrt(
            np.mean(((h_q - h_pred) / (h_q + 1e-12))**2)
        ))
        converged  = bool(result.success or result.cost < 1e-20)

        return {
            "theta0_hat": theta0_hat,
            "k_hat":      k,
            "relRMSE":    relRMSE,
            "n_quiet":    n_quiet,
            "converged":  converged,
            "status":     "ok",
        }

    except Exception as e:
        return {"theta0_hat": 0.0, "k_hat": k, "relRMSE": np.nan,
                "n_quiet": n_quiet, "converged": False,
                "status": f"error: {e}"}
```

### Step 4: Update the call site in `process_workbook`

Replace the current theta0 estimation call with:

```python
# Build quiet mask from theta_estimation_windows
quiet_mask = build_quiet_mask(date_arr, theta_estimation_windows)

# Fit theta0 using Gompertz+depletion model
theta0_info = fit_theta0_gompertz(
    h_arr          = h_obs,            # weekly hazard for this cohort
    t_arr          = t_weeks,          # weeks from enrollment
    quiet_mask     = quiet_mask,
    hve_skip       = DYNAMIC_HVE_SKIP_WEEKS,
    k_anchor_weeks = k_anchor_weeks,   # from YAML, default 4
    gamma_per_week = gamma_per_week,   # 0.085/52
)

theta0_hat = theta0_info["theta0_hat"]

# Degenerate fit detection — unchanged
if theta0_hat > degenerate_theta_max:
    log(f"[KCOR7_DEGENERATE] theta0_raw={theta0_hat:.4e} > {degenerate_theta_max}"
        f" -> theta0_applied=0.0")
    theta0_hat = 0.0

# Apply correction — UNCHANGED from current v7
# invert_gamma_frailty is correct regardless of how theta0 was estimated
theta_for_h0 = theta0_hat
H0_full = invert_gamma_frailty(H_obs, theta_for_h0)
```

### Step 5: The H(t) adjustment formula is unchanged

`invert_gamma_frailty` stays exactly as-is:

```python
def invert_gamma_frailty(H_obs, theta):
    if theta < KCOR6_THETA_EPS:
        return H_arr.copy()
    return np.expm1(theta * H_arr) / theta
```

The Gompertz baseline affects only **how theta0 is estimated**, not how it
is applied. Once theta0 is known, the gamma frailty inversion formula:

```
H_adj(t) = (1/θ₀) · ln(1 + θ₀ · H_obs(t))
```

is correct regardless of the underlying baseline hazard model. Do not change
this function.

---

## What to Remove

Delete or disable the following — no longer needed:

- Old `fit_theta0_global` function (flat-k two-parameter version)
- Any references to fitting both k and theta0 simultaneously
- The `theta_min` degenerate check (kept only `theta_max > 100`)

Keep everything else unchanged.

---

## Logging

Add these log lines per cohort fit:

```
KCOR7_FIT: enroll={enroll}, yob={yob}, dose={dose},
           k={k:.4e}, gamma={gamma_per_week*52:.4f}/yr,
           theta0={theta0_hat:.4f}, relRMSE={relRMSE:.4e},
           n_quiet={n_quiet}, status={status}
```

And for degenerate fits (unchanged):
```
KCOR7_DEGENERATE: enroll={enroll}, yob={yob}, dose={dose},
                  theta0_raw={theta0_raw:.4e} -> theta0_applied=0.0
```

---

## Validation Checks After Implementation

Run on `2021_24` enrollment and verify:

1. **k values** match mean h(t) in first 4 weeks post-HVE-skip for each cohort
2. **theta0 monotonicity** for dose=0: should increase from YoB=1920 → 1970
   - Expected approximate values: 0.5, 1.5, 4.8, 10.9, 25.7, 55.3
3. **Zero degenerate fits** above threshold (all caught and logged)
4. **relRMSE < 0.15** for cohorts 1930-1970
5. **Window-mean errors < 10%** for post-booster and post-COVID windows
6. **invert_gamma_frailty output unchanged** — confirm H_adj values are
   identical to current v7 for same theta0 input
7. **Negative control** — synthetic data with known theta should return
   KCOR ratio ≈ 1.0 (no spurious signal)

---

## Summary of Changes

| Component | Change |
|---|---|
| `fit_theta0_gompertz` | NEW — replaces flat-k two-parameter fit |
| `invert_gamma_frailty` | UNCHANGED |
| YAML: `gompertz_gamma` | NEW parameter (default 0.085/year) |
| YAML: `k_anchor_weeks` | NEW parameter (default 4 weeks) |
| YAML: `theta_estimation_windows` | UNCHANGED |
| YAML: `degenerate_theta_max` | UNCHANGED (default 100) |
| Downstream KCOR ratio computation | UNCHANGED |

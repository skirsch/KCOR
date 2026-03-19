# KCOR v7
The original KCORv6 derives a estimated FIXED theta for each cohort from a 2023 quiet window and used the fixed theta for the correction.

In V7, for each cohort, we do a single nonlinear fit to ALL H(t) datapoints in all quiet windows following enrollment in oder to estimate theta(0) which is theta after enrollement. 

We then use the method already in place to adjust H(t). That method already accounts for theta depletion.

* * * *

**What v6 already does correctly:**

The one-step formula `H_adj = (1/θ) - ln(1 + θ - H_obs)` already implicitly accounts for the full θ(t) trajectory. As H\_obs grows over time, the correction automatically increases --- capturing the cumulative effect of frailty depletion including through COVID waves. The **functional form** is correct.

* * * *

**What v6 gets wrong:**

The theta value fed into that formula is estimated from a **single 2023 quiet window** using a constant-theta assumption. As we established:

-   The 2023 window gives a theta that reflects the **depleted late-period population**
-   This underestimates the true θ₀ at enrollment
-   The correction is therefore too small throughout --- under-corrected everywhere
-   The bias direction makes the vaccine appear **worse** than it truly is

* * * *

**What v7 fixes:**

Replaces the single-window late theta with the true θ₀ estimated from the global regression across all quiet windows. This gives a larger, correctly anchored theta value --- and when fed into the same `invert_gamma_frailty` formula, produces the correct adjustment.

* * * *

**So in one sentence:**

V6 has the right machinery but the wrong input. V7 just gives that machinery the right input --- θ₀ at enrollment rather than θ at late depletion. The formula itself does not change at all.

Yes — identical method, identical code, just applied to the vaccinated cohort's h(t) and H(t).

---

**The only difference is the inputs:**

```python
# Unvaccinated
theta0_unvaxxed = fit_theta0_global(h_unvaxxed, H_unvaxxed, quiet_mask)

# Vaccinated — exactly the same function
theta0_vaxxed = fit_theta0_global(h_vaxxed, H_vaxxed, quiet_mask)
```

Same quiet windows, same formula, same one-step correction applied independently to each cohort.

---

**And this resolves the earlier question about apply_to:**

Previously we debated whether to apply the correction to unvaccinated only or both cohorts, and near-zero vaccinated theta values made it unclear. With the global regression approach:

- If the vaccinated cohort has real frailty depletion signal → fits a meaningful theta0 → applies a real correction
- If the vaccinated cohort has near-zero frailty signal → optimizer returns theta0 ≈ 0 → correction is negligible → effectively no correction applied

**The method self-regulates.** You do not need to special-case the vaccinated arm or hardcode `apply_to: unvaccinated_only`. The data tells you whether a correction is warranted for each cohort.

---

**This is actually important for KCOR's validity:**

Applying correction only to the unvaccinated arm while leaving the vaccinated arm uncorrected is asymmetric and potentially introduces bias. 

Applying the same principled correction to both arms is cleaner and more defensible in the paper — the correction is applied uniformly and each cohort's own frailty structure determines the magnitude.

## implement this starting from KCORv6.py. 

Copy KCORv6.py to KCOR.py as the starting point (discarding the changes we made).

Yes --- start fresh from v6. The v7 changes you need are minimal and surgical. All the complexity that got added (in the current KCOR.py) can be discarded entirely.

* * * *

**The complete diff from v6 to v7 is just two changes:**

**Change 1 --- Replace the theta estimation:**

In v6 you estimate a single constant theta from a 2023 quiet window using `fit_k_theta_cumhaz`. Replace that with the global regression across all quiet windows:

```
# V6 (replace this)
theta = estimate_theta_from_quiet_window(H_obs, quiet_window_2023)

# V7 (replace with this)
theta0 = fit_theta0_global(h_obs, H_obs, quiet_mask)
# where: h_obs = k / (1 + theta0 * H)
# fitted to all quiet timepoints simultaneously

```

**Change 2 --- Nothing.** `invert_gamma_frailty` already does the correct one-step adjustment and is unchanged.

* * * *

**The only new code needed is one function:**

```
def fit_theta0_global(h_arr, H_cum, quiet_mask, theta0_init=1.0):
    h_q = h_arr[quiet_mask]
    H_q = H_cum[quiet_mask]
    k0  = float(np.mean(h_q))

    def residuals(params):
        k, theta0 = params
        return h_q - k / (1.0 + theta0 * H_q)

    result = least_squares(
        residuals,
        x0=[k0, theta0_init],
        bounds=([1e-12, 0.0], [np.inf, np.inf]),
        method='trf', ftol=1e-14, xtol=1e-14, gtol=1e-14,
    )
    return float(result.x[1])  # theta0_hat

```

With bounds `k >= 1e-12` and `theta0 >= 0`, `theta0 = 0` is an explicit valid boundary solution.
At `theta0 = 0`, the model reduces to `h_q = k` (flat hazard with no frailty correction), which is the correct degenerate case for cohorts with no detectable frailty curvature. No special-casing is needed; the optimizer should be allowed to land there naturally when supported by data.

That is literally the entire v7 upgrade --- one ~15 line function replacing the old theta estimation, everything else identical to v6.

* * * *

**YAML change is also minimal:**

```
# V6
KCOR6_QUIET_WINDOW: 2023-01..2023-52

# V7 --- just add multiple windows, remove everything else. This is already added to Czech.yaml
theta_estimation_windows:
    - ['2023-22', '2023-37', 'post COVID']
    - ['2022-22', '2022-25', 'post booster']
    - ['2021-26', '2021-36', 'post primary']

```

* * * *

**Start from clean v6, make these two targeted changes, and you have a correct v7.** 

Make sure the README reflects this V7 change (replace the V7 description that is there)

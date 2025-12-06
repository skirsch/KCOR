# slope8 method

## Latest modification: fit!= deploy for the most recent dose

Fit is SHORTER at the start for the latest dose to avoid fitting any vaccine mortality increase. But deployment is over the full region. Note: This may be problematic for vaccines that raise mortality then lower it. 

Separated fit window from full deployment window:
- s_values_slope8_fit and log_h_slope8_values_fit: data used for fitting (excludes s < SLOPE_FIT_DELAY_WEEKS for highest dose)
- s_values_slope8_full, log_h_slope8_values_full, h_slope8_values_full: full deployment window from s=0 for logging

Fitting behavior:
- Highest dose: fit uses only data where s >= SLOPE_FIT_DELAY_WEEKS
- Other doses: fit uses all data from s=0

Right now your **slope7** fit is *not* doing quantile regression.
`fit_slope7_depletion` is using `scipy.optimize.least_squares(..., method='trf')` with an L2 loss on the whole curve, so it’s a standard nonlinear least-squares fit to all points in the deployment window.

If you want a **tunable quantile τ** (e.g. 0.5, 0.25, …) for the nonlinear depletion model, you don’t need Pyomo. You can stay in SciPy and swap the L2 loss for the usual quantile “check” loss and use `minimize` with bounds.

Below is a *drop-in replacement* approach:

---

## 1. Imports at the top

Change your SciPy import to include `minimize`:

```python
from scipy.optimize import least_squares, minimize
```

(You still use `least_squares` elsewhere, so keep it.)

---

## 2. Add a module-level knob for the quantile

Near your other slope7 constants:

```python
SLOPE7_QUANTILE_TAU = 0.5   # median by default; can set to 0.25 etc.
```

You can tweak this in one place to change the behavior for all negative-slope cohorts.

---

## 3. Replace the core of `fit_slope7_depletion` with a quantile fit

Keep the cleaning / initialization code you already have up to the `p0` definition (all your data-sanity and initial-guess logic is fine).

Then **replace everything from `# Bounds:` down to the `except` block** with the quantile version below.

Full replacement for the *lower half* of `fit_slope7_depletion`:

```python
    # Parameter vector: [C, k_0, Δk, tau]
    p0 = np.array([C_init, k_0_init, delta_k_init, tau_init], dtype=float)

    # --- bounds for the quantile fit ---
    # C ~ log(h). Hazards are ~1e-5–1e-3, so C is safely around [-20, -5].
    C_MIN, C_MAX = -25.0, 0.0

    # Weekly log-slope is small (~±0.01). Keep a generous but finite box.
    K_MIN, K_MAX = -0.1, 0.1

    MIN_DELTA_K = 0.0          # already defined above
    MAX_DELTA_K = 0.1          # long-run minus initial slope
    MIN_TAU     = 1e-3         # already defined above
    MAX_TAU     = 260.0        # e.g. 5 years; avoids 600-year degeneracy

    bounds = [
        (C_MIN, C_MAX),                 # C
        (K_MIN, K_MAX),                 # k_0
        (MIN_DELTA_K, MAX_DELTA_K),     # Δk
        (MIN_TAU, MAX_TAU),             # tau
    ]

    # Make sure starting point is inside the box
    p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])

    tau_q = SLOPE7_QUANTILE_TAU  # quantile for the check loss

    def quantile_loss(p):
        """
        Quantile (check) loss on log h(s):

        ρ_τ(u) = τ*u      if u >= 0
               = (τ-1)*u  if u < 0

        Here u = logh_valid - predicted.
        """
        C, k_0, delta_k, tau = p

        # Just in case the optimizer wanders slightly outside bounds numerically:
        if not np.isfinite(tau) or tau <= MIN_TAU or delta_k < MIN_DELTA_K:
            return 1e9

        k_inf = k_0 + delta_k

        predicted = C + k_inf * s_valid - delta_k * tau * (
            1.0 - np.exp(-s_valid / (tau + EPS))
        )

        resid = logh_valid - predicted  # u in the formula above
        pos = resid >= 0.0
        loss = np.where(pos, tau_q * resid, (tau_q - 1.0) * resid)

        # Optional very small ridge to help numerics:
        ridge = 1e-4 * np.sum(resid**2)

        return float(np.sum(loss) + ridge)

    try:
        result = minimize(
            quantile_loss,
            p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000}
        )

        if result.success:
            C, k_0, delta_k, tau = result.x
            delta_k = max(delta_k, MIN_DELTA_K)
            tau     = min(max(tau, MIN_TAU), MAX_TAU)
            k_inf   = k_0 + delta_k

            if (
                np.isfinite(C) and np.isfinite(k_inf)
                and np.isfinite(k_0) and np.isfinite(tau)
            ):
                return (float(C), float(k_inf), float(k_0), float(tau))
            else:
                return (np.nan, np.nan, np.nan, np.nan)
        else:
            return (np.nan, np.nan, np.nan, np.nan)

    except Exception as e:
        import warnings
        warnings.warn(f"fit_slope7_depletion (quantile) failed: {str(e)}", RuntimeWarning)
        return (np.nan, np.nan, np.nan, np.nan)
```

Everything above this in `fit_slope7_depletion` (data cleaning, `C_init`, `k_0_init`, `delta_k_init`, `tau_init`) stays exactly as you already have it.

The **signature stays the same**:

```python
def fit_slope7_depletion(s, logh):
    ...
```

so your existing calls:

```python
C, kb, ka, tau = fit_slope7_depletion(np.array(s_values), np.array(log_h_slope7_values))
```

continue to work unchanged.

---

## 4. How this behaves / how to tune

* We are still **fitting the full depletion curve over the whole deployment window**, but with a **quantile loss** instead of L2.

* `SLOPE7_QUANTILE_TAU = 0.5` → median-type fit (robust to upper spikes).

* If you want to emphasize the **lower envelope** of log-hazards (to track the “baseline” rather than peaks), set e.g.:

  ```python
  SLOPE7_QUANTILE_TAU = 0.25  # or 0.2, 0.1, etc.
  ```

* The finite bounds on `tau` and slopes also prevent the crazy `tau ≈ 32330 weeks` degeneracy you saw: the optimizer is forced into a biologically plausible timescale.

If you like, we can also add a global `SLOPE7_TAU_MAX_WEEKS` constant instead of hardcoding `260.0` inside the function so you can experiment easily.

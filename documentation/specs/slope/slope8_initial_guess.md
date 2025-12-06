# better initial guesses

Nice, this is exactly the right place to tweak things — bad inits are *the* reason these nonlinear fits blow up or go “invalid.”

Right now your init logic is way too local:

* C_init: mean of first 3 points
* k_0_init: slope of first 2 points
* later slope: last vs 3rd-to-last
* tau_init: ⅓ of span

That’s very fragile if there’s noise or a blip at either end.

A much more stable approach is:

1. **Use small OLS fits over early and late windows**, not single finite differences.
2. Derive (k_0), (k_\infty), and τ from those, not from just 2–3 points.
3. Use a global linear fit as a fallback / sanity check.

Below is a concrete replacement strategy you can drop into `fit_slope8_depletion` (or a helper).

---

## 1. Depletion model reminder (your parameterization)

You’re using:

* parameters: (C, k_0, \Delta k, \tau)
* with (k_\infty = k_0 + \Delta k)
* and:

[
\log h(s) = C + k_\infty s - \Delta k,\tau \big(1 - e^{-s/\tau}\big)
]

Slope in log-space:

[
\frac{d}{ds}\log h(s) = k_\infty - \Delta k, e^{-s/\tau}
]

* At (s=0): slope = (k_0 = k_\infty - \Delta k)
* As (s \to \infty): slope → (k_\infty)

So we want:

* (k_0): **early slope**
* (k_\infty): **late slope**
* τ: timescale over which slope transitions from (k_0) to (k_\infty)

---

## 2. Better initial guesses

### Step 0: sort and basic checks

```python
s_valid = np.asarray(s_valid, float)
logh_valid = np.asarray(logh_valid, float)

order = np.argsort(s_valid)
s_valid = s_valid[order]
logh_valid = logh_valid[order]

n = len(s_valid)
if n < 5:
    # fall back: very small window; just do linear
    # we'll still produce a depletion init but extremely mild
    ...
```

### Step 1: global linear fit (baseline)

Use all points to get a baseline slope:

```python
# global linear fit in log-space
A = np.vstack([s_valid, np.ones_like(s_valid)]).T
b_global, a_global = np.linalg.lstsq(A, logh_valid, rcond=None)[0]
# logh ≈ a_global + b_global * s
```

We’ll use this as a fallback and as a sanity check.

### Step 2: early and late local slopes

Use **small windows** instead of single differences:

```python
# choose window size: ~1/3 of data but at least 3 points
w = max(n // 3, 3)

# Early window: first w points
s_early = s_valid[:w]
h_early = logh_valid[:w]
b0, a0 = np.linalg.lstsq(
    np.vstack([s_early, np.ones_like(s_early)]).T,
    h_early,
    rcond=None
)[0]

# Late window: last w points
s_late = s_valid[-w:]
h_late = logh_valid[-w:]
b_inf, a_inf = np.linalg.lstsq(
    np.vstack([s_late, np.ones_like(s_late)]).T,
    h_late,
    rcond=None
)[0]
```

Now:

* `b0`  ≈ initial slope (k_0)
* `b_inf` ≈ long-run slope (k_\infty)

### Step 3: k_0_init, delta_k_init

You expect depletion-like shape: **slope becomes less negative over time**, so:

[
k_0 \le k_\infty \quad (\text{more negative early, less negative late})
]

So:

```python
k_0_init  = float(b0)
k_inf_init = float(b_inf)

# If the pattern is "wrong" (slope more negative late than early),
# then either the cohort isn't depletion-like, or noise reversed it.
# In that case, collapse toward global slope and small Δk.
if k_inf_init < k_0_init:
    # Use global slope as both and tiny curvature
    k_0_init  = float(b_global)
    k_inf_init = float(b_global)

delta_k_init = max(k_inf_init - k_0_init, 1e-4)  # small but > 0
```

So in a clean depletion case (steep early, flatter later):

* `k_0_init` ≈ early slope (more negative)
* `k_inf_init` ≈ late slope (less negative)
* `delta_k_init` positive and reasonable

In non-depletion cases, this collapses to “almost linear” with tiny Δk.

### Step 4: τ_init based on data span (but not insane)

Your current:

```python
tau_init = (s_max - s_min) / 3.0
```

is not terrible, but can be too big. I’d:

* tie it to data span,
* keep it in a **moderate band** (say 2–52 weeks).

```python
span = max(s_valid) - min(s_valid)
if span <= 0:
    span = 10.0  # arbitrary small default

tau_init = span / 3.0
tau_init = min(max(tau_init, 2.0), 52.0)  # between 2 weeks and 1 year
```

That avoids the 30,000+ week nonsense and gives the solver something realistic.

### Step 5: C_init from early intercept (not ad-hoc mean)

Instead of averaging the first 3 log-hazard values:

```python
# Use early line's intercept evaluated at s=0
C_init = float(a0)  # logh when s=0 from early-line fit
```

This makes C consistent with the early slope, which is more stable than an arbitrary mean of first 3 points (especially if the first point is noisy).

---

## 3. Putting it together (summary)

A much better init block for slope8 would be:

```python
def make_slope8_initial_guess(s_valid, logh_valid):
    s_valid = np.asarray(s_valid, float)
    logh_valid = np.asarray(logh_valid, float)

    order = np.argsort(s_valid)
    s_valid = s_valid[order]
    logh_valid = logh_valid[order]

    n = len(s_valid)

    # Global slope as fallback
    A = np.vstack([s_valid, np.ones_like(s_valid)]).T
    b_global, a_global = np.linalg.lstsq(A, logh_valid, rcond=None)[0]

    if n < 5:
        # Too few points: almost linear
        C_init = float(a_global)
        k_0_init = float(b_global)
        delta_k_init = 1e-4
        tau_init = 10.0
        return C_init, k_0_init, delta_k_init, tau_init

    # Window size for early/late fits
    w = max(n // 3, 3)

    # Early window
    s_early = s_valid[:w]
    h_early = logh_valid[:w]
    b0, a0 = np.linalg.lstsq(
        np.vstack([s_early, np.ones_like(s_early)]).T,
        h_early,
        rcond=None
    )[0]

    # Late window
    s_late = s_valid[-w:]
    h_late = logh_valid[-w:]
    b_inf, a_inf = np.linalg.lstsq(
        np.vstack([s_late, np.ones_like(s_late)]).T,
        h_late,
        rcond=None
    )[0]

    k_0_init  = float(b0)
    k_inf_init = float(b_inf)

    # Enforce depletion pattern; otherwise fall back toward linear
    if k_inf_init < k_0_init:
        k_0_init  = float(b_global)
        k_inf_init = float(b_global)

    delta_k_init = max(k_inf_init - k_0_init, 1e-4)

    # Tau: tied to span, kept reasonable
    span = max(s_valid) - min(s_valid)
    if span <= 0:
        span = 10.0
    tau_init = span / 3.0
    tau_init = min(max(tau_init, 2.0), 52.0)

    # C from early intercept
    C_init = float(a0)

    return C_init, k_0_init, delta_k_init, tau_init
```

Then in `fit_slope8_depletion` you just call this instead of your current ad-hoc block and pass `C_init, k_0_init, delta_k_init, tau_init` into your optimizer.

This will:

* dramatically reduce invalid fits,
* avoid insane τ values,
* respect the expected depletion pattern when it’s present,
* gracefully degrade to “almost linear” when it’s not.

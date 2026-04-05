---
name: NPH post-inversion refactor
overview: Replace the current pre-inversion raw-hazard NPH correction (`apply_covid_correction_in_place`) with a new post-inversion correction that operates on frailty-neutral cumulative hazard H0(t), using the cohort's fitted Gompertz baseline path as the wave-excess reference.
todos:
  - id: deprecate-old
    content: Add early return to apply_covid_correction_in_place to disable pre-inversion hazard_raw mutation
    status: completed
  - id: new-helper
    content: Implement apply_nph_correction_post_inversion with Gompertz H_ref path, F_d correction, monotonicity repair, identity check, and diagnostics
    status: completed
  - id: wire-main-loop
    content: Insert post-inversion correction call after H0 = invert_gamma_frailty() in the KCOR6 normalization loop (~line 7797) in process_workbook
    status: completed
  - id: wire-build-kcor-rows
    content: Skip NPH correction for YoB=-2 blocks inside build_kcor_rows (no valid aggregate k_hat); leave H0 unchanged there
    status: completed
  - id: hoist-gcf
    content: Ensure get_gompertz_theta_fit_config() and covid_cfg are fetched once before the normalization loop and passed through
    status: completed
isProject: false
---

# NPH post-inversion correction refactor — `KCOR.py`

## Current pipeline (to be changed)

```
MR → hazard_raw
  → apply_covid_correction_in_place()   ← mutates hazard_raw IN PLACE (pre-inversion)
  → H_obs = cumsum(h_eff_obs)
  → H0 = invert_gamma_frailty(H_obs, theta)
  → h0_inc = diff(H0)
  → hazard_adj = h0_inc
  → CH = cumsum(hazard_eff)
```

## New pipeline

```
MR → hazard_raw
  → H_obs = cumsum(h_eff_obs)
  → H0 = invert_gamma_frailty(H_obs, theta)
  → H_corr = apply_nph_correction_post_inversion(H0, ...)   ← NEW, post-inversion
  → h0_inc = diff(H_corr)
  → hazard_adj = h0_inc
  → CH = cumsum(hazard_eff)
```

## File to change

[`code/KCOR.py`](code/KCOR.py) — all changes are localized here.

---

## Change 1 — Deprecate `apply_covid_correction_in_place` call sites

Two call sites exist:

- Line **7461–7467**: main per-sheet `df` (individual YoB cohorts)
- Line **7501–7507**: `all_ages_agg` (YoB = −2)

**Action:** Convert both calls to no-ops by adding an early return at the top of `apply_covid_correction_in_place` (keep the function body intact for reference, just short-circuit it):

```python
def apply_covid_correction_in_place(df, covid_cfg, kcor6_params_map=None, sheet_name=None):
    # DEPRECATED: pre-inversion raw-hazard correction replaced by
    # apply_nph_correction_post_inversion() called after invert_gamma_frailty().
    return
    ...
```

This is the safest change: zero risk of breaking other callers, body preserved for audit.

---

## Change 2 — New helper `apply_nph_correction_post_inversion`

Add a new top-level function near `apply_covid_correction_in_place` (~line 2447). Signature:

```python
def apply_nph_correction_post_inversion(
    H0,           # frailty-neutral cumulative hazard, shape (n,)
    t_vals,       # cohort time index (same convention as KCOR pipeline)
    dates,        # DateDied array aligned with H0
    theta_applied,# cohort theta used in invert_gamma_frailty
    alpha,        # NPH alpha
    start_dt,     # wave window start (datetime)
    end_dt,       # wave window end (datetime)
    k_hat,        # Gompertz k from kcor6_params_map["k_hat"]
    gamma_per_week,  # from get_gompertz_theta_fit_config()["gamma_per_week"]
    t_rebased=None,  # t - DYNAMIC_HVE_SKIP_WEEKS; if None, derived from t_vals
):
    -> (H_corr, diagnostics_dict)
```

### Steps inside the helper

1. **Identity shortcut:** if `abs(alpha - 1.0) <= NPH_IDENTITY_TOL`, return `H0.copy()` + diagnostics with `max_abs_change=0`.

2. **Wave mask:**
   ```python
   wave_mask = (dates >= start_dt) & (dates <= end_dt)
   ```

3. **F_d values:**
   ```python
   theta_t = propagate_theta(theta_applied, H0)
   f_vals = gamma_moment_alpha(theta_t, alpha)
   ```

4. **Gompertz baseline path (H_ref):**
   - `t_reb = t_vals - DYNAMIC_HVE_SKIP_WEEKS` if `t_rebased` is None
   - `h_base = k_hat * safe_exp(gamma_per_week * t_reb)`  (per-week Gompertz hazard)
   - Find first wave index `i0 = first index where wave_mask is True`
   - `H_ref[i0] = H0[i0]`
   - For `i > i0` in wave: `H_ref[i] = H_ref[i-1] + h_base[i]`
   - Outside wave: `H_ref` is not used (H_corr = H0 there)

5. **Correction — positive excess only:**
   ```python
   H_corr = H0.copy()
   delta = H0 - H_ref
   # Only correct points with strictly positive excess above the Gompertz baseline.
   # If delta <= 0 (H0 at or below the baseline path), leave H_corr = H0 unchanged.
   positive_excess = delta > EPS
   valid = wave_mask & positive_excess & np.isfinite(f_vals) & (f_vals > EPS) & np.isfinite(H_ref)
   H_corr[valid] = H_ref[valid] + delta[valid] / f_vals[valid]
   ```
   Do **not** clip negative `delta` to zero before applying the correction — instead, simply skip those points (`H_corr = H0` there). Log the count of skipped non-positive-excess points in diagnostics.

6. **Monotonicity repair:**
   ```python
   H_corr = np.maximum.accumulate(H_corr)
   ```
   Log if any repair was needed.

7. **Diagnostics dict:** `max_abs_change`, `wave_points`, `corrected_points`, `skipped_nonpositive_excess`, `f_min`, `f_max`, `any_nonmonotone_fixed`, `correction_mode`.

8. **Guard rails:**
   - If `k_hat` is not finite or `gamma_per_week` is not finite: return `H0.copy()` with `correction_mode="skipped_missing_params"`.
   - Only correct points where `f_vals > EPS`, finite, and `delta > EPS`.
   - Do **not** clip or modify `delta` before the positivity check — negative delta is a valid signal that H0 is below the Gompertz baseline (e.g. pre-wave or low-mortality week) and should be left uncorrected.

---

## Change 3 — Apply correction in the KCOR6 normalization loop

**Location:** `process_workbook`, lines **7761–7818** — the `for key, g in df.groupby(groupby_norm)` loop.

Currently:
```python
H0 = invert_gamma_frailty(H_obs, theta)
h0_inc = np.diff(H0, prepend=0.0)
```

**After the change:**
```python
H0 = invert_gamma_frailty(H_obs, theta)

# NPH correction is applied after gamma-frailty inversion in cumulative-hazard space,
# relative to the fitted no-wave Gompertz baseline path.
if covid_cfg is not None and bool(covid_cfg.get("apply_correction", False)):
    alpha_nph = float(covid_cfg["apply_alpha"])
    nph_k_hat = float(params.get("k_hat", np.nan)) if isinstance(params, dict) else np.nan
    nph_gamma = _gcf["gamma_per_week"]   # fetched once before the loop
    t_reb = t_vals - float(DYNAMIC_HVE_SKIP_WEEKS)
    H0, _nph_diag = apply_nph_correction_post_inversion(
        H0,
        t_vals,
        g_sorted["DateDied"].to_numpy(),
        theta,
        alpha_nph,
        covid_cfg["start_dt"],
        covid_cfg["end_dt"],
        nph_k_hat,
        nph_gamma,
        t_rebased=t_reb,
    )
    if _nph_diag.get("correction_mode") not in ("identity", "skipped_missing_params"):
        dual_print(
            f"[NPH_POST_INV] enroll={sh} yob={yob} dose={dose} "
            f"alpha={alpha_nph:.4f} k_hat={nph_k_hat:.6g} "
            f"f_min={_nph_diag.get('f_min', float('nan')):.4f} "
            f"f_max={_nph_diag.get('f_max', float('nan')):.4f} "
            f"max_abs_H_change={_nph_diag.get('max_abs_change', float('nan')):.6g} "
            f"wave_points={_nph_diag.get('wave_points', 0)}"
        )

h0_inc = np.diff(H0, prepend=0.0)
```

**Fetch `_gcf` and `covid_cfg` once before the loop** (they are already available in `process_workbook` scope; `covid_cfg` is fetched at line 7460, `_gcf = get_gompertz_theta_fit_config()` is already called at ~6692 area — verify and hoist if needed).

---

## Change 4 — Identity check

Inside `apply_nph_correction_post_inversion`, when `correction_mode == "forced_alpha"` and `abs(alpha - 1.0) <= NPH_IDENTITY_TOL`:

```python
max_delta = float(np.max(np.abs(H_corr - H0)))
if max_delta > NPH_IDENTITY_TOL:
    raise RuntimeError(
        f"NPH post-inversion identity check failed: max H_corr delta {max_delta:.3e}"
    )
```

---

## Change 5 — `build_kcor_rows` all-ages path (lines 4867–4954)

There are two additional `H0 → h0_inc` blocks inside `build_kcor_rows` (for the YoB = −2 aggregated cohort). **Do not apply the post-inversion NPH correction there.** The YoB = −2 aggregate does not have a per-cohort Gompertz `k_hat` derived from individual quiet-window fits; applying the correction without a valid baseline path would be incorrect. Leave `H0` unchanged in those blocks.

Add a comment at each block:
```python
# NPH post-inversion correction is not applied for the YoB=-2 all-ages aggregate:
# no valid per-cohort Gompertz k_hat is available for this derived cohort.
```

---

## Acceptance checks (inline)

- `alpha=1.0` → `H_corr == H0` exactly (identity enforced).
- `apply_covid_correction_in_place` no longer mutates `hazard_raw` (early return).
- Correction uses `k_hat * exp(gamma_per_week * t_rebased)` as baseline, not wave-start anchor alone.
- Only `delta = H0 - H_ref` where `delta > EPS` is divided by `F_d`; non-positive excess points are left as `H_corr = H0`.
- YoB = −2 all-ages aggregate blocks are left uncorrected (no valid `k_hat`).
- `CH = cumsum(hazard_eff)` downstream is unchanged — it reads from `hazard_adj` which comes from the corrected `h0_inc`.

---

## Key lines to touch

| Location | Line(s) | Action |
|---|---|---|
| `apply_covid_correction_in_place` | ~2364 | Add `return` at top (deprecate) |
| New function | after ~2446 | Add `apply_nph_correction_post_inversion` |
| KCOR6 norm loop | ~7797 | Insert post-inversion correction after `H0 = invert_gamma_frailty(...)` |
| `build_kcor_rows` all-ages loops | ~4896, ~4946 | Add comment only — no correction applied (YoB = −2, no valid k_hat) |
| `_gcf` hoist | ~7750 | Ensure `get_gompertz_theta_fit_config()` is called once before the norm loop |

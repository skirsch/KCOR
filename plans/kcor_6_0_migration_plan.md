# KCOR 6.0 (gamma-frailty) implementation plan

## Goal
Migrate the main KCOR pipeline in `code/KCOR.py` from hazard-level slope normalization (v5.4) to **KCOR 6.0 gamma-frailty normalization**, per `documentation/specs/KCORv6/kcor_6_0_spec.md`.

## Confirmed choices
- **Fit method**: Option B — nonlinear least squares in cumulative-hazard space (**no likelihood optimization / no MLE**).
- **Observed hazard for fitting**: `hazard_obs = hazard_from_mr_improved(MR)` (per user choice 1a).
- **Model time axis**: `t_fit = t` (weeks since enrollment) (per user choice 2b).
  - Pre-skip weeks contribute **zero** to `H_obs` via `hazard_eff = 0` for `t < DYNAMIC_HVE_SKIP_WEEKS`.
- **Fit granularity**: per `(EnrollmentDate sheet, YearOfBirth-group, Dose)`, including `YearOfBirth=-2` (All Ages).
- **Logging**: one diagnostics line per `(EnrollmentDate, YearOfBirth-group, Dose)` in `KCOR_summary.log` (CSV-like tag line).

## Scope / non-goals (v6.0 cutover)
- **Scope**
  - Primary fixed-cohort KCOR run in `code/KCOR.py` (non-SA mode).
  - Preserve output schema (keep `MR_adj`, `hazard_raw`, `hazard_adj`, `CH`, etc.).
- **Non-goals**
  - No MLE / likelihood optimization.
  - No alternative baseline shapes (stay with $g(t)=1$).
  - Keep SA mode slope-based initially (avoid breaking existing SA grid sweep + fixtures).

## Key behavioral change
Replace the current `hazard_adj → CH` (slope6/slope8) path with KCOR 6.0 cumulative-hazard normalization:

- Compute observed cumulative hazard $H^{obs}$ (starting at accumulation start after `DYNAMIC_HVE_SKIP_WEEKS`).
- Fit $(k,\theta)$ on quiet-window points (ISO `2022-24`..`2024-16`).
- Normalize by inversion to obtain $H_0$:

$$
H_0(t)=\frac{e^{\theta H^{obs}(t)}-1}{\theta}.
$$

- Use `CH` in outputs as the **normalized cumulative hazard** $H_0$, so KCOR uses `CH_num/CH_den` on normalized cumulative hazards.

## Code touchpoints (current v5.4 code structure)
- `code/KCOR.py`
  - `VERSION` (line ~80): update to `v6.0`.
  - `compute_slope6_normalization(...)` (line ~1626): current slope8/linear normalization fit. KCOR 6.0 adds a parallel fit for $(k,\theta)$.
  - `process_workbook(...)` (line ~3367): current hazard/CH computation; v6.0 normalization should be applied here (outside SA mode).
  - `build_kcor_rows(...)` (line ~2392): consumes `CH`, `hazard_raw`, `hazard_adj` (and uses `hazard_adj/hazard_raw` in CI scaling). Docstring assumptions must be updated for v6.0.
  - All-ages aggregation block in `build_kcor_rows` (line ~2782): must use fitted `theta` for `YearOfBirth=-2`.

## Implementation plan (phases + acceptance criteria)

### Phase A — Add KCOR 6.0 helpers in `code/KCOR.py`
- [ ] **Quiet window constants**
  - `KCOR6_QUIET_START_ISO = "2022-24"`
  - `KCOR6_QUIET_END_ISO   = "2024-16"`
- [ ] **ISO helpers**
  - `iso_to_int(year, week) -> year*100 + week`
  - `in_quiet_window(iso_year, iso_week) -> bool`
- [ ] **Cumhaz model + fit**
  - `H_model(t, k, theta)` implementing $H^{model}(t)=\frac{1}{\theta}\log(1+\theta k t)$ with $\theta \to 0$ limit $k t$
  - `fit_k_theta_cumhaz(t, H_obs)` using `scipy.optimize.least_squares` with bounds `k>0`, `theta>=0`
- [ ] **Normalization**
  - `invert_gamma_frailty(H_obs, theta)` using `np.expm1(theta*H_obs)/theta` with $\theta \to 0$ limit

**Acceptance (Phase A)**
- Numerical limit cases behave correctly (small-$\theta$ stability via `log1p`/`expm1`).

### Phase B — Fit $(k,\theta)$ per cohort (YoB × Dose) in the quiet window + log
In the main sheet loop in `process_workbook()`:
- [ ] Build per-cohort weekly series **and** an All Ages cohort (`YearOfBirth=-2`) aggregated across YoB groups.
- [ ] Compute per cohort:
  - `hazard_obs = hazard_from_mr_improved(MR)`
  - `hazard_eff = hazard_obs` if `t >= DYNAMIC_HVE_SKIP_WEEKS` else `0.0`
  - `H_obs = cumsum(hazard_eff)` within each cohort
  - `t_fit = t` (weeks since enrollment)
- [ ] Select fit points where:
  - calendar ISO week is inside `[2022-24, 2024-16]`, and
  - `t_fit >= DYNAMIC_HVE_SKIP_WEEKS`
- [ ] Fit $(k,\theta)$ per `(EnrollmentDate, YoB-group, Dose)` including `YoB=-2`.
- [ ] Log one line per cohort to `KCOR_summary.log`:

```text
KCOR6_FIT,EnrollmentDate=...,YoB=...,Dose=...,k_hat=...,theta_hat=...,RMSE_Hobs=...,n_obs=...,success=...,note=...
```

**Acceptance (Phase B)**
- Exactly one `KCOR6_FIT` line per cohort per sheet (including `YoB=-2`), with finite `k_hat`, `theta_hat`, and `RMSE_Hobs`.

### Phase C — Apply normalization to all outputs (age-stratified + all-ages)
Using fitted `theta_by_cohort[(YearOfBirth, Dose)]`:
- [ ] For each cohort:
  - compute `H0 = invert_gamma_frailty(H_obs, theta)`
  - set `df["CH"] = H0` (core swap)
  - set `df["hazard_adj"]` as the per-step increment of `CH` (groupwise `diff()`), so downstream CI scaling can continue to use `hazard_adj/hazard_raw`
  - keep `MR_adj = MR` for schema compatibility

**Acceptance (Phase C)**
- `CH` is non-decreasing and equals 0 for `t < DYNAMIC_HVE_SKIP_WEEKS`.
- `hazard_adj` is non-negative and groupwise sums to `CH`.

### Phase D — All Ages (`YearOfBirth=-2`) output path
- [ ] Update the All Ages aggregation path in `build_kcor_rows()` to use the fitted `theta_by_cohort[(-2, dose)]` rather than slope8 normalization.

**Acceptance (Phase D)**
- All-ages `CH` is computed from the aggregated `H_obs` series via gamma-frailty inversion (not slope-based scaling).

### Phase E — Versioning + compatibility strategy
- [ ] Update `VERSION = "v6.0"`.
- [ ] Keep SA mode (`SENSITIVITY_ANALYSIS=1`) on legacy slope normalization initially (avoid breaking `test/sensitivity` fixtures).
- [ ] Recommended toggle for safe rollout: only enable v6.0 normalization when `KCOR6_ENABLE=1` (default off until validated), then flip the default once validated.

### Phase F — Validation & tests
- [ ] Verify `KCOR_summary.log` contains `KCOR6_FIT` lines (one per cohort, including `YoB=-2`).
- [ ] Verify output `CH` columns reflect normalized cumulative hazards and KCOR curves are sensible.
- [ ] Run tests:
  - `make test` (negative-control + sensitivity)
  - optionally `make slope-test` (should remain valid if slope/SA paths are unchanged)

## Risks / edge cases
- **Quiet-window sparsity**: if `n_obs` is too small, log `success=false` and fall back to `theta=0` (identity) or skip normalization for that cohort (decision to be made during implementation).
- **Small-$\theta$ numerics**: require `log1p`/`expm1` and explicit $\theta \to 0$ limits.
- **Noisy cohorts**: cumulative-hazard fitting is robust but still needs guardrails (bounds, clipping, and clear logging of failure reasons).



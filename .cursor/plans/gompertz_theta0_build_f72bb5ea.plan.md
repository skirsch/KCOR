---
name: Gompertz theta0 build
overview: Implement the KCOR v7 Gompertz+depletion θ₀ estimator from `documentation/specs/KCORv7/KCOR_v7_gompertz_instructions.md` in `code/KCOR.py`, add YAML knobs under existing `time_varying_theta`, and keep **top-level** `theta_estimation_windows` as the current list-of-triples format (no schema change there). `invert_gamma_frailty` and downstream KCOR math stay unchanged.
todos:
  - id: yaml-loader
    content: Add gompertz_gamma + k_anchor_weeks to time_varying_theta parsing, cache, getter, and process_workbook config print
    status: completed
  - id: fit-func
    content: fit_theta0_gompertz with t_rebased (H_gom(0)=0 at first accumulating week); no hve_skip param; anchor_mask [0,k_anchor); anchor_only status; remove fit_theta0_global
    status: completed
  - id: call-sites
    content: At each call site t_rebased = t_vals - DYNAMIC_HVE_SKIP_WEEKS; pass t_rebased + quiet_mask + h_eff; align diagnostics with fit_mask_theta
    status: completed
  - id: params-log-summary
    content: Update params_dict / _log_kcor6_fit / create_summary_file columns if new diagnostics keys added; keep backward-compatible fields where practical
    status: completed
  - id: validate-docs
    content: Validate k = mean h on first k_anchor post-HVE weeks (rebased t in [0,k_anchor)); spec YAML note + rebase convention for Gompertz block
    status: completed
isProject: false
---

# Gompertz θ₀ estimation — build plan

## Scope (unchanged vs changed)


| Area                                | Action                                                                                                                                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `invert_gamma_frailty`              | **No code changes**                                                                                                                                                                                          |
| `theta_estimation_windows` in YAML  | **Keep** `[start, end, optional_label]` triples; parsing stays in `_load_dataset_dose_pairs_config` (~lines 1220–1242), `get_theta_estimation_windows`, `_build_theta_quiet_mask` in `code/KCOR.py`          |
| θ₀ estimation                       | **Replace** `fit_theta0_global` (~1050–1125) with `fit_theta0_gompertz` per spec                                                                                                                             |
| Degenerate / identifiability guards | **Keep** `get_theta_degenerate_fit_config`, `_apply_theta_degenerate_guard`, `_apply_theta_min_deaths_guard` (spec “remove theta_min” ↔ no separate θ floor in code today—only min quiet deaths + theta max) |


## YAML changes (minimal)

Add under `time_varying_theta` in e.g. `data/Czech/Czech.yaml` (create block if missing):

```yaml
time_varying_theta:
  gompertz_gamma: 0.085    # per year; default 0.085 in code if omitted
  k_anchor_weeks: 4      # default 4 if omitted
  # existing optional keys still work:
  # degenerate_fit: { theta0_max: 100, action: set_zero }
  # min_quiet_deaths: 30
```

**Do not** nest or duplicate `theta_estimation_windows` under `time_varying_theta`; leave the existing **top-level** list as-is.

## Loader and effective config

1. Extend `_load_dataset_dose_pairs_config` in `code/KCOR.py` to read `gompertz_gamma` and `k_anchor_weeks` from `time_varying_theta` (same dict already used for degenerate config ~1257+).
2. Add a small cached struct (e.g. `_DATASET_GOMPERTZ_THETA_CFG_CACHE`) and include it in the loader’s “all caches loaded” guard so one missing key does not skip loading.
3. Add `get_gompertz_theta_fit_config()` returning at least:
  - `gamma_per_year` (default `0.085`)
  - `k_anchor_weeks` (default `4`)
  - `gamma_per_week = gamma_per_year / 52.0`
4. In `process_workbook` configuration dump (~4861–4891), print effective `gompertz_gamma` and `k_anchor_weeks` next to theta window / degenerate lines.

## Time axis — pipeline vs Gompertz fit

**Pipeline `t` (unchanged everywhere else in KCOR):**

- `t` = `groupby(...).cumcount()` after sort by `DateDied` (~5385 etc.): **first row has `t = 0` at enrollment / start of series**.
- `DYNAMIC_HVE_SKIP_WEEKS` zeros `hazard_eff` for `t < skip` only; `**t` is not rebased** in the dataframe.

**Gompertz θ₀ fit only — rebase at call sites (recommended):**

- Before calling `fit_theta0_gompertz`:

```python
  t_rebased = t_vals - float(DYNAMIC_HVE_SKIP_WEEKS)
  

```

- Pass `**t_rebased**` into the fitter (not enrollment `t_vals`). **Do not pass `hve_skip` into the fitter** — origin is fixed inside the rebased axis.
- **Physics:** `t_rebased = 0` is the **first week with nonzero accumulated-hazard policy** (first “real” hazard week). Then `**H_gompertz(0) = 0`** exactly, `h ≈ k` at `t_rebased = 0` under the model, and `**anchor_mask = (t_rebased >= 0) & (t_rebased < k_anchor_weeks)**` with no skip arithmetic inside the fitter.
- **Why not use enrollment `t` in the fitter?** With enrollment `t`, `H_gompertz(hve_skip) > 0` while pre-skip rows have `h_eff = 0`, which misaligns “Gompertz clock starts” vs “hazard accumulation starts” and slightly biases `k`/shape.

## Core implementation: `fit_theta0_gompertz`

Implement next to the removed `fit_theta0_global`, following `KCOR_v7_gompertz_instructions.md` with **rebased time** and the **anchor-in-fit** rule:

- **Inputs:** `h_arr` (`hazard_eff` / `h_eff_obs`), `**t_rebased`** (= enrollment `t` − skip, computed at call site), `quiet_mask`, `k_anchor_weeks`, `gamma_per_week`, `theta0_init`. **No `hve_skip` parameter.**
- **Anchor mask:** `anchor_mask = (t_rebased >= 0) & (t_rebased < k_anchor_weeks)` (equivalent to enrollment-week anchor after rebase).
- **k:** mean of `h_arr[anchor_mask]`; handle empty anchor / tiny k per spec.
- **H_Gompertz:** `(k/gamma) * expm1(gamma * t_rebased)` (vector over all rows — for diagnostics; fit uses masked rows only).
- **Model on fit points:** `h = k * exp(gamma * t_rebased) / (1 + theta0 * H_gom)`.
- **Fit sample:** `fit_mask_theta = (quiet_mask | anchor_mask) & (t_rebased >= 0)` then intersect finite `h_arr`. The `**t_rebased >= 0`** clause drops pre-HVE “ghost” rows if any ISO-quiet hit applies there. **Anchor weeks stay in the fit** (same as before).
- **Return:** `theta0_hat`, `k_hat`, `relRMSE`, `n_fit`, `converged`, `status` (`ok` / `insufficient_data` / `anchor_only` / `error:...`).

**Threshold:** `n_fit < 3` → `insufficient_data`.

## Call sites (3 places)

Replace `fit_theta0_global(h, H_obs, fit_mask)` with `fit_theta0_gompertz(...)`:

**Every call site:** `t_rebased = t_vals - float(DYNAMIC_HVE_SKIP_WEEKS)` (or the SA loop’s `skip_weeks` variable — same numeric value).

1. **Main per-(YoB, dose)** ~5541–5546: `quiet_mask` from `_build_theta_quiet_mask`, `hazard_eff`, `**t_rebased`** into fitter.
2. **All-ages (YoB -2)** ~5754–5757: same.
3. **SA mode** ~5127–5131: ISO mask as `quiet_mask`, `**t_rebased`**, `h_eff_obs`.

Align min-deaths / H-span diagnostics with `fit_mask_theta` unless explicitly documented otherwise.

**Downstream mapping:**

- `theta_hat` ← `theta0_hat` (then existing guards).
- `success` ← `status == "ok"` and optimizer converged (treat `anchor_only` as success for optimization but **log clearly** — θ poorly identified from early weeks alone).
- `n_obs` ← `n_fit`.
- `rmse_hazard` ← RMS of residuals for backward compatibility with `_log_kcor6_fit` and `params_dict["rmse_Hobs"]`; optionally add `relRMSE_hazard_fit` (spec `relRMSE`) so it is not confused with `relRMSE_HobsSpan` (cumulative-H span).

`**k_hat` in `params_dict`:** use anchor **k** from the new fitter (not jointly fitted k).

**Tabular `relRMSE` / degenerate logs:** Prefer spec hazard `relRMSE` as the primary printed metric; keep `relRMSE_HobsSpan` in the spreadsheet for continuity unless you explicitly rename—note in release notes.

## Remove / dead code

- Delete `fit_theta0_global`.
- Grep for `fit_theta0_global` across the repo (tests, scripts).

## Validation (from spec)

- Czech workbook, enrollment `2021_24`: **k** equals mean `hazard_eff` over `**t_rebased ∈ [0, k_anchor_weeks)`** (first `k_anchor_weeks` post-HVE rows with finite `h`).
- **Anchor in fit:** every such anchor row with finite `h` is in the θ least-squares sample.
- Spot-check θ₀ monotonicity dose 0 by YoB (spec ranges approximate).
- `invert_gamma_frailty`: for fixed θ and same `H_obs`, output unchanged (sanity test).
- MC / negative control paths if applicable.

## Docs touch-up

- At top of `documentation/specs/KCORv7/KCOR_v7_gompertz_instructions.md`, note that `theta_estimation_windows` stays **root-level triple-list** in this repo; nested `start`/`end` YAML in the spec is non-normative here.
- In that spec, amend the “Select quiet window points” step: **k-anchor weeks included** in θ fit (`quiet_mask | anchor_mask`), and document `**t_rebased = t - DYNAMIC_HVE_SKIP_WEEKS`** for the Gompertz block so `H_gom(0)=0` at first accumulating week.

## Risk / edge cases

- **Anchor / series length:** if fewer than `k_anchor_weeks` post-HVE rows exist, anchor mean still defined over available `anchor_mask`; if anchor empty or `k` tiny → insufficient.
- `**quiet_mask` all False:** `fit_mask_theta` → `anchor_mask` only; `n_fit < 3` → insufficient, θ₀ = 0. If `n_fit ≥ 3` on anchor-only → set `**status = 'anchor_only'`**, log `[KCOR7_THETA_ANCHOR_ONLY]` (θ poorly identified from early weeks alone).
- **γ:** `gamma_per_week = gompertz_gamma / 52.0` everywhere (per spec).
- **Optional rollback:** env flag to restore legacy fit only if you need bisection during rollout.


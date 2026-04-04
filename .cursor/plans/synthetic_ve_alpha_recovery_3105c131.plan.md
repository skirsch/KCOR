---
name: Synthetic VE alpha recovery
overview: Tighten synthetic cohort construction to full strata blocks, add a cross-dose completeness filter on `primary_df` before conditional-VE diagnostics, implement `test_ve_alpha_recovery` with assertions and reporting, and append results into the existing synthetic VE report produced by `synthetic_recovery`.
todos:
  - id: simulate-truncate
    content: Truncate simulate_synthetic_table to n_effective full strata blocks; warn if n_cohorts not divisible by 6
    status: completed
  - id: filter-helper
    content: Add filter_synthetic_primary_cross_dose_pairs(primary_df) grouped by enrollment_date, yob_decade, iso_int
    status: completed
  - id: conditional-ve-filter
    content: Apply filter in conditional_ve_alpha_identification before base and adjusted evaluate_synthetic_best_with_diagnostics
    status: completed
  - id: test-ve-alpha-recovery
    content: Implement test_ve_alpha_recovery (10 reps; require ≥8 successful reps; mean alpha_hat_raw within 0.05 of 1.2; print/embed table)
    status: completed
  - id: wire-synthetic-recovery
    content: Call test from synthetic_recovery when enabled; append ve_alpha_recovery_report to ve_report; extend return dict
    status: completed
isProject: false
---

# Synthetic VE recoverability: cross-dose filter + `test_ve_alpha_recovery`

## Context (current code)

- Synthetic tables are built in [`test/alpha/code/estimate_alpha.py`](test/alpha/code/estimate_alpha.py) (`simulate_synthetic_table`, ~1080–1161). `cohort_strata` is already the 6-tuple you listed; [`n_cohorts: 18`](test/alpha/params_alpha.yaml) yields 3 full enrollment blocks (18 = 3×6).
- Pairwise and collapse objectives group weeks by **`iso_int` only** ([`evaluate_pairwise_objective`](test/alpha/code/estimate_alpha.py) ~707–714, [`evaluate_collapse_objective`](test/alpha/code/estimate_alpha.py) ~770–771). So at a given week, all cohorts (all YOBs and doses) enter the same bucket; pairs like ref@1930 vs ref@1940 still count as “within-dose” relative to ref-vs-vacc.
- [`conditional_ve_alpha_identification`](test/alpha/code/estimate_alpha.py) (~1489–1599) builds `primary_df` via `build_synthetic_primary_subset`, runs `evaluate_synthetic_best_with_diagnostics` on unfiltered `primary_df` (base) and on `apply_conditional_ve_adjustment(...)` output (adjusted).

## 1. `simulate_synthetic_table`: full strata blocks only

**Goal:** Never emit a **partial** last block of cohorts that would break “one ref + one vacc per YOB per enrollment group.”

- Introduce `n_strata = len(cohort_strata)` (6) and `n_effective = (n_cohorts // n_strata) * n_strata`.
- Loop `for cohort_idx in range(n_effective):` instead of `range(n_cohorts)`.
- If `n_cohorts % n_strata != 0`, log a single clear warning (or `print`) that `n_cohorts` was truncated to `n_effective` so each enrollment block stays a full Cartesian product of the 3 YOB decades × 2 doses.

This matches “restrict … per YOB/enrollment-group only” without changing the literal `cohort_strata` list.

**Note:** This does **not** by itself remove ref–ref or vacc–vacc pairs in the pairwise objective (those arise across YOBs within the same `iso_int` group).

## 2. Helper: cross-dose completeness filter

Add a small function (e.g. `filter_synthetic_primary_cross_dose_pairs(primary_df: pd.DataFrame) -> pd.DataFrame`) that:

- Groups by `(enrollment_date, yob_decade, iso_int)` (types consistent with columns on synthetic primary frames).
- For each group, requires both `SYNTHETIC_DOSE_REFERENCE` (0) and `SYNTHETIC_DOSE_VACCINATED` (2) in `dose`.
- Keeps rows only for groups that satisfy that; drops groups with a single dose represented.

Use the existing module constants [`SYNTHETIC_DOSE_REFERENCE`, `SYNTHETIC_DOSE_VACCINATED`](test/alpha/code/estimate_alpha.py) (~30–31).

## 3. `conditional_ve_alpha_identification`: filter before diagnostics

After `primary_df = build_synthetic_primary_subset(...)`, replace with:

```text
primary_df = filter_synthetic_primary_cross_dose_pairs(primary_df)
```

Then:

- Base path: `evaluate_synthetic_best_with_diagnostics(primary_df, ...)`.
- Adjusted path: `adjusted_df = apply_conditional_ve_adjustment(primary_df, ve_assumed)` (so adjustment applies to the same filtered rows).

Preserve `validate_conditional_ve_adjustment` by validating against the **pre-adjustment** filtered frame (pass the filtered `primary_df` as `original_df`).

**TODO in code (do not skip):** Immediately after applying `filter_synthetic_primary_cross_dose_pairs` in `conditional_ve_alpha_identification` (and document the same intent beside the filter helper if helpful), add a short `# TODO` comment stating that this filter only guarantees ref+vacc within each `(enrollment_date, yob_decade, iso_int)` cell, while `evaluate_pairwise_objective` still groups by `iso_int` alone—so cross-YOB pairs (e.g. ref@1930 vs vacc@1940) still enter the objective and can contaminate θ geometry unrelated to VE. If the recovery test fails, follow up with single-`yob_decade` evaluation or a pairwise mode that groups by `(enrollment_date, yob_decade, iso_int)` and only uses the ref–vacc contrast.

## 4. `test_ve_alpha_recovery`

New standalone function in the same file, signature along the lines of `test_ve_alpha_recovery(cfg: dict, alpha_values: np.ndarray) -> dict` returning structured results for reporting (e.g. per-rep rows + pass flag + mean bias per estimator).

**Fixed DGP / analysis settings (per your spec):**

- `alpha_true = 1.2`, `ve_multiplier = 0.5`, `noise_model = "lognormal_fixed"`, `ve_assumed = 0.5`, `n_reps = 10`.
- Seeds: derive from `int(cfg["synthetic"]["seed"])` with a fixed large offset (e.g. `+ 900_000 + rep`) so reps are reproducible and disjoint from other synthetic loops.

**Per rep:**

1. `table = simulate_synthetic_table(cfg, alpha_true, seed, noise_model=..., ve_multiplier=0.5)`
2. `primary_df = filter_synthetic_primary_cross_dose_pairs(build_synthetic_primary_subset(table, NEUTRALIZATION_REFERENCE))`
3. `adjusted_df = apply_conditional_ve_adjustment(primary_df, 0.5)`; optional `validate_conditional_ve_adjustment(primary_df, adjusted_df, 0.5)`
4. `_, metrics_df, _ = evaluate_synthetic_best_with_diagnostics(adjusted_df, cfg, alpha_values, NEUTRALIZATION_REFERENCE, seed_base=...)`
5. Record `alpha_hat_raw` for `pairwise` and `collapse`; `bias = alpha_hat_raw - alpha_true`.

**Empty diagnostics guard:** [`evaluate_synthetic_best_with_diagnostics`](test/alpha/code/estimate_alpha.py) returns empty `metrics_df` when `set(best_df["estimator"]) != {"pairwise", "collapse"}` (~1241–1242), so a rep can “fail” silently with no row. Filtered or reduced `primary_df` makes that more likely.

- Count a rep as **successful** only when `metrics_df` is non-empty and contains both `pairwise` and `collapse` with finite `alpha_hat_raw`.
- **Assert** `n_successful >= 8` (out of 10). If this fails, `raise AssertionError` with a clear message (e.g. how many reps succeeded vs expected 10, and that empty diagnostics are not acceptable).
- **Then** compute `mean_pair`, `mean_collapse` over **successful reps only** and require `abs(mean_pair - 1.2) <= 0.05` and `abs(mean_collapse - 1.2) <= 0.05`.
- On mean-bias failure, `raise AssertionError` with both means, `alpha_true`, tolerance, successful rep count, and per-estimator mean bias as useful.

For reps that did not produce metrics, still include rows in the printed table (e.g. `alpha_hat_raw`/`bias` as `NA` or skip those rows but mention the count in the summary) so the report explains missing data.

**Console / report table:** print (and embed in markdown) lines like: `rep | estimator | alpha_hat_raw | bias` for each rep × estimator (up to 20 lines), using `NA`/explicit missing markers when that rep had no diagnostics, plus a one-line summary including **successful rep count**, mean `alpha_hat_raw`, and mean bias per estimator (means over successful reps only).

**Grid check:** `alpha_grid` in [`params_alpha.yaml`](test/alpha/params_alpha.yaml) already spans 1.00–1.30 with step 0.005, so 1.2 is on-grid.

## 5. Wire into `synthetic_recovery` and the VE report

- At the end of [`synthetic_recovery`](test/alpha/code/estimate_alpha.py) (after the main loops, before or after `validate_synthetic_ve_regression`), when synthetic is **enabled**, call `test_ve_alpha_recovery(cfg, alpha_values)` and let **`AssertionError` propagate** (no handling). **Do not** wrap this in `try`/`except AssertionError: pass` or any pattern that swallows the failure—if a `try`/`except` block is used for logging only, the handler must **`raise`** (re-raise the same exception). The only acceptable outcome on assertion failure is process exit via uncaught `AssertionError` (or explicit re-raise).
- Extend the return dict with a string field, e.g. `ve_alpha_recovery_report: str`, containing the markdown section (title, PASS line with means, the table).
- In `synthetic_recovery`, append that section to `ve_report` after `render_synthetic_vaccine_effect_report(...)` (concatenate with a horizontal rule / newline), so [`write_outputs`](test/alpha/code/estimate_alpha.py) continues to write a single [`alpha_synthetic_vaccine_effect_report.md`](test/alpha/out/alpha_synthetic_vaccine_effect_report.md) that now includes pass/fail context.
- When synthetic is disabled (`synthetic.enabled: false`), skip the test and set `ve_alpha_recovery_report` to an empty string or a short “skipped” note consistent with existing early-return behavior.

No change to [`write_outputs`](test/alpha/code/estimate_alpha.py) signature is required if the extra content is folded into `synthetic_ve_report`.

## 6. Technical caveat (pairwise geometry)

Same substance as the **TODO** in section 3: expect the recovery test **may** fail until a follow-up changes pairwise grouping or restricts YOB. The empirical gate plus ≥8/10 successful reps makes failure visible; the in-code `TODO` flags the fix path for the next iteration.

## Files to touch

| File | Change |
|------|--------|
| [`test/alpha/code/estimate_alpha.py`](test/alpha/code/estimate_alpha.py) | Truncate `simulate_synthetic_table` cohort loop; add filter helper; update `conditional_ve_alpha_identification`; add `test_ve_alpha_recovery`; extend `synthetic_recovery` return + `ve_report` concatenation |
| [`test/alpha/README.md`](test/alpha/README.md) | Optional one-sentence note that the synthetic VE report includes the conditional-VE α recovery gate (only if you want docs updated; skip if you prefer minimal diff) |

## Verification

- Run the alpha sandbox with default config (WSL): `estimate_alpha` / Makefile target you normally use; confirm run completes, `alpha_synthetic_vaccine_effect_report.md` contains the new section, and conditional VE CSVs reflect filtered cohort counts if row counts change.
- If CI runs this path, the `AssertionError` will fail the job when recoverability regresses (intended).

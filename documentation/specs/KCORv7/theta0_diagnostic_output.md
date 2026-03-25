Let me read the Excel file first to understand what's currently being logged.Now I have a clear picture. The current output has no theta0 diagnostics at all — no `theta0`, no `k`, no `delta`, no flags like `DELTA_INAPPLICABLE`. The `dose_pairs` sheet has `theta_num` and `theta_den` but these appear to be applied correction values, not the raw estimation diagnostics. Here's the Cursor instruction:

---

**Add a `theta0_diagnostics` sheet to the KCOR output Excel file.**

The current output has no visibility into the theta0 estimation process — there's no way to diagnose why specific cohorts get `DELTA_INAPPLICABLE`, whether `delta < 0` is being triggered correctly or spuriously, or what the quality of the Gompertz fit is. Add a new sheet called `theta0_diagnostics` written once per unique `(EnrollmentDate, YearOfBirth, Dose)` cohort.

**Columns to include:**

| Column | Description |
|--------|-------------|
| `EnrollmentDate` | e.g. `2021_13` |
| `YearOfBirth` | birth year or `-2` for all-ages |
| `Dose` | 0, 1, 2, 3, 4 |
| `k_hat` | fitted baseline hazard at t=0 |
| `theta0_hat` | fitted frailty variance |
| `n_quiet_bins` | number of quiet-window bins used in the theta fit |
| `n_iter` | iterations taken to converge |
| `delta_raw` | raw delta value before any clamping (one column per wave, e.g. `delta_raw_1`, `delta_raw_2`) |
| `delta_applied` | delta actually used after clamping (one column per wave) |
| `delta_negative` | boolean — was any raw delta < 0? |
| `status` | one of: `OK`, `DELTA_INAPPLICABLE`, `DELTA_NEGATIVE_CLAMPED`, `INSUFFICIENT_DEATHS`, `NOT_IDENTIFIED` |
| `quiet_window_start` | first week index of the quiet window used |
| `quiet_window_end` | last week index of the quiet window used |
| `fit_residual_max` | max absolute residual on the quiet window at convergence |
| `fit_residual_rms` | RMS residual on the quiet window at convergence |

**The key diagnostic we need:** `delta_raw` before clamping. This is the field that will tell us whether `DELTA_INAPPLICABLE` is firing because of a genuine model issue (baseline miscalibration, post-vaccination mortality spike) or a data/implementation problem. Right now that value is silently discarded.

**Also add these two columns to the existing `dose_pairs` sheet:**
- `theta0_status_num` — the status flag for the numerator cohort
- `theta0_status_den` — the status flag for the denominator cohort

This lets anyone filtering the output immediately see which KCOR values are backed by a clean theta0 estimate versus a fallback.

## additional info to tweak the plan

The plan looks good and correctly captures everything. A few additions and one important correction:

**Correction: status priority order is wrong**

The current priority in item 2 puts `DELTA_INAPPLICABLE` before `INSUFFICIENT_DEATHS`. It should be the reverse — `INSUFFICIENT_DEATHS` means there isn't enough data to fit anything at all, which is a more fundamental failure than the delta fallback. The correct priority order is:

1. `NOT_IDENTIFIED` — identifiability check failed, no fit attempted
2. `INSUFFICIENT_DEATHS` — not enough deaths in quiet window, theta set to zero
3. `DELTA_INAPPLICABLE` — delta ≤ 0, fell back to single-window fit
4. `DELTA_NEGATIVE_CLAMPED` — delta was negative but iteration continued (this should arguably not exist post-fix — see below)
5. `OK`

**Additional items to add to the plan:**

**Item 1 addition:** Also capture `theta0_init` — the initial theta estimate from the pre-wave window before iteration begins. This is the single most useful diagnostic for the double-fixed-point problem we identified: if `theta0_init` is far from `theta0_hat`, the iteration had to travel a long way and is more likely to have landed on the wrong fixed point. Add it as a column.

**Item 2 addition:** Add `n_quiet_bins_prewave` separately from `n_quiet_bins` — the number of bins in the pre-wave window specifically. This is the quantity that directly predicts estimation quality at low mortality. When `n_quiet_bins_prewave` is very small (fewer than ~10), the initial estimate is unreliable and the diagnostics should reflect that.

**Item 5 addition:** Specifically verify that `delta_raw_*` is populated even for `DELTA_INAPPLICABLE` cohorts — this is the whole point of adding it. The current code discards the raw delta before logging, which is exactly what prevents diagnosis of whether the negative delta is legitimate or spurious. Confirm the raw value is written to the sheet regardless of status.

**Architectural note for Cursor:** The `delta_raw_*` and `delta_applied_*` columns are dynamic — their count depends on the number of waves in the data. Cursor should use sorted wave indices as suffixes (e.g. `delta_raw_1`, `delta_raw_2`) and ensure the `theta0_diagnostics` builder handles the case where different cohorts may have different numbers of detected waves gracefully, filling missing wave columns with `NaN` rather than failing.

### tweak 2
This plan is clean and ready. The `DELTA_NEGATIVE_CLAMPED` removal is correctly reflected throughout — the status ladder now has exactly four values and the validation checklist matches.

Two final things to tell Cursor alongside this plan:

**1. The `delta_negative` boolean needs careful placement in the fallback logic.** It must be set *before* the `DELTA_INAPPLICABLE` branch exits — currently the fallback probably returns early without ever writing the delta fields. The sequence inside `fit_theta0_gompertz` should be:

```
compute delta_raw for all waves
set delta_negative = any(delta_raw < 0)
write delta_raw_* to return payload   ← must happen here, before any branching
if delta_negative → set status=DELTA_INAPPLICABLE, run fallback fit, return
else → continue with delta iteration
```

If this order isn't enforced explicitly, Cursor will likely put the field writes after the early return and `delta_raw_*` will be `NaN` for all `DELTA_INAPPLICABLE` rows — defeating the entire diagnostic purpose.

**2. `theta0_init` should also be written before the fallback branch.** The pre-wave seed estimate is computed before the delta check, so it's available regardless of status. It's especially valuable for `DELTA_INAPPLICABLE` cohorts because it's the only theta estimate you have — and comparing it to `theta0_hat` (the fallback single-window result) tells you whether the two windows agree. If they diverge substantially, that's a strong signal of post-vaccination mortality distortion in the inter-wave period.

## tweak 3
Both .xlsx files should be updated; they serve different purposes and should contain different things.

**KCOR.xlsx** — full per-cohort diagnostics table, exactly as specced. This is the primary artifact for anyone investigating specific cohorts. The `theta0_diagnostics` sheet with every column belongs here because the audience is someone opening the file to debug or audit a specific `(EnrollmentDate, YearOfBirth, Dose)` combination.

**KCOR_summary.xlsx** — aggregated flags only, not the full table. The summary is a high-level operational view, so adding 15 columns of per-wave deltas would bury the signal. What belongs there is:

- A count of cohorts by status: how many `OK`, `DELTA_INAPPLICABLE`, `INSUFFICIENT_DEATHS`, `NOT_IDENTIFIED` per enrollment date
- A flag column on existing summary rows indicating whether any cohort in that enrollment group has a non-OK status
- Possibly `theta0_hat` and `n_quiet_bins_prewave` on the per-cohort summary rows, since those two numbers directly predict KCOR reliability

The practical reason to include anything in the summary at all is that most users will open `KCOR_summary` first. If a run has 40 `DELTA_INAPPLICABLE` cohorts they need to know immediately, without having to open the full workbook and filter the diagnostics sheet.

So the instruction to Cursor would be: write the full diagnostics table to `KCOR.xlsx` as specced, and add a compact `theta0_status` summary block to `KCOR_summary` — counts by status per enrollment date, plus `theta0_hat` and `n_quiet_bins_prewave` appended to any existing per-cohort rows.

### additional diagnostics
Here are the Cursor instructions:

---

**Add vaccine-contamination diagnostics to `fit_theta0_gompertz` and the `theta0_diagnostics` sheet.**

The hypothesis is that for vaccinated cohorts (dose > 0), the pre-wave quiet window used to fit `k` is contaminated by elevated post-vaccination mortality lasting up to ~26 weeks. This makes `k` overestimated, `H_gom` too high, and delta artificially negative. The diagnostics below are designed to make this contamination visible and quantifiable without changing any existing estimation logic.

---

**1. Add these fields to the `fit_theta0_gompertz` return payload:**

`h_obs_quiet_slope` — OLS slope of `h_obs` over the pre-wave quiet window bins (units: hazard per week). A significantly negative slope means mortality is declining during the supposedly quiet period — the vaccine harm wearing off. Compute with `numpy.polyfit(t_prewave, h_obs[prewave_mask], 1)[0]`.

`h_obs_first4_mean` — mean of `h_obs` over the first 4 bins of the pre-wave quiet window.

`h_obs_last4_mean` — mean of `h_obs` over the last 4 bins of the pre-wave quiet window. If `h_obs_last4_mean` is materially lower than `h_obs_first4_mean`, the harm is decaying within the window.

`h_obs_first4_vs_last4_ratio` — `h_obs_first4_mean / h_obs_last4_mean`. Values significantly above 1.0 indicate a declining hazard in the quiet window, consistent with vaccine harm decay. Emit `NaN` if either window has fewer than 4 bins.

`quiet_window_duration_weeks` — total number of weeks spanned by the pre-wave quiet window (i.e. `t_prewave[-1] - t_prewave[0] + 1`). If this is less than 26, the window is shorter than the hypothesised vaccine harm duration and `k` should be considered unreliable for vaccinated cohorts.

`k_hat_first4` — `k` estimated using only the first 4 pre-wave quiet bins. Compare to `k_hat` (full-window estimate) to see how much `k` shifts as the window extends — a large drop indicates the harm is decaying within the window.

`k_hat_last4` — `k` estimated using only the last 4 pre-wave quiet bins. This is the closest within-window proxy for true background mortality if harm is decaying. Emit `NaN` if fewer than 4 bins available.

All seven fields must be computed and written to the return payload **before any branching on delta sign or fallback logic**, so they are available regardless of final status.

---

**2. Add one cross-cohort field to the `theta0_diagnostics` sheet builder:**

`k_hat_vs_dose0_ratio` — ratio of this cohort's `k_hat` to the `k_hat` of the dose=0 cohort with the same `EnrollmentDate` and `YearOfBirth`. Compute this in the diagnostics DataFrame builder after all cohorts have been collected, by joining on `(EnrollmentDate, YearOfBirth)` where `Dose == 0` to get the reference `k_hat`, then dividing. Emit `NaN` where no matching dose=0 cohort exists (e.g. all-ages aggregate rows where `YearOfBirth == -2` may need separate handling).

This is the single most important diagnostic: if `k_hat_vs_dose0_ratio > 1` systematically for dose > 0 cohorts of the same age and enrollment, that is direct evidence that the vaccinated cohort's quiet window is contaminated and `k` is overestimated relative to the true background rate.

---

**3. Add to the `theta0_diagnostics` sheet column order**, after the existing residual columns:

```
h_obs_quiet_slope
h_obs_first4_mean
h_obs_last4_mean
h_obs_first4_vs_last4_ratio
quiet_window_duration_weeks
k_hat_first4
k_hat_last4
k_hat_vs_dose0_ratio
```

---

**4. Validation checks to add to item 5 of the existing plan:**

- Confirm `k_hat_vs_dose0_ratio` is `NaN` for all dose=0 rows (they are their own reference)
- Confirm `h_obs_first4_vs_last4_ratio > 1` for at least some dose=1/2/3 cohorts — if every vaccinated cohort shows a flat ratio, either the window is too short to see decay or the hypothesis is wrong
- Confirm all 7 new scalar fields are non-null for `DELTA_INAPPLICABLE` cohorts — these are precisely the cohorts where contamination is suspected and the diagnostics must be populated
- Print a summary to the run log: mean `k_hat_vs_dose0_ratio` by dose, so the contamination pattern is visible without opening the workbook

---

**Do not change any existing estimation logic.** These are read-only diagnostics computed from data already available inside `fit_theta0_gompertz`. No changes to `k` fitting, delta iteration, fallback branching, or KCOR computation.

## tweak 4
Yes, this plan is correct and complete. It accurately captures everything discussed and the scope is properly constrained — diagnostics only, no estimation changes.

Two small things to verify before handing to Cursor:

**1. `k_hat_first4` and `k_hat_last4` computation** — the plan says "using same `h/exp(gamma*t)` transform as `k_hat`". Make sure Cursor understands this means averaging `h_obs[t] / exp(gamma*t)` over those 4 bins, not re-running the full least-squares fit. It should be a direct mean of the de-trended hazard values, consistent with however `k_hat` is computed in the anchor-weeks averaging logic.

**2. The all-ages (`YearOfBirth == -2`) rows in `k_hat_vs_dose0_ratio`** — the plan says "emit NaN when no dose-0 reference exists" but doesn't explicitly address the all-ages case. The all-ages dose=0 row *does* exist (it's the `-2` aggregate), so the join should work. But confirm with Cursor that the join key includes `YearOfBirth == -2` matching correctly and doesn't accidentally drop these rows or produce a spurious NaN — they're arguably the most important rows to get the ratio for since they represent the full population.

Otherwise the plan is ready to run.

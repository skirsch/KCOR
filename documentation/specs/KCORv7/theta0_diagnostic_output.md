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

Here's the Cursor instruction:

---

**Replace `DELTA_INAPPLICABLE` fallback with `INSUFFICIENT_SIGNAL` zero-theta rule.**

This is a spec correction. The current `DELTA_INAPPLICABLE` fallback is based on a wrong diagnosis — it fires primarily on young/low-mortality cohorts where `theta0_raw` hits the optimizer upper bound (100), not on vaccinated cohorts with genuine wave suppression. The fallback single-window fit makes things worse by fitting theta on the weakest data. Replace it entirely with a zero-theta rule.

---

**Changes to make in `code/KCOR.py`:**

**1. Replace the `DELTA_INAPPLICABLE` trigger condition**

Currently the code checks `delta <= 0` after the first reconstruction pass and branches into the fallback single-window fit. Replace this entire branch with:

```python
# If delta <= 0 OR theta0_raw hit the upper bound -> INSUFFICIENT_SIGNAL
if delta <= 0 or theta0_raw >= theta_max * 0.999:
    theta_applied = 0.0
    theta_fit_status = 'insufficient_signal'
    # still write all diagnostic fields (delta_raw, theta0_raw, etc.)
    # do NOT run a separate single-window fallback fit
    return {
        'theta_applied': 0.0,
        'theta0_raw': theta0_raw,
        'theta_fit_status': 'insufficient_signal',
        # ... all other diagnostic fields populated as normal
    }
```

The `0.999` multiplier on `theta_max` catches cases where the optimizer converged to exactly the boundary.

**2. Remove the single-window fallback fit entirely**

Delete the code block that re-fits theta using only the pre-wave quiet window when `DELTA_INAPPLICABLE` fires. It should not exist in any form. The zero-theta rule replaces it completely.

**3. Update the status enum everywhere**

Replace all occurrences of `'delta_inapplicable'` and `'DELTA_INAPPLICABLE'` with `'insufficient_signal'` and `'INSUFFICIENT_SIGNAL'` respectively — in status assignment, logging, log output strings, the `theta0_status` priority ladder, the summary sheet status counts, and any downstream column comparisons.

The priority ladder in the diagnostics builder becomes:
1. `NOT_IDENTIFIED`
2. `INSUFFICIENT_DEATHS`
3. `INSUFFICIENT_SIGNAL` ← replaces DELTA_INAPPLICABLE
4. `OK`

**4. Update `theta_applied` logic downstream**

Wherever `theta_applied` is consumed to compute `adj_cum_hazard` or the KCOR correction, ensure that `theta_applied = 0` produces identity behavior — i.e. no frailty correction is applied and the raw cumulative hazard is used directly. This should already be the case if the formula is `1 / (1 + theta * H)` with theta=0, but verify there are no divide-by-zero or special-case branches that assumed theta > 0.

**5. Update the `theta0_diagnostics` sheet builder**

Add a boolean column `theta_hit_bound` — `True` when `theta0_raw >= theta_max * 0.999`. This makes it easy to distinguish the two triggers for `INSUFFICIENT_SIGNAL` (bound hit vs genuine negative delta) in the diagnostic output.

**6. Update the summary sheet status counts**

In `create_summary_file()` and the `theta0_status_summary` sheet, replace the `DELTA_INAPPLICABLE` column with `INSUFFICIENT_SIGNAL`. Update the `dual_print` log summary accordingly.

**7. Update all spec comments and docstrings**

Replace any comment that says `DELTA_INAPPLICABLE` is "expected and normal for vaccinated cohorts" with the correct explanation: `INSUFFICIENT_SIGNAL` fires when frailty signal is too weak to identify theta reliably, indicated by either a negative delta (model misspecification or boundary artifact) or theta hitting the optimizer upper bound. `theta_applied = 0` is the correct response — no correction is better than a wrong correction.

---

**Do not change:**
- The delta computation logic itself
- The quiet window definition or masking
- The `INSUFFICIENT_DEATHS` path
- The `NOT_IDENTIFIED` path
- Any other estimation logic

**Validation checks to run after implementation:**
- Confirm no cohort has `theta_fit_status = 'delta_inapplicable'` in output — string should not appear anywhere
- Confirm all previously-`DELTA_INAPPLICABLE` cohorts now show `theta_applied = 0.0` and `theta_fit_status = 'insufficient_signal'`
- Confirm `theta_hit_bound` is `True` for the young cohorts (YOB 1970-2000) that were hitting theta=100
- Confirm KCOR values for dose=0 older cohorts (YOB 1920-1950) are unchanged — they should not be affected
- Run lints on `code/KCOR.py` and fix any issues introduced by this change only

## additional info
The plan is correct and complete. It accurately captures everything discussed and the scope is properly constrained.

Two small things to verify before handing to Cursor:

**1. Item 1 trigger condition needs both checks in the right order**

The plan says "trigger when first-pass `delta_negative` is true **or** raw theta is on boundary." Make sure Cursor understands the boundary check requires running the full iteration first to get `theta0_raw`, whereas `delta_negative` can be checked after the very first reconstruction pass. The logic should be:

```
run first reconstruction pass
check delta_negative → if true, return INSUFFICIENT_SIGNAL immediately (no iteration needed)
else run full delta iteration to get theta0_raw
check theta0_raw >= theta0_max * 0.999 → if true, return INSUFFICIENT_SIGNAL
else return OK with theta0_raw as theta_applied
```

This matters because if you check the boundary hit before running the iteration you don't have `theta0_raw` yet, and if you run the full iteration before checking `delta_negative` you waste computation on cohorts that should exit early.

**2. Item 3 verification needs to be explicit**

"Add only minimal defensive checks if needed" is vague. Tell Cursor specifically to verify that the formula `h_adj = h0 / (1 + theta * H0_eff)` with `theta=0` reduces to `h_adj = h0` — i.e. confirm no division, no special branch, just identity passthrough. If there's any code path that does `if theta > 0: apply_correction() else: use_raw()` that's fine, but if the formula has a divide-by-theta anywhere that would blow up at zero, that needs fixing explicitly.

Otherwise the plan is ready to run.

## added
Yes, this plan is ready to hand to Cursor as-is. The two clarifications from the previous review are both correctly incorporated — the explicit 5-step order in item 1 and the theta=0 identity verification in item 3 are exactly right.

One last thing to add verbally when you hand it to Cursor — not a plan change, just a runtime instruction:

**After implementation, run the full Czech dataset and check three specific numbers before declaring success:**

1. Count of `insufficient_signal` rows — should be roughly equal to the previous count of `delta_inapplicable` (48) plus any new bound-hit catches. If it's dramatically higher or lower something went wrong in the trigger logic.

2. KCOR value for `2021_13 | All Ages | dose=1 vs dose=0` — this was `1.1935` in the current run. It should shift after the fix since some of those cohorts were getting wrong theta corrections. Note the new value as the v7.5 baseline.

3. Confirm zero rows with `relRMSE > 1e9` that also have `theta_fit_status = ok` — the catastrophic relRMSE cases on dose=1 older cohorts should either resolve or become `insufficient_signal`. If they stay `ok` with billion-level relRMSE after this change, that's a separate bug that needs its own fix next.

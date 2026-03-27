# insufficient deaths fix

Looking at the data first to make sure the instruction is precise:The root cause is completely clear now. These are **tiny populations** (164-477 people) with extremely noisy mortality data — zero deaths in many weeks, then 2-9 in a single week purely by chance. The `k_hat` anchor average over the first 4 weeks is meaningless at this population size, and `relRMSE` is in the billions because the model is trying to fit a smooth Gompertz curve to essentially random binary noise.

Note that `2022_26` and `2022_47` already correctly show `insufficient_deaths=True` in the full row data — but the status is still `ok`. That's the bug: `insufficient_deaths` is being detected but not acted on.

Here's the Cursor instruction:

---

**Fix: enforce `INSUFFICIENT_DEATHS` status when `insufficient_deaths=True`.**

In `code/KCOR.py`, the `insufficient_deaths` flag is being computed correctly but is not being used to gate the status. These three cohorts (`2021_30`, `2022_26`, `2022_47`, all YOB=1920 dose=1) have `insufficient_deaths=True` in the diagnostics but `theta_fit_status='ok'` — they should be `INSUFFICIENT_DEATHS` with `theta_applied=0`.

The root cause is tiny populations (164-477 people) producing zero deaths in most weeks, making the `k_hat` anchor average and hazard fit meaningless. The `relRMSE` values of 2-3 billion are the symptom.

**Specific fix:**

In `fit_theta0_gompertz`, after computing `total_quiet_deaths` and setting `insufficient_deaths`, ensure the following gate is applied **before any fitting proceeds**:

```python
if insufficient_deaths:
    return {
        'theta_applied': 0.0,
        'theta0_raw': 0.0,
        'theta_fit_status': 'insufficient_deaths',
        # populate all diagnostic fields with available values
        # do NOT attempt the fit
    }
```

Currently it appears the `insufficient_deaths` flag is computed but the code falls through to the fit anyway. The early return must happen before `_fit_theta` or any least-squares call.

**Also add a `relRMSE` guard as a second line of defence:**

After the fit completes for any cohort that passes the deaths check, if `relRMSE_hazard_fit > 1e6`, override the status to `INSUFFICIENT_SIGNAL` and set `theta_applied=0`. A relRMSE above one million means the model fit is completely invalid regardless of how it got there — applying a theta from such a fit would corrupt the KCOR correction. Log it as `[KCOR7_BAD_FIT]` with the relRMSE value so it's auditable.

**Validation:**
- Confirm zero `ok` rows with `relRMSE > 1e6` after the fix
- Confirm the three YOB=1920 dose=1 cohorts now show `theta_applied=0`
- Confirm dose=0 older cohorts (YOB=1920-1950) KCOR values are unchanged
- Run lints on `code/KCOR.py`

### note
This plan is correct and more thorough than the instruction I wrote — it correctly identifies the SA path alignment issue and makes the relRMSE threshold configurable rather than hardcoded. Both are good additions.

One thing to verify before handing to Cursor:

**The control-flow diagram has the order wrong.** It shows:

```
badFit → boundCheck → returnOk
```

But based on our spec the correct order is:

```
quietDeaths → fitTheta → deltaOrBound → badFit → returnOk
```

The `delta/bound` check (`INSUFFICIENT_SIGNAL`) should come before the `relRMSE` check (`BAD_FIT`), because if delta is negative we return immediately without even computing relRMSE. The diagram puts `badFit` before `boundCheck` which would mean running the full fit and computing relRMSE even for negative-delta cohorts that should have already exited. Tell Cursor to follow this corrected order:

```
1. pre-fit deaths gate        → INSUFFICIENT_DEATHS (exit)
2. run fit
3. delta negative or bound hit → INSUFFICIENT_SIGNAL (exit)  
4. relRMSE > 1e6              → INSUFFICIENT_SIGNAL / BAD_FIT (exit)
5. return OK
```

Everything else in the plan is correct and ready to run.

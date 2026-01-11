# Create a new identifiability folder in the KCOR repo that examines the 
# data from the data/Czech/records.csv

# you can modify KCOR_CMR.py to support this, just don't break anything!
# or just copy over the code otherwise.

Yes — that makes sense, and it’s a solid identifiability design. It’s basically a **weekly “incident booster” emulation** with **calendar-matched risk sets** and **proper censoring**, which is exactly what you want to separate “dose-timed” effects from “calendar/wave” effects.

A couple of small tweaks will make it cleaner and avoid subtle bias.

## What you proposed, restated precisely

For each enrollment week (E) (start 2021-10-18, then weekly for 10 weeks):

### Cohort definitions use status **as of the week before enrollment** (good)

Let (S = E-7) days (the “as-of” week).

* **Dose 3 (incident)**: people who received **dose 3 during week ([S, E))** (i.e., the week immediately before enrollment).
  This makes dose-3 an “incident” cohort anchored to dose timing.

* **Dose 2 (prevalent)**: people whose **dose count is 2 as of (S)** (and not dose-3 yet as of (S)).

* **Dose 0 (prevalent)**: people with **dose count 0 as of (S)**.

Restrict all to birth years **1930–1939** and alive at enrollment (E).

### Follow-up and censoring (yes)

Follow cohorts forward in calendar weeks (E+t). Censor individuals at:

* **death** (event), or
* **transition to a new dose** (dose2→dose3, dose3→dose4, etc.), or
* end of follow-up window.

This is the right way to keep the risk sets clean. (It’s not “fixed cohorts” anymore, but for identifiability that’s fine — it’s closer to a target-trial style analysis.)

## Two important tweaks

### 1) Don’t put all 3 dose groups “on the same graph” as raw (h(t)) unless you also show ratios

Putting (h_0(t), h_2(t), h_3(t)) on one plot is useful, but the identifiability question is really about whether the **dose-3 effect is time-since-dose locked**.

So also compute and plot:

* (HR_{30}(t) = h_3(t)/h_0(t))
* (HR_{20}(t) = h_2(t)/h_0(t))

Those HR plots are what will show “same peak week across enrollments” cleanly.

### 2) Make the dose-0 and dose-2 groups “as-of (S)” but treat them as **risk sets** during follow-up

With censoring, this is natural: at each week (E+t), the hazard is computed among those still:

* alive, and
* not transitioned (still dose-0 or still dose-2).

That avoids immortal-time / contamination issues.

## Cursor-ready instructions for `KCOR/identifiability/czech_weekly_incident_booster_2021/`

Create:

```
KCOR/identifiability/czech_weekly_incident_booster_2021/
  README.md
  build_weekly_emulation.py
  outputs/
```

### Script behavior

Parameters:

* enrollment_start = 2021-10-18 (Monday)
* n_enrollments = 10 (weekly)
* lookback_days = 7  (status as-of S = E - 7d)
* followup_weeks = 26
* birth_year_min=1930, birth_year_max=1939

For each enrollment date `E`:

1. Define `S = E - 7 days`.
2. Build baseline cohorts (all must be alive at E):

   * Dose3_incident: received dose3 in [S, E)
   * Dose2_prevalent: dose==2 at S
   * Dose0_prevalent: dose==0 at S
3. Follow-up weekly for t=0..followup_weeks-1 using calendar week W = E + 7*t:
   For each cohort d ∈ {0,2,3}:

   * alive_d(E,t): number alive and uncensored at start of week W
   * dead_d(E,t): number who die during week W
   * h_d(E,t)=dead/alive
     Censor rule for each person:
   * censor at death week end (count death as event, then remove)
   * censor at week of first dose transition beyond baseline dose (dose0->1, dose2->3, dose3->4)
4. Compute HR series:

   * HR30(E,t)=h3/h0
   * HR20(E,t)=h2/h0

Outputs:

* `outputs/series.csv` with columns:
  enrollment_date, t, calendar_week,
  dead0, alive0, h0,
  dead2, alive2, h2,
  dead3, alive3, h3,
  HR20, HR30
* Per-enrollment plots:

  * `h_curves_EYYYYMMDD.png` (h0,h2,h3 vs t)
  * `HR_curves_EYYYYMMDD.png` (HR20,HR30 vs t)
* Combined spaghetti plots across all E:

  * `HR30_spaghetti.png`
  * `HR20_spaghetti.png`
* Summary table `outputs/summary.csv` per E:

  * peak_week_HR30, peak_value_HR30, HR30_at_t0, HR30_at_t2, etc.
  * same for HR20

README:

* explain identifiability logic: if HR30 peak week is stable across E → dose-time locking; if it shifts with E → calendar locking.

If you want one extra “safety” variant (recommended): run the exact same thing but define dose3 as **prevalent** at S (not incident in [S,E)) and show that the dose-locked signature weakens. That acts like an internal control.

This design is coherent and implementable, and censoring is the right move for this particular identifiability test.

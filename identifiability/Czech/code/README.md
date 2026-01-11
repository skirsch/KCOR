# Czech weekly incident booster identifiability (2021)

This code implements the analysis described in `documentation/preprint/identifiability.md`:
a **weekly “incident booster” (dose 3)** emulation with **calendar-matched risk sets** and **censoring on dose transitions**.

## Design (high level)

For each weekly enrollment date **E** (starting 2021-10-18, for 10 weeks):

- Define **S = E − 7 days** (the “as-of” week).
- Restrict to **alive at E**. (Default is **all ages**; optionally restrict to a birth-year band.)

Baseline cohorts are defined using status **as-of week-start S**:

- **Dose3_incident**: received dose 3 in the **4 weeks** before enrollment \([E-28d, E)\). With ISO-week data, this means `Date_ThirdDose` is one of the 4 ISO-week Mondays immediately preceding `E`.
- **Dose2_prevalent**: dose count is 2 as-of S (dose2 ≤ S and dose3 is missing (`NaT`) or > S).
- **Dose0_prevalent**: dose count is 0 as-of S (dose1 is missing (`NaT`) or > S).

Follow-up is in calendar weeks \(W(t)=E+7t\), for `t=0..followup_weeks-1`.

### Risk set and censoring (rule B: censor starting next week)

All cohort membership and risk-set inclusion is **week-start**.

For baseline cohort `d`, each person is censored at the earliest of:

- **Death**: removed starting the week after their `DateOfDeath`.
- **Dose transition beyond baseline**: removed starting the week after the first “next dose” date:
  - dose0 → transition at `Date_FirstDose`
  - dose2 → transition at `Date_ThirdDose`
  - dose3 → transition at `Date_FourthDose`

So a transition on week \(w\) means the person is still **at risk in week \(w\)** and removed starting week \(w+1\).

### Edge case: death and transition in the same ISO-week

Under rule B, a person who **dies and transitions in the same ISO-week** remains in the **old cohort for that week**
(status is defined at week-start; censoring starts next week). The script reports how many such cases exist.

## Outputs

Written under the output directory you pass (for `make identifiability`, default is `identifiability/Czech/booster/`):

- `series.csv`: per enrollment and follow-up week:
  - `alive{0,2,3}`, `dead{0,2,3}`, discrete-time hazards `h{0,2,3}`
  - `HR20 = h2/h0`, `HR30 = h3/h0`
- `summary.csv`: per enrollment date summary metrics, including peak HR week/value and edge-case counts.
- Plots:
  - Per-enrollment hazards: `h_curves_EYYYYMMDD.png`
  - Per-enrollment HRs: `HR_curves_EYYYYMMDD.png`
  - Spaghetti plots across enrollments: `HR30_spaghetti.png`, `HR20_spaghetti.png`

## Run

From repo root:

```bash
python identifiability/Czech/code/build_weekly_emulation.py \
  --input data/Czech/records.csv \
  --outdir identifiability/Czech/booster
```

Or via Makefile:

```bash
make identifiability
```

To restrict to a birth-year band (optional):

```bash
make identifiability IDENT_BIRTH_YEAR_MIN=1930 IDENT_BIRTH_YEAR_MAX=1939
```

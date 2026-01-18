# KCOR_CMR Export Specification (derived from 2-table master format)
**Version 1.0**

This export spec defines a derived, analysis-friendly dataset for **fixed enrollment cohorts** in KCOR.  
It is computed from:

- `kcor_weekly_state` (risk sets + deaths/censoring)  
- `kcor_weekly_transitions` (dose transitions)

The goal is to emit a “cohort-followup cube” keyed by **enrollment_week** and **time-since-enrollment**.

---

## 1. Output table: `kcor_cmr`

Each row corresponds to one stratum, one enrollment cohort, one follow-up week.

### Required columns

| Column | Description |
|---|---|
| `enroll_week` | Calendar week used as enrollment date \(E\) |
| `t_week` | Integer follow-up week index \(t=0,1,2,...\) where `cal_week = enroll_week + t_week` |
| `age_band_5yr` | Age band at enrollment week start |
| `sex` | `M`, `F`, `U` |
| `dose_enroll` | Dose state at enrollment week start |
| `n_risk_start` | At-risk count at start of follow-up week \(t\) for this enrolled cohort |
| `d_death` | Deaths during follow-up week \(t\) among `n_risk_start` |
| `c_other` | Non-dose censoring during follow-up week \(t\) among `n_risk_start` |
| `c_transition` | Censoring due to transition to higher dose during follow-up week \(t\) among `n_risk_start` |

### Optional columns (recommended)

| Column | Description |
|---|---|
| `cal_week` | The actual calendar week for follow-up week `t_week` |
| `dose_state_start` | For “no-transition cohorts” this will equal `dose_enroll`; for multi-state exports it may differ |
| `pt` | Person-time in person-weeks for the interval (approx or exact if available) |

---

## 2. Export mode A: Fixed cohort with censor-on-higher-dose (recommended default)

### Cohort membership
At enrollment week `E`:

- Cohort is defined by the population in `kcor_weekly_state` where:
  - `cal_week = E`
  - `dose_state = dose_enroll`
  - `age_band_5yr, sex` match

Cohort membership is fixed at `E` and tracked forward.

### Follow-up recursion (per stratum)
Let `R(t)` be the risk set size at start of follow-up week `t` for this enrolled cohort.

Initialization:
- `R(0) = n_start(E, age_band_5yr, sex, dose_enroll)`

For each follow-up week `t` with calendar week `W = E + t`:

Inputs from master tables:
- `d_death(W, age_band_5yr, sex, dose_enroll)` from state table
- `c_other(W, ...)` from state table
- `c_transition(W, ...)` computed from transition table as:
  - sum of `x_count` where `dose_from = dose_enroll` and `dose_to > dose_enroll`
  - same `(W, age_band_5yr, sex)`

But these master-table flows are defined on the *entire* population in that stratum, not just the enrolled cohort.  
Therefore the cohort-specific flows for follow-up week `t` are computed by proportional allocation:

- Define the stratum “stock” at start of week `W`:
  - `S(W) = n_start(W, age_band_5yr, sex, dose_enroll)`

- Compute scaling factor:
  - `alpha = R(t) / S(W)` (if `S(W)=0`, stop follow-up)

Then cohort-specific flows:
- `d_death_cmr = round_or_float(alpha * d_death_master)`
- `c_other_cmr = round_or_float(alpha * c_other_master)`
- `c_transition_cmr = round_or_float(alpha * c_transition_master)`

Update:
- `R(t+1) = R(t) - d_death_cmr - c_other_cmr - c_transition_cmr`

**Notes**
- This proportional-allocation step is exact only if you assume within-stratum exchangeability.  
- If you want exact fixed cohorts without this assumption, you must build `kcor_cmr` directly from record-level data or from enrollment-indexed aggregates.

Because your earlier goal was “construct anything from aggregates,” this spec makes the assumption explicit.

---

## 3. Export mode B: Enrollment-indexed exact cohorts (if you choose to materialize them)

If you generate enrollment-indexed cubes directly upstream (recommended when possible), then `kcor_cmr` is exact with no proportional allocation.

In that case, for each enrollment week `E` and follow-up `t` you store:
- `n_risk_start`
- `d_death`
- `c_other`
- `c_transition`

computed on the actual individuals enrolled at `E`.

This is the ideal export for KCOR.

---

## 4. KCOR consumption expectations

KCOR computations on `kcor_cmr` assume:

- Hazard estimate per week: `h(t) = d_death / n_risk_start` (or using `pt`)
- Cumulative hazard: cumulative sum over weeks (or Nelson–Aalen)
- KCOR ratio: `H_vax(t) / H_ref(t)`
- Anchoring: divide by KCOR at specified `t0` (quiet-window anchor)

---

## 5. Minimal file header for `kcor_cmr.csv`


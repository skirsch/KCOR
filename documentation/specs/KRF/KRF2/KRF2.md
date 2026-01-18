# KCOR Population Aggregation Specification

**Version 1.0**

[Full conversation](https://chatgpt.com/share/696ab6e5-f274-8009-8e57-7aa87c48111b)


This specification defines the minimal aggregated data structures required to support KCOR analyses, including:

* Fixed enrollment cohorts at arbitrary calendar dates
* Censor-on-transition cohort designs
* Multi-state dose trajectory analysis
* Anchored KCOR normalization
* Arbitrary maximum dose counts (e.g., pediatric schedules)

The design is intentionally calendar-time indexed and lossless at the aggregate-flow level.

---

## Time Resolution

All data are aggregated at **weekly resolution**.

* A "week" is defined as a consistent 7-day reporting interval (ISO week or fixed epidemiological week).
* All counts refer to events occurring within that week.

---

# Table 1 — Weekly State Table (`kcor_weekly_state`)

This table defines the **at-risk population at the start of each week**, stratified by demographic and dose state.

Each row represents a single stratum in a single calendar week.

---

## Required Columns

| Column         | Type            | Description                                                                                             |
| -------------- | --------------- | ------------------------------------------------------------------------------------------------------- |
| `cal_week`     | integer or date | Calendar week identifier (ISO week index or week-start date)                                            |
| `age_band_5yr` | string          | Five-year age bin at start of week (see definition below)                                               |
| `sex`          | categorical     | `M`, `F`, `U` (unknown/other)                                                                           |
| `dose_state`   | integer         | Dose state at start of week (0,1,2,...,N)                                                               |
| `n_start`      | integer         | Number alive and in this dose state at start of week                                                    |
| `d_death`      | integer         | Deaths during the week among `n_start`                                                                  |
| `c_other`      | integer         | Non-dose censoring during week (emigration, registry end, loss to follow-up). Optional but recommended. |

---

## Five-Year Age Band Definition

Age is computed at **start of week**.

Standard bins:

```
0-4
5-9
10-14
...
80-84
85-89
90-94
95+
```

Rules:

* Closed on lower bound, inclusive
* Upper bound exclusive except final bin
* Once assigned within a week, age band is fixed for that week

---

## Semantics

* `n_start` is the population **alive and in the stated dose at the beginning of the week**
* `d_death`, `c_other`, and all transitions must originate from `n_start`
* Events are counted during the week interval

---

## Required Invariants

For every `(cal_week, age_band_5yr, sex, dose_state)`:

```
d_death ≥ 0
c_other ≥ 0
d_death + c_other + Σ(transitions_out) ≤ n_start
```

---

# Table 2 — Weekly Transition Table (`kcor_weekly_transitions`)

This table records **dose transitions occurring during each week**.

Each row represents a directed flow between dose states.

---

## Required Columns

| Column         | Type            | Description                                 |
| -------------- | --------------- | ------------------------------------------- |
| `cal_week`     | integer or date | Calendar week identifier                    |
| `age_band_5yr` | string          | Same binning as state table                 |
| `sex`          | categorical     | `M`, `F`, `U`                               |
| `dose_from`    | integer         | Dose at start of week                       |
| `dose_to`      | integer         | Dose after transition (may be +1, +2, etc.) |
| `x_count`      | integer         | Number transitioning during the week        |

---

## Transition Rules

* Transitions always originate from `dose_from` at start of week
* Multiple destination jumps are permitted (ex: 1 → 3)
* No assumptions are made about maximum dose number

---

## Required Invariants

For each `(cal_week, age_band_5yr, sex, dose_from)`:

```
Σ(x_count) ≤ n_start (from state table)
```

---

# Event Ordering Convention

The following ordering is assumed unless otherwise documented:

1. Population defined at start of week (`n_start`)
2. Deaths and transitions occur during week
3. Deaths take precedence over transitions if both occur in same reporting interval

This convention must remain fixed across the dataset.

---

# Supported Analyses Enabled by This Schema

This structure allows exact construction of:

## Fixed Enrollment Cohorts

Example:

"Enroll dose=2 population on week E and censor at booster"

Procedure:

* Select `n_start` rows where `cal_week = E` and `dose_state = 2`
* Follow forward in calendar time
* Subtract deaths and censor at any transition in Table 2

---

## Dynamic State ("As-Treated") Hazards

Weekly hazard estimation by current dose:

```
h(t) = d_death / n_start
```

Computed directly from Table 1.

---

## Multi-State Propagation

Dose progression modeling:

* Use Table 2 flows to propagate populations between dose states
* Combine with Table 1 mortality

---

## KCOR and Anchored KCOR

For any enrollment date:

* Compute cumulative hazards from weekly hazards
* Form KCOR ratios across dose groups
* Normalize using quiet-window anchor

---

# Recommended File Formats

* Storage: Parquet or compressed CSV
* Sorting order:

Primary key ordering suggestion:

```
cal_week, age_band_5yr, sex, dose_state
```

---

# Validation Checklist

Before analysis:

* Verify mass balance per week
* Confirm no negative values
* Confirm transitions never exceed risk sets
* Confirm age bin assignment logic
* Confirm consistent calendar-week alignment

---

# Summary

This two-table design is:

* Enrollment-date agnostic
* Transition exact
* Censor-safe
* Dose-count unlimited
* Fully KCOR compatible

It is the minimal lossless aggregated representation for population-scale KCOR analysis.

---

If you'd like, I can generate:

* A **schema YAML** version for pipeline validation
* A **CSV header template**
* A **Python/Pandas validator** that checks all invariants automatically
* A **KCOR_CMR export spec** derived from this master format

# KCOR Data Specification v1.0

## 1. Purpose

This document defines the data formats and semantic rules required to support KCOR (Kirsch Cumulative Outcomes Ratio) analyses on observational cohort data, while accommodating a wide range of privacy and disclosure-control regimes.

The specification defines **three data formats**, corresponding to increasing privacy constraints and decreasing disclosure surface:

1. Record-level data  
2. Outcome (mortality/disease) summary data  
3. Hazard summary data  

All three formats are designed to support identical KCOR analyses when interpreted under the shared definitions below.

---

## 2. Core Concepts and Definitions (Shared)

### 2.1 Enrollment and Cohort Freezing

- Each individual is assigned to a cohort **once**, at enrollment.
- Cohort assignment is determined by exposure/intervention status **as of a fixed enrollment cutoff** (e.g., end of enrollment week).
- No reassignment between cohorts occurs after enrollment.
- Cohorts remain fixed for the duration of follow-up.

This rule applies regardless of later exposures, additional interventions, or behavioral changes.

---

### 2.2 Time Axis

- Time is indexed by an integer `t`, representing discrete intervals since enrollment.
- The unit of `t` (e.g., weeks, days) is defined by the data producer.
- Interval boundaries and binning rules (e.g., ISO weeks vs rolling windows) must be stated explicitly.

---

### 2.3 Risk Set Definition

- `n_at_risk(t)` represents the number of cohort members **alive and under observation at the start of interval `t`**.
- Events occurring during interval `t` do not affect the risk set for that interval.

---

### 2.4 Outcomes

- An outcome is defined by a dated event (e.g., infection, death, hospitalization).
- Datasets MAY include multiple outcomes.
- One outcome is designated as the **primary outcome** for KCOR analysis.
- Outcomes are identified by a canonical string label (e.g., `infection`, `death`).

---

### 2.5 Stratifiers

- Stratifiers (e.g., sex, age band, geography, comorbidity index) are optional.
- If included, stratifiers appear as **explicit columns**.
- A stratum is implicitly defined as the unique combination of all stratifier columns present in the file.
- No explicit `stratum_id` column is required or expected.
- Analysis code MAY generate internal stratum identifiers for computational convenience.

---

### 2.6 Disclosure Control

- Summary data MAY be aggregated across time intervals to satisfy minimum count thresholds.
- Once time intervals are merged to meet disclosure thresholds, finer-grained intervals MUST NOT be released if they would enable back-calculation.
- Disclosure control policies and thresholds must be documented.

---

## 3. Format 1: Record-Level Data (`records.csv`)

### 3.1 Description

One row per individual, containing individual-level dates and attributes.

### 3.2 Required Concepts

- Unique individual identifier
- Enrollment date or enrollment interval
- Exposure/intervention dates sufficient to determine cohort at enrollment
- Outcome dates (as available)

### 3.3 Optional Fields

- Sex
- Date of birth or age resolution
- Geography
- Comorbidity indices (e.g., DCCI)
- Additional outcomes

### 3.4 Notes

- Record-level data enables derivation of all downstream formats.
- Typically restricted to secure environments.

---

## 4. Format 2: Outcome Summary Data (`events.csv`)

### 4.1 Description

Aggregated counts by cohort, time interval, outcome, and optional stratifiers.

### 4.2 Required Columns

- `t` : integer time index since enrollment  
- `cohort_id` : cohort identifier  
- `n_at_risk` : number at risk at start of interval  
- `outcome` : outcome label  
- `n_events` : number of outcome events during interval  

### 4.3 Optional Columns

- Stratifier columns (e.g., `sex`, `age_band`, `state`, `dcci_band`)  
- `n_switched` : number receiving another intervention during interval (informational only)  
- `n_admin_censored` : administrative censoring count (if applicable)  

### 4.4 Notes

- `n_at_risk` is not reduced by `n_switched` or non-terminal events.
- Time intervals MAY be merged to satisfy disclosure thresholds.
- This format supports full KCOR reconstruction.

---

## 5. Format 3: Hazard Summary Data (`hazard.csv`)

### 5.1 Description

Hazard rates by cohort, time interval, outcome, and optional stratifiers.

### 5.2 Required Columns

- `t` : integer time index since enrollment  
- `cohort_id`  
- `outcome`  
- `hazard` : estimated hazard for the interval  

### 5.3 Optional Columns

- Stratifier columns (same semantics as Format 2)
- Precision metadata (e.g., rounding, noise indicators)

### 5.4 Notes

- Hazards are assumed piecewise-constant within each interval.
- This format is intended for environments with strict disclosure control.
- No event counts or risk set sizes are included.

---

## 6. YAML Metadata (Recommended)

A companion YAML file SHOULD describe:

- Dataset identity and version
- Jurisdiction and disclosure rules
- Time bin definitions
- Enrollment cutoff semantics
- Cohort definitions
- Outcome definitions
- Available stratifiers

The YAML file defines semantics; the CSV files contain data only.

---

## 7. Versioning

- This specification is versioned.
- Files SHOULD declare the spec version they conform to.
- Backward-compatible extensions are permitted.

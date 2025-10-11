# KCOR Record Format spec (KRF)
The goal of this spec is to define a standard output format that can be interpreted by the KCOR_CMR tool.

Then all we have to do is write a converter for each dataset into the standard format.

One record per person. Columns are case-sensitive.

## Canonical columns

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `ID` | string/int | optional | Unique person identifier if available (stable within dataset). |
| `YearOfBirth` | int | required | Exact 4-digit year of birth (e.g., 1950). If data is banded, see `birthBandYears` in sidecar. |
| `DeathDate` | date (YYYY-MM-DD) | optional | Date of death; empty if alive by `observationEndDate`/`CensorDate`. |
| `Sex` | enum | optional | Allowed: `M`, `F`, `O`; may be empty. |
| `DCCI` | int | optional | Charlson comorbidity index if known; empty otherwise. |
| `CensorDate` | date (YYYY-MM-DD) | optional | Person-specific end of observation if they exit early. |

Vaccination columns follow a numbered pattern and are described below.

## Vaccination columns

- Pattern: `V1Date`, `V1Brand`, `V2Date`, `V2Brand`, …, `VnDate`, `VnBrand` (n ≥ 1).
- `VxDate` is required for a dose to exist; `VxBrand` is optional if `VxDate` is present.
- Dates must be strictly increasing per person: `V1Date` < `V2Date` < … < `VnDate`.
- If multiple doses occur on the same date, consolidate to a single dose for that date.
- Brand codes (optional):
  - `P` = Pfizer/BioNTech
  - `M` = Moderna
  - `J` = Johnson & Johnson
  - `O` = Other/Unknown

Parsers should accept arbitrarily many doses (continue until the first missing `VxDate`/`VxBrand` pair, ignoring extraneous undefined columns).

## CSV and date conventions

- Encoding: UTF-8 (no BOM), comma-separated, header row required.
- Quoting: Standard CSV quoting with double-quotes (`"`) when needed.
- Column order: Arbitrary; headers must match exactly.
- Missing values: use empty string; do not use `NA`, `NULL`, or `\N`.
- Dates: ISO 8601 `YYYY-MM-DD`. No time component; local civil calendar. Timezone may be documented in sidecar as `timezone`.

## Validation and data quality rules

- `YearOfBirth` ∈ [1900, current_year].
- `DeathDate` ≥ `1900-01-01` if present.
- Dose dates must be strictly increasing per person.
- `DeathDate` must be ≥ the earliest recorded date for that person; flag if violated.
- Each row must correspond to a unique person (uniqueness determined by dataset; `ID` recommended).
- Additional columns are allowed; consumers should ignore unknown columns.

## Observation window

- Dataset-level `observationEndDate` should cap follow-up for anyone without `DeathDate` or `CensorDate`.
- Optional per-person `CensorDate` can end observation earlier for that person.
- Provide dataset-level settings in a YAML sidecar (see example below) or via tool configuration.

## Minimal CSV example

```csv
ID,YearOfBirth,DeathDate,Sex,DCCI,V1Date,V1Brand,V2Date,V2Brand
234,1950,2021-05-20,M,1,2021-03-10,P,2021-10-01,M
,1955,,F,,2022-02-15,M,,
```

## Optional YAML sidecar (dataset metadata)

```yaml
schema: KRF-1.0
observationEndDate: 2024-12-31
birthBandYears: 5  # allowed: 1, 5, 10
timezone: local    # dates are local calendar; no time component
notes: "Japan: birth years provided in 5-year bands"
```

Notes:
- If `birthBandYears` > 1, interpret `YearOfBirth` as the lower bound of the band (e.g., `1950` with `birthBandYears: 5` means [1950-01-01, 1955-01-01)).
- `Sex` is optional; KCOR aggregates across sexes unless stratification is requested.
- Manufacturer/brand is optional; dose count and dates are the primary fields used.

## YAML schema (optional validation)

Use the following JSON Schema (expressed in YAML) to validate the sidecar metadata.

```yaml
# JSON Schema (YAML form) for KRF sidecar metadata
$schema: "https://json-schema.org/draft/2020-12/schema"
title: "KRF Sidecar Metadata Schema"
type: object
additionalProperties: true
properties:
  schema:
    type: string
    pattern: "^KRF-\\d+\\.\\d+$"
    description: "KRF spec version, e.g., KRF-1.0"
  observationEndDate:
    type: string
    format: date
    description: "Dataset-level end of observation window (YYYY-MM-DD)"
  birthBandYears:
    type: integer
    enum: [1, 5, 10]
    description: "Width of birth year bands; interpret YearOfBirth as lower bound when >1"
  timezone:
    type: string
    description: "Timezone context; 'local', 'UTC', or an IANA TZ identifier (e.g., 'Asia/Tokyo')"
    default: "local"
  notes:
    type: string
    description: "Freeform notes about dataset transformations or assumptions"
required: []
```

## Versioning

- This document defines `KRF-1.0`. Future revisions should bump the `schema` value in the sidecar.

## Japan data next steps
1. Write a converter from Japan source data → KRF CSV (+ optional YAML sidecar).
2. Adjust `KCOR_CMR` to accept a `.csv` file in KRF (and optional sidecar for metadata like `observationEndDate`).

With this, new datasets only require a converter to KRF.

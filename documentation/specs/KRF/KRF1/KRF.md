# KCOR Record Format (KRF) — Permanent Specification

This document defines the permanent, canonical KRF record format and the associated converter configuration. One row per person; columns are case‑sensitive.

## Canonical columns (KRF CSV)

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `ID` | string/int | optional | Unique person identifier if available (stable within dataset). |
| `YearOfBirth` | int | required | 4‑digit year. When derived from age bands, it encodes the lower bound of the birth‑year band. |
| `DeathDate` | date (YYYY‑MM‑DD) | optional | Date of death; empty if alive by `observationEndDate`/`CensorDate`. |
| `Sex` | enum | optional | `M`, `F`, `O`; may be empty. |
| `DCCI` | int | optional | Charlson comorbidity index if known; empty otherwise. |
| `CensorDate` | date (YYYY‑MM‑DD) | optional | Person‑specific end of observation if they exit early. |

Vaccination columns pattern:

- `V1Date`, `V1Brand`, `V2Date`, `V2Brand`, …, `VnDate`, `VnBrand` (n ≥ 1)
- `VxDate` required for the dose to exist; `VxBrand` optional
- Dates strictly increasing per person; same‑day duplicates consolidated
- Brand codes: `P`=Pfizer/BioNTech, `M`=Moderna, `J`=J&J, `O`=Other/Unknown

## CSV and date conventions

- UTF‑8 (no BOM), comma‑separated, header row required
- Double quotes for fields when needed
- Column order arbitrary; headers must match exactly
- Missing values: empty string only (no NA/NULL/\N)
- Dates in ISO `YYYY‑MM‑DD`, no time component. Timezone context documented separately.

## Validation and quality rules

- `YearOfBirth` ∈ [1900, current_year]
- `DeathDate` ≥ `1900‑01‑01` if present
- Dose dates strictly increasing per person
- Each row corresponds to a unique person
- Unknown/extra columns may be present and are ignored by consumers

## Observation window

- Dataset‑level `observationEndDate` caps follow‑up for records without `DeathDate` or `CensorDate`
- Optional per‑person `CensorDate` can end observation earlier

## Minimal KRF CSV example

```csv
ID,YearOfBirth,DeathDate,Sex,DCCI,CensorDate,V1Date,V1Brand,V2Date,V2Brand
234,1957,,F,,,2021-07-28,P,2021-08-18,P
```

---

## Converter configuration (YAML)

This YAML is consumed by the converter to define inputs, outputs, and derivation parameters. This is distinct from the KRF sidecar used by analysis tools.

```yaml
# Converter config YAML
input: data/japan/hamamatsu.csv.gz          # input CSV/CSV.GZ
output: data/japan/japan_krf.csv            # output KRF CSV
referenceDate: earliest_vdate               # 'earliest_vdate' or YYYY-MM-DD
startDate: 2021-01-01                       # include doses >= this date (optional)
endDate: 2025-12-31                         # include doses <= this date (optional)
observationEndDate: 2024-12-31              # dataset observation end (optional)
birthBandYears: 5                           # 1, 5, or 10; see notes below
timezone: local                             # 'local', 'UTC', or IANA TZ id
```

### Field notes

- referenceDate
  - `earliest_vdate`: use each person’s earliest vaccination date (after applying start/end filters) to derive `YearOfBirth` from age bands
  - fixed date: use that single date for all persons
- startDate/endDate
  - Only doses with `VxDate` within `[startDate, endDate]` are included and used for referenceDate=earliest_vdate
- birthBandYears
  - If data provides age bands rather than exact birth years, `YearOfBirth` encodes the lower bound of the band
  - Example: age `60–64` at reference year 2021 → high=64 → `YearOfBirth = 2021 − 64 = 1957`
  - This metadata simply documents band width used to interpret `YearOfBirth` values (1=exact year; 5 or 10=lower bound of that band)
- 100+ rule
  - Open‑ended `100歳～` (100+) is treated as age 100 for derivation (i.e., high=100)

## YAML schemas (optional validation)

Sidecar (analysis metadata) schema appears in `documentation/KRF data format standard.md`.

Converter config schema (JSON Schema in YAML form):

```yaml
$schema: "https://json-schema.org/draft/2020-12/schema"
title: "KRF Converter Config Schema"
type: object
additionalProperties: false
properties:
  input: { type: string }
  output: { type: string }
  referenceDate:
    anyOf:
      - { type: string, const: "earliest_vdate" }
      - { type: string, format: date }
  startDate: { type: ["string", "null"], format: "date" }
  endDate: { type: ["string", "null"], format: "date" }
  observationEndDate: { type: ["string", "null"], format: "date" }
  birthBandYears: { type: integer, enum: [1, 5, 10] }
  timezone: { type: string }
required: [input, output, referenceDate]
```



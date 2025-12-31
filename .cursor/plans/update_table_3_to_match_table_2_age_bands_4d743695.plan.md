---
name: Update Table 3 to match Table 2 age bands
overview: Update Table 3 to use the same age bands as Table 2 (40–49, 50–59, 60–69, 70–79, 80–89, 90–99, All ages) and recompute cumulative hazard values from KCOR_CMR.xlsx for Czech cohorts enrolled in 2021_24.
todos: []
---

# Plan: Update Table 3 to Match Table 2 Age Bands

## Problem

- Table 2 has age bands: 40–49, 50–59, 60–69, 70–79, 80–89, 90–99, All ages (full population)
- Table 3 currently has: 40–49, 50–59, 60–69, 70+ (incorrect aggregation)
- Table 3 needs to match Table 2's exact age bands and recompute cumulative hazard values

## Steps

### 1. Identify source data file

- Locate the correct `KCOR_CMR.xlsx` file for Czech data (likely `data/Czech/KCOR_CMR.xlsx`)
- Verify it contains sheet "2021_24" with enrollment data

### 2. Create script to extract cumulative hazards

- Write a Python script (`documentation/preprint/extract_table3_values.py`) that:
- Reads `KCOR_CMR.xlsx`, sheet "2021_24"
- Maps YearOfBirth to age bands matching Table 2 (using 2020 as reference year):
- 40–49: YOB 1971-1980 (ages 40-49 in 2020)
- 50–59: YOB 1961-1970
- 60–69: YOB 1951-1960
- 70–79: YOB 1941-1950
- 80–89: YOB 1931-1940
- 90–99: YOB 1921-1930
- All ages: YOB = -2 (or aggregate all YOB)
- Filters to Dose 0 and Dose 2
- Computes raw cumulative hazards:
- MR = Dead / Alive (at start of week)
- hazard = -ln(1 - MR) (clipped to avoid log(0))
- CH = cumulative sum of hazards (starting after DYNAMIC_HVE_SKIP_WEEKS)
- Extracts final cumulative hazard at week 2024-16 (end of follow-up)
- Computes Ratio = CH_Dose0 / CH_Dose2

### 3. Verify computation matches Table 2

- Check that rows matching Table 2 (40–49, 50–59, 60–69) produce same values
- If discrepancies found, investigate:
- Age band mapping (YOB ranges)
- Time window (week 2024-16 cutoff)
- Hazard computation method

### 4. Update Table 3 in paper.md

- Replace Table 3 content (lines 632-636) with:
- Same age bands as Table 2
- Newly computed cumulative hazard values
- Format: `| Age band (years) | Dose 0 cumulative hazard | Dose 2 cumulative hazard | Ratio |`
- Ensure formatting matches Table 2 style

### 5. Verify consistency

- Ensure Table 3 notes still apply (week 2024-16, 2021_24 enrollment)
- Check that ratios are computed correctly
- Verify age band labels match Table 2 exactly

## Files to modify

- `documentation/preprint/paper.md` - Update Table 3 (lines 632-636)
- Create `documentation/preprint/extract_table3_values.py` - Script to compute values

## Dependencies

- pandas, openpyxl (for reading Excel)
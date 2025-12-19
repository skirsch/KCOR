---
name: Update SA mode for new KCOR algorithm
overview: Update sensitivity analysis mode to compute KCOR for age=-2 cohort only, varying SLOPE8_QUANTILE_TAU (0.1-0.5) and KCOR_NORMALIZATION_WEEKS (2-8), outputting grid tables (tau columns, weeks rows) per cohort in Excel sheets.
todos:
  - id: modify_sa_mode
    content: Replace SA mode output logic in process_workbook() to iterate over tau and normalization weeks, computing KCOR for age=-2 only
    status: completed
  - id: create_grid_function
    content: Create create_sa_grid_output() function to organize results into grid format and write Excel sheets
    status: completed
    dependencies:
      - modify_sa_mode
  - id: filter_age_minus2
    content: Ensure SA mode filters to YearOfBirth=-2 only throughout the computation pipeline
    status: completed
    dependencies:
      - modify_sa_mode
  - id: extract_final_kcor
    content: Extract final KCOR values at reporting date for each parameter combination and store in results structure
    status: completed
    dependencies:
      - modify_sa_mode
  - id: update_makefile
    content: Update test/sensitivity/Makefile to remove obsolete SA environment variables and simplify
    status: completed
---

# Update Sensitivity Analysis Mode for New KCOR Algorithm

## Overview

Modify the sensitivity analysis (SA) mode in `code/KCOR.py` to:

1. Compute KCOR only for the age=-2 cohort (all ages aggregated)

2. Vary `SLOPE8_QUANTILE_TAU` (0.1, 0.2, 0.3, 0.4, 0.5) and `KCOR_NORMALIZATION_WEEKS` (2, 3, 4, 5, 6, 7, 8)
3. Output Excel sheets with grid tables (tau as columns, normalization weeks as rows) containing final KCOR values

## Implementation Plan

### 1. Modify SA Mode Logic in `process_workbook()` Function

**Location**: `code/KCOR.py` around line 5144-5210**Changes**:

- Replace the current SA output logic (lines 5145-5210) with a new implementation

- Create nested loops over tau values [0.1, 0.2, 0.3, 0.4, 0.5] and normalization weeks [2, 3, 4, 5, 6, 7, 8]
- For each parameter combination:

- Temporarily override `SLOPE8_QUANTILE_TAU` and `KCOR_NORMALIZATION_WEEKS`

- Recompute `KCOR_NORMALIZATION_WEEKS_EFFECTIVE` based on the new normalization weeks

- Run KCOR computation for all cohorts, filtering to YearOfBirth=-2 only
- Extract KCOR values at reporting date (from `KCOR_REPORTING_DATE` dict) for each cohort and dose pair
- Store results in a grid structure

### 2. Create Grid Output Function

**New Function**: `create_sa_grid_output()`

**Purpose**: Organize KCOR results into grid format and write Excel sheets**Parameters**:

- `sa_results`: Dictionary mapping (cohort, dose_num, dose_den, tau, norm_weeks) -> KCOR value
- `out_path`: Output file path

- `dual_print`: Logging function

**Logic**:

- Group results by cohort (enrollment date)
- For each cohort:

- Create a DataFrame with normalization weeks as index (rows) and tau values as columns

- Fill grid with KCOR values

- Write to Excel sheet named after the cohort (e.g., "2021_24")

### 3. Filter to Age=-2 Only

**Location**: Multiple places in `process_workbook()`**Changes**:

- In SA mode, filter `OVERRIDE_YOBS` to only include [-2] (all ages cohort)
- Ensure `build_kcor_rows()` processes YearOfBirth=-2 cohort
- Filter combined results to YearOfBirth=-2 before extracting final KCOR values

### 4. Extract Final KCOR at Reporting Date

**Location**: Within SA mode loop in `process_workbook()`

**Logic**:

- After each parameter combination computation:
- Filter `combined` DataFrame to YearOfBirth=-2

- For each cohort and dose pair:

    - Get reporting date from `KCOR_REPORTING_DATE.get(sheet_name)` or use max date

    - Find closest date to reporting date in the data
    - Extract KCOR value at that date

    - Store in results dictionary: `results[(cohort, dose_num, dose_den, tau, norm_weeks)] = kcor_value`

### 5. Update Makefile (if needed)

**Location**: `test/sensitivity/Makefile`**Changes**:

- Remove obsolete SA environment variables (SA_SLOPE_START, SA_SLOPE_LENGTH, etc.)
- Keep SA_COHORTS and SA_DOSE_PAIRS for optional filtering

- Simplify the make target since parameter ranges are now hardcoded

### 6. Handle Parameter Overrides

**Location**: `code/KCOR.py` around line 655-685**Changes**:

- In SA mode, the parameter override section should still allow `SA_ANCHOR_WEEKS` to override `KCOR_NORMALIZATION_WEEKS`, but the SA loop will override it anyway

- Consider adding environment variables `SA_TAU_VALUES` and `SA_NORM_WEEKS` for flexibility (optional)

## Key Implementation Details

1. **Parameter Override Strategy**: 

- Store original `SLOPE8_QUANTILE_TAU` and `KCOR_NORMALIZATION_WEEKS` values

- In SA loop, temporarily modify these globals before calling `build_kcor_rows()`

- Restore original values after each iteration (or use a context manager)

2. **Efficiency Consideration**:

- The current approach will recompute KCOR for all cohorts multiple times (5 tau × 7 weeks = 35 iterations)

- This is acceptable for sensitivity analysis but may be slow

- Consider caching intermediate computations if performance becomes an issue

3. **Output Format**:

- Excel workbook: `KCOR_SA.xlsx` in `test/sensitivity/out/`
- One sheet per cohort (e.g., "2021_24", "2022_06")

- Each sheet: grid with normalization weeks as rows, tau values as columns

- First column: normalization week values (2, 3, 4, 5, 6, 7, 8)

- Remaining columns: tau values (0.1, 0.2, 0.3, 0.4, 0.5) with KCOR values

4. **Dose Pairs**:

- Process all dose pairs from `get_dose_pairs(sheet_name)`

- Each dose pair gets its own grid (or combine in a single grid with multi-index)

## Files to Modify

1. `code/KCOR.py`:

- Modify `process_workbook()` function (lines 5144-5210)
- Add new `create_sa_grid_output()` function

- Update parameter override section if needed

2. `test/sensitivity/Makefile`:

- Simplify environment variable handling
- Update comments

## Testing

- Run `make sensitivity` from root directory

- Verify output file `test/sensitivity/out/KCOR_SA.xlsx` contains expected grid sheets

- Check that each sheet has correct dimensions (7 rows × 6 columns including header)

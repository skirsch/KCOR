---
name: Add Monte Carlo Mode to KCOR.py
overview: "Modify KCOR.py to support Monte Carlo mode processing: detect MC mode via environment variable, process KCOR_CMR_MC.xlsx with one sheet per iteration, compute KCOR for each iteration, and summarize values at end of 2022 in the log."
todos: []
---

# Add Monte Carlo Mode Support to KCOR.py

## Overview
Modify `KCOR.py` to detect and process Monte Carlo mode, similar to how `KCOR_CMR.py` handles it. In MC mode, process the `KCOR_CMR_MC.xlsx` file (one sheet per iteration) and compute KCOR for each iteration, then summarize results at end of 2022.

## Changes Required

### 1. Add Monte Carlo Mode Detection
- **File**: `code/KCOR.py`
- **Location**: Near top of file (after imports, around line 209-211)
- **Change**: Add environment variable check similar to `KCOR_CMR.py`:
  ```python
  MONTE_CARLO_MODE = str(os.environ.get('MONTE_CARLO', '')).strip().lower() in ('1', 'true', 'yes')
  ```

### 2. Modify `process_workbook()` Function
- **File**: `code/KCOR.py`
- **Location**: `process_workbook()` function starting at line 3230
- **Changes**:
  - **Input file detection**: When `MONTE_CARLO_MODE` is True, check if input file is `KCOR_CMR_MC.xlsx` (or detect MC mode from filename pattern)
  - **Sheet processing**: In MC mode, process all sheets (numbered "1", "2", "3", etc.) as iterations
  - **Enrollment date**: For MC sheets, treat as enrollment date "2022-06" (or "2022_06")
  - **Dose filtering**: In MC mode, limit to doses 0-3 (max_dose=3) and use dose pairs: `[(1,0), (2,0), (2,1), (3,2), (3,1), (3,0)]`
  - **YearOfBirth**: MC sheets have `YearOfBirth=-2` (all ages aggregated), handle this appropriately
  - **Output structure**: Write one output sheet per iteration (mirror input structure)

### 3. Add MC-Specific Sheet Processing Logic
- **File**: `code/KCOR.py`
- **Location**: Inside `process_workbook()` loop (around line 3338)
- **Change**: Add conditional logic to handle MC sheets:
  ```python
  if MONTE_CARLO_MODE:
      # Treat sheet name as iteration number
      # Set enrollment_date to 2022-06
      # Process with max_dose=3
      # YearOfBirth will be -2 (all ages)
  ```

### 4. Add End-of-2022 Summary Function
- **File**: `code/KCOR.py`
- **Location**: After `create_summary_file()` function (around line 5089)
- **Change**: Create new function `create_mc_summary()` that:
  - Collects KCOR values for all dose pairs at end of 2022 (2022-12-31) from all iterations
  - Computes statistics across iterations (mean, median, percentiles, etc.)
  - Writes summary to log file using `dual_print()`
  - Format: Show KCOR values for each dose pair (e.g., "1 vs 0", "2 vs 1", "3 vs 2") with statistics

### 5. Update Main Processing Flow
- **File**: `code/KCOR.py`
- **Location**: End of `process_workbook()` function (around line 4926)
- **Change**: After processing all sheets, if in MC mode:
  - Call `create_mc_summary()` with collected results
  - Write summary to log file

### 6. Handle MC Input File Path
- **File**: `code/KCOR.py`
- **Location**: `main()` function (line 5089)
- **Change**: If `MONTE_CARLO_MODE` is True and input file is not explicitly MC file, optionally auto-detect or require explicit MC file path

### 7. Update Makefile for Monte Carlo Pipeline
- **File**: `code/Makefile`
- **Location**: Around line 90-104 (monte_carlo target)
- **Changes**:
  - Modify `monte_carlo` target to be a phony target that depends on both CMR_MC and KCOR_MC outputs
  - Add new target `KCOR_MC_OUTPUT` (e.g., `../data/$(DATASET)/KCOR_MC.xlsx`) that:
    - Depends on `$(MC_OUTPUT)` (KCOR_CMR_MC.xlsx)
    - Calls `KCOR.py` with `MONTE_CARLO=1` environment variable
    - Uses `$(MC_OUTPUT)` as input and `KCOR_MC_OUTPUT` as output
  - Update `monte_carlo` target to build both CMR_MC and KCOR_MC:
    ```makefile
    monte_carlo: $(KCOR_MC_OUTPUT)
    
    $(KCOR_MC_OUTPUT): $(MC_OUTPUT) KCOR.py | ../data/$(DATASET)
        @echo "Running KCOR analysis on Monte Carlo CMR data..."
        MONTE_CARLO=1 $(PYTHON) KCOR.py $(MC_OUTPUT) $(KCOR_MC_OUTPUT) "Monte Carlo Analysis" KCOR_MC_summary.log
        @echo "Monte Carlo KCOR analysis complete!"
    ```
  - Ensure `$(MC_OUTPUT)` target builds CMR_MC first if needed (already implemented)

### 8. Update Version Number
- **File**: `code/KCOR.py`
- **Location**: Line 498 (VERSION constant)
- **Change**: Update version from `v5.3` to `v5.4`

### 9. Update README.md
- **File**: `README.md`
- **Location**: 
  - Line 1: Update header from `v5.3` to `v5.4`
  - Version History section (around line 1158): Add new version entry
- **Changes**:
  - Add new version 5.4 entry describing Monte Carlo mode support
  - Include description of:
    - Monte Carlo bootstrap sampling support
    - Processing of MC iterations
    - End-of-2022 summary statistics
    - Usage instructions for `make monte_carlo`

## Key Implementation Details

1. **Sheet Name Handling**: MC sheets are numbered "1", "2", "3", etc. Treat these as iteration numbers, not enrollment dates.

2. **Enrollment Date**: All MC iterations use enrollment date "2022-06" (from `KCOR_CMR.py` line 249-250).

3. **Dose Pairs**: Use `get_dose_pairs("2022_06")` which returns `[(1,0), (2,0), (2,1), (3,2), (3,1), (3,0)]`.

4. **YearOfBirth**: MC data has `YearOfBirth=-2` (all ages). Ensure `build_kcor_rows()` handles this correctly (it should, as it processes unique YearOfBirth values).

5. **End of 2022**: Use date `2022-12-31` (matching `KCOR_REPORTING_DATE['2022_06']`).

6. **Summary Format**: Include for each dose pair:
   - Mean KCOR across iterations
   - Median KCOR
   - Percentiles (e.g., 2.5th, 97.5th for 95% CI equivalent)
   - Min/Max values
   - Standard deviation

7. **Makefile Integration**: The `make monte_carlo` command should:
   - First build `KCOR_CMR_MC.xlsx` if it doesn't exist (via existing `$(MC_OUTPUT)` target)
   - Then run `KCOR.py` in Monte Carlo mode to process the MC file
   - Produce `KCOR_MC.xlsx` output with one sheet per iteration
   - Generate summary log with end-of-2022 statistics

## Testing Considerations

- Verify MC mode detection works via environment variable
- Ensure MC input file is processed correctly
- Check that output has one sheet per iteration
- Validate summary statistics are computed correctly
- Confirm summary is written to log file
- Test Makefile `monte_carlo` target builds both CMR_MC and KCOR_MC outputs
- Verify version number is updated in both KCOR.py and README.md

## Files to Modify

- `code/KCOR.py`: Main changes to add MC mode support
- `code/Makefile`: Add KCOR_MC target and update monte_carlo target
- `README.md`: Update version number and add version 5.4 entry


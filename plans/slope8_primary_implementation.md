# Switch to Slope8 Normalization with Abnormal Fit Flagging

## Plan

### Overview
Replace the current Slope6/Slope7 normalization logic with Slope8 for all cohorts. Track abnormal fits via optimizer diagnostics and flag affected KCOR results with asterisks in output.

### Changes Required

#### 1. Modify `compute_slope6_normalization` function (lines ~1610-2264)
   - **Remove Slope6/Slope7 decision logic**: Currently decides between linear (slope6) and depletion-mode (slope7) based on `b_lin >= 0`
   - **Use Slope8 for all cohorts**: Always fit slope8 and use its results for normalization
   - **Store abnormal fit flag**: Add `"abnormal_fit": bool` to params dict based on `diagnostics["optimizer_status"] == 5` or `not diagnostics["success"]`
   - **Apply from s=0**: Ensure normalization uses slope8 parameters applied from s=0 (enrollment) for all cohorts, even though fit window starts later for highest dose
   - **Keep logging**: Continue logging linear, slope7 TRF, slope7 LM, and slope8 fits for diagnostics

#### 2. Modify `apply_slope6_norm` function (lines ~3561-3594)
   - **Add slope8 mode handling**: Add `elif mode == "slope8"` branch with same formula as slope7:
     ```python
     h_norm = h * exp(-C - kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))
     ```
     where s = t (time since enrollment, NOT centered)

#### 3. Update `get_slope` and `get_scale_factor` helpers (lines ~3603-3647)
   - **Handle slope8 mode**: Return appropriate values for slope8 (similar to slope7 handling)
   - **For slope8**: Return `b_original` (from initial linear fit) for slope, compute scale_factor using slope8 parameters

#### 4. Track abnormal fits through KCOR computation
   - **In `build_kcor_rows`** (lines ~2299-2849):
     - Add logic to check if either numerator or denominator cohort had abnormal fit
     - Add `"abnormal_fit"` column to output DataFrame indicating if KCOR is affected
   - **In `process_workbook`** (lines ~3514-3532):
     - Pass abnormal fit information from `slope6_params_map` to `build_kcor_rows` if needed

#### 5. Flag KCOR results in console/log output (lines ~3895-4096)
   - **In reporting section**: Check `abnormal_fit` flag for each KCOR value
   - **Add asterisk**: Append `"*"` to KCOR value string when either numerator or denominator had abnormal fit
   - **Example**: Change `{kcor_val:8.4f}` to `f"{kcor_val:8.4f}*"` when flagged

#### 6. Flag KCOR results in summary spreadsheet (lines ~4675-4692)
   - **In `create_summary_file`**: Check for abnormal fit flag
   - **Add asterisk**: Append `"*"` to KCOR string in summary rows when flagged
   - **Example**: Change `f"{row['KCOR']:.4f}"` to `f"{row['KCOR']:.4f}*"` when flagged

#### 7. Update version and documentation
   - **Version**: Update `VERSION` constant to "v5.3" (line ~494)
   - **Version history**: Add entry documenting Slope8 as primary method
   - **Docstring**: Update `compute_slope6_normalization` docstring to reflect Slope8 usage

### Implementation Details

#### Abnormal Fit Detection
- Abnormal fit = `diagnostics["optimizer_status"] == 5` OR `not diagnostics["success"]`
- Store in params dict as: `"abnormal_fit": True/False`
- KCOR is flagged if EITHER numerator OR denominator cohort had abnormal fit

#### Slope8 Application
- Fit window: For highest dose, starts at `SLOPE_FIT_DELAY_WEEKS` (default 15 weeks)
- Application window: Always starts at s=0 (enrollment) for all cohorts
- Formula: Same depletion-mode formula as slope7, using slope8 fitted parameters

#### Backward Compatibility
- Keep `slope6_params_map` name (even though using slope8) to avoid breaking existing code
- Keep existing column names and structure in output DataFrames
- Add new `abnormal_fit` column to track flags

### Files to Modify
- `code/KCOR.py`: Main implementation file (~4750 lines)

### Testing Considerations
- Verify slope8 fits are used for normalization
- Verify abnormal fits are correctly detected and flagged
- Verify asterisks appear in both console/log and summary spreadsheet
- Verify highest dose uses delayed fit window but applies from s=0

---

## Implementation Summary

### Completed Changes

1. **Version updated** to v5.3 with version history entry documenting Slope8 as primary method

2. **Slope8 as primary method**: Modified `compute_slope6_normalization` to use Slope8 for all cohorts instead of the Slope6/Slope7 decision logic. The function now:
   - Attempts Slope8 fit for all cohorts
   - Falls back to linear mode only if Slope8 fails or has insufficient data
   - Stores results with abnormal fit tracking

3. **Abnormal fit tracking**: Added `abnormal_fit` flag to params dict based on optimizer diagnostics (status=5 or not success). This flag is:
   - Stored in normalization parameters for each cohort
   - Propagated through to KCOR computation
   - Used to flag affected KCOR results

4. **Normalization application**: Updated `apply_slope6_norm` and helper functions (`get_slope`, `get_scale_factor`) to handle slope8 mode (same formula as slope7)

5. **KCOR output tracking**: Added `abnormal_fit` column to all KCOR output rows:
   - Per-age KCOR rows
   - ASMR pooled rows (YearOfBirth=0)
   - All-ages rows (YearOfBirth=-2)

6. **Asterisk flagging**: 
   - **Console/log output**: KCOR values are flagged with `*` when affected by abnormal fit in the reporting section
   - **Summary spreadsheet**: KCOR values are flagged with `*` in the summary spreadsheet output

7. **Documentation**: Updated docstring for `compute_slope6_normalization` to reflect Slope8 usage and abnormal fit tracking

### Key Implementation Details

- **Slope8 fit window**: For highest dose, fit uses data from `s >= SLOPE_FIT_DELAY_WEEKS` (default 15 weeks), but normalization is applied from `s=0` (enrollment) for all cohorts
- **Abnormal fit detection**: Based on `optimizer_status == 5` (abnormal termination) or `not success` from L-BFGS-B optimizer diagnostics
- **Flagging logic**: KCOR is flagged if EITHER numerator OR denominator cohort had abnormal fit
- **Backward compatibility**: Maintained existing column names and structure, added new `abnormal_fit` column

### Files Modified
- `code/KCOR.py`: Complete implementation (~4711 lines)

### Status
✅ All changes implemented
✅ No linter errors
✅ Ready for testing

The implementation ensures that:
- Slope8 fit window starts later for highest dose (SLOPE_FIT_DELAY_WEEKS), but normalization is applied from s=0 for all cohorts
- Abnormal fits are detected and flagged in both console/log and spreadsheet outputs
- All existing functionality is preserved with backward compatibility


# Add Slope8 Quantile Regression Method

## Overview
Add slope8 as a fourth diagnostic method alongside linear, slope7_success (TRF), and slope7_lm. Slope8 uses quantile regression with scipy.optimize.minimize instead of least squares, providing robustness to outliers. Results will be logged to the debug CSV but not yet applied for normalization.

## Implementation Steps

### 1. Add Configuration Parameters ([code/KCOR.py](code/KCOR.py))
- **After line 133**: Add `SLOPE8_QUANTILE_TAU = 0.5` (quantile level for quantile regression, default median)
- **After line 133**: Add `SLOPE_FIT_DELAY_WEEKS = 15` (delay in weeks for highest dose fit start)
- These control the quantile level (default median) and the delay for highest dose fits

### 2. Update Imports ([code/KCOR.py](code/KCOR.py))
- **Line 93**: Modify `from scipy.optimize import least_squares` to `from scipy.optimize import least_squares, minimize`
- Add `minimize` to the existing import statement

### 3. Create `fit_slope8_depletion` Function ([code/KCOR.py](code/KCOR.py))
- **After `fit_slope7_depletion_lm` (around line 1000)**: Create new function `fit_slope8_depletion(s, logh)` that:
  - Signature: `fit_slope8_depletion(s, logh)` returning `((C, k_inf, k_0, tau), initial_params)`
  - Implementation based on `documentation/specs/slope8.md`:
    - Use same initial parameter estimation as slope7 functions
    - Define bounds: 
      - C ∈ [-25, 0] (hazards are ~1e-5–1e-3, so C is safely around [-20, -5])
      - k_0 ∈ [-0.1, 0.1] (weekly log-slope is small)
      - Δk ∈ [0, 0.1] (long-run minus initial slope)
      - tau ∈ [1e-3, 260.0] (e.g., 5 years; avoids 600-year degeneracy)
    - Implement quantile_loss function using check loss: 
      - ρ_τ(u) = τ*u if u >= 0, (τ-1)*u if u < 0
      - where u = logh_valid - predicted
    - Use `scipy.optimize.minimize` with method='L-BFGS-B' and bounds
    - Add small ridge term (1e-4 * sum(residuals^2)) for numerical stability
    - Return same format as slope7 functions: ((C, k_inf, k_0, tau), initial_params)
    - Handle exceptions gracefully, returning (np.nan, np.nan, np.nan, np.nan) on failure

### 4. Modify `compute_slope6_normalization` Function ([code/KCOR.py](code/KCOR.py))
- **Location**: around line 1001-1501
- **After building slope7 data (around line 1201-1236)**, add slope8 fitting logic:
  
  **Determine highest dose:**
  - Get dose pairs for the enrollment date using `get_dose_pairs(sheet_name)`
  - Compute `max_dose = max(max(pair) for pair in dose_pairs)`
  
  **Build slope8 deployment window:**
  - For each cohort, determine if it's the highest dose: `is_highest_dose = (dose == max_dose)`
  - Build deployment window data from `enrollment_date` to `SLOPE_FIT_END_ISO`
  - For highest dose: filter to start at `enrollment_date + timedelta(weeks=SLOPE_FIT_DELAY_WEEKS)`
  - For other doses: use full range from `enrollment_date` to `SLOPE_FIT_END_ISO`
  - Create sequential s_values starting from 0 (s=0 at enrollment_date for all doses)
  - For highest dose, s_values will be offset: s=0 corresponds to enrollment_date, but fit starts at week SLOPE_FIT_DELAY_WEEKS
  
  **Fit and log slope8:**
  - After slope7_lm fit (around line 1336), add slope8 fit block
  - Call `fit_slope8_depletion(np.array(s_values), np.array(log_h_slope8_values))`
  - Compute RMS error for diagnostics
  - Log to `log_slope7_fit_debug` with mode="slope8"
  - Include all same fields as slope7 logging (C, ka, kb, tau, initial params, RMS error, etc.)
  - Handle invalid parameters gracefully (log with note)

### 5. Update Debug Logging
- The existing `log_slope7_fit_debug` function already supports arbitrary mode strings
- Ensure slope8 logs include all required fields (s_values, logh_values, h_values, iso_weeks_used, etc.)
- Mode string will be "slope8" to distinguish from other methods

## Key Design Decisions

1. **s=0 at enrollment**: For all doses, s=0 corresponds to enrollment_date. For highest dose, the fit window starts later, but s-values still reference enrollment_date.

2. **Full deployment range**: Slope8 fits the entire deployment window (enrollment to SLOPE_FIT_END_ISO) except for highest dose which starts SLOPE_FIT_DELAY_WEEKS after enrollment.

3. **Diagnostic only**: Slope8 results are logged but not used for normalization. The existing normalization logic remains unchanged.

4. **Consistent interface**: Slope8 function follows same signature and return format as slope7 functions for consistency.

## Files to Modify

- `code/KCOR.py`:
  - Add configuration parameters (lines ~130)
  - Update imports (line 93)
  - Add `fit_slope8_depletion` function (after line ~1000)
  - Modify `compute_slope6_normalization` function (around lines 1201-1336)

## Testing Considerations

- Verify slope8 fits are logged for all cohorts
- Verify highest dose uses delayed fit window
- Verify s=0 corresponds to enrollment_date for all doses
- Verify quantile loss computation matches specification
- Verify bounds prevent pathological tau values (e.g., 32330 weeks)

## Reference
- Specification: [documentation/specs/slope8.md](documentation/specs/slope8.md)


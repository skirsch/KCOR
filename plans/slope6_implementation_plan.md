# Implement Slope6 Normalization for KCOR

## Overview
Replace slope5 with slope6 normalization method that uses time-centered linear/quadratic quantile regression per the specification. The fit window remains the same as slope5 (2022-01 to 2024-12), but normalization application uses enrollment_date to 2024-16 to determine the centerpoint.

## Key Changes

### 1. Version Update
- Update `VERSION` from "v4.9" to "v5.0" in `code/KCOR.py`
- Add version history entry for v5.0

### 2. Dependencies
- Add `cvxpy` import (for quadratic regression with c >= 0 constraint)
- Keep `statsmodels` import (already present, used for linear regression)
- Add fallback handling if cvxpy is not available

### 3. Configuration Parameters
- Rename `SLOPE5_*` constants to `SLOPE6_*`:
  - `SLOPE6_BASELINE_WINDOW_LENGTH_MIN` (keep same value: 30)
  - `SLOPE6_BASELINE_WINDOW_LENGTH_MAX` (keep same value: 60)
  - `SLOPE6_BASELINE_START_YEAR` (keep same value: 2023)
  - `SLOPE6_MIN_DATA_POINTS` (keep same value: 5)
  - `SLOPE6_QUANTILE_TAU` (keep same value: 0.5)
- Add `SLOPE6_APPLICATION_END_ISO = "2024-16"` constant

### 4. Helper Functions (per kcor_slope6_helpers.md)
- Replace `select_slope5_baseline_window()` with `select_slope6_baseline_window()` (same logic, returns 2022-01 to 2024-12)
- Replace `compute_slope5_normalization()` with `compute_slope6_normalization()`:
  - Implement `fit_linear_median()` helper (time-centered, uses statsmodels)
  - Implement `fit_quadratic_quantile()` helper (time-centered, uses cvxpy with c >= 0)
  - Implement decision logic: if b_lin >= 0 use linear, else use quadratic
  - Return parameters dict with mode, a, b, c, t_mean, tau
- Update `_iso_week_list_slope5()` to `_iso_week_list_slope6()` (or keep generic)

### 5. Time Centering Logic
- **Fit window**: 2022-01 to 2024-12 (same as slope5) - used for regression fitting
- **Application window**: enrollment_date to 2024-16 - used to determine centerpoint
- Compute `t_mean = mean(t)` over application window (enrollment_date to 2024-16)
- Use `t_c = t - t_mean` for all regression fits and normalization
- Store `t_mean` per cohort for normalization application

### 6. Normalization Application
- Update normalization application in main processing loop:
  - Linear mode: `h_norm = h * exp(-b_lin * t_c)` where `t_c = t - t_mean`
  - Quadratic mode: `h_norm = h * exp(-(b * t_c + c * t_c^2))` where `t_c = t - t_mean`
- Ensure `t_mean` is computed from application window (enrollment_date to 2024-16)

### 7. Summary Output Updates
- Update summary printout to show:
  - `SLOPE6_FIT_WINDOW = 2022-01 to 2024-12`
  - `SLOPE6_APPLICATION_ENDPOINT = 2024-16` (rightmost point for centerpoint)
- Update all SLOPE5 references in output to SLOPE6

### 8. Documentation Updates
- Update README.md:
  - Change version header from v4.8 to v5.0
  - Update slope normalization section to describe slope6
  - Add version history entry for v5.0
  - Document the time-centering approach and application window
- Update code docstrings and comments

### 9. Error Handling
- Handle cvxpy import failures gracefully
- Handle solver failures in quadratic regression
- Fallback to linear mode if quadratic fails

## Implementation Files
- `code/KCOR.py`: Main implementation changes
- `README.md`: Documentation updates

## Testing Considerations
- Verify time centering produces t_c=0 at centerpoint
- Verify linear mode when b_lin >= 0
- Verify quadratic mode when b_lin < 0
- Verify c >= 0 constraint in quadratic mode
- Verify normalization application uses correct t_mean from application window


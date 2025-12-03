# Implement Slope7 Depletion-Mode Normalization

## Overview
Implement slope7 normalization for cohorts with negative initial slopes (b_lin < 0), replacing the quadratic mode. Slope7 uses Levenberg-Marquardt nonlinear least squares to fit a depletion-shape curve over the full deployment window (enrollment to slope7_end_ISO).

## Changes Required

### 1. Configuration Parameters ([code/KCOR.py](code/KCOR.py))
- **Line 161**: Change `VERSION = "v5.0"` to `VERSION = "v5.1"` (version bump for slope7 method)
- **Line 124**: Rename `ENABLE_QUADRATIC_SLOPE_FIT` to `ENABLE_NEGATIVE_SLOPE_FIT` and keep it set to 1
- **After line 123**: Add `SLOPE7_END_ISO = "2024-16"` constant (same as SLOPE6_APPLICATION_END_ISO)

### 2. Imports ([code/KCOR.py](code/KCOR.py))
- **After line 88**: Add `from scipy.optimize import least_squares` for Levenberg-Marquardt optimization

### 3. New Function: `fit_slope7_depletion` ([code/KCOR.py](code/KCOR.py))
- **After `fit_quadratic_quantile` (around line 600)**: Create new function `fit_slope7_depletion(s, logh)` that:
  - Takes time `s` (weeks since enrollment, NOT centered) and `logh` (log hazard values)
  - Implements the depletion model: `log h(s) = C + k_∞*s + (k_0 - k_∞)*τ*(1 - e^(-s/τ))`
  - Uses `scipy.optimize.least_squares` with Levenberg-Marquardt method
  - Returns `(C, k_∞, k_0, τ)` where:
    - `C` = intercept
    - `k_∞` = long-run background slope (kb, must be > 0)
    - `k_0` = slope at enrollment (ka, may be negative)
    - `τ` = depletion timescale in weeks
  - Handles parameter bounds: `k_∞ > 0`, `τ > 0`
  - Returns `(np.nan, np.nan, np.nan, np.nan)` on failure

### 4. Modify `compute_slope6_normalization` ([code/KCOR.py](code/KCOR.py))
- **Line 741**: Change condition from `ENABLE_QUADRATIC_SLOPE_FIT` to `ENABLE_NEGATIVE_SLOPE_FIT`
- **Lines 762-836**: Replace quadratic mode logic with slope7 mode:
  - When `b_lin < 0` and `ENABLE_NEGATIVE_SLOPE_FIT != 0`:
    - Change fit window: use deployment window (enrollment_date to SLOPE7_END_ISO) instead of baseline_window
    - Use `s = t` (time since enrollment) directly, NOT `t_c = t - t_mean` (no centering)
    - Call `fit_slope7_depletion(s_values, log_h_values)` where `s_values` are weeks since enrollment
    - Store parameters in params dict with:
      - `"mode": "slope7"`
      - `"C"`: intercept
      - `"ka"`: k_0 (starting slope)
      - `"kb"`: k_∞ (final slope)
      - `"tau"`: τ (depletion timescale)
      - `"b_original"`: original b_lin value from linear fit
      - `"t_mean"`: 0.0 (no centering for slope7)
  - Update logging to print slope7 parameters: `SLOPE7_FIT,EnrollmentDate=...,YoB=...,Dose=...,mode=slope7,C=...,ka=...,kb=...,tau=...,b_original=...,rms_error=...,points=...`

### 5. Update Normalization Application ([code/KCOR.py](code/KCOR.py))
- **Lines 2108-2137**: Modify `apply_slope6_norm` function to handle slope7 mode:
  - Add `elif mode == "slope7"` branch
  - Extract `C`, `ka`, `kb`, `tau` from params
  - Use `s = row["t"]` (time since enrollment, NOT centered)
  - Apply formula: `h_norm = h * exp(-C - kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))`
- **Lines 2155-2179**: Update `get_scale_factor` function to handle slope7 mode:
  - Add `elif mode == "slope7"` branch
  - Compute scale factor: `exp(-C - kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))`

### 6. Update Summary Printing ([code/KCOR.py](code/KCOR.py))
- **Lines 2504-2555**: Modify parameter extraction and printing logic:
  - Extract slope7 parameters: `C`, `ka`, `kb`, `tau`, `b_original` from params dict
  - For slope7 mode, print: `b_original=..., C=..., ka=..., kb=..., tau=...`
  - Keep existing logic for linear/quadratic modes
  - Update param_parts building to include slope7 parameters when mode is "slope7"

### 7. Update Version History ([code/KCOR.py](code/KCOR.py))
- **After line 214**: Add version history entry for v5.1:
  ```
  v5.1 - Replaced quadratic mode with Slope7 depletion-mode normalization for b < 0 cohorts
         - Uses Levenberg-Marquardt nonlinear least squares to fit exponential relaxation depletion curve
         - Fit window = deployment window (enrollment to slope7_end_ISO)
         - Time axis s = weeks since enrollment (no centering)
         - Parameters: C, ka (k_0), kb (k_∞), tau (τ)
         - Provides robust handling of depletion-driven curvature while preserving frailty model constraints
  ```

### 8. Update Documentation/Comments ([code/KCOR.py](code/KCOR.py))
- **Line 110-115**: Update Slope6 method comment to mention slope7 for b < 0 cases
- **Line 601-633**: Update docstring for `compute_slope6_normalization` to document slope7 mode
- **Line 2102-2105**: Update comment to mention slope7 mode

## Implementation Notes
- Slope7 uses the deployment window (enrollment to slope7_end_ISO) as both fit and application window
- Time axis `s` is weeks since enrollment (no centering)
- Parameters are stored with keys: `C`, `ka`, `kb`, `tau`, `b_original`
- Summary output shows original `b` value plus fitted `C`, `ka`, `kb`, `tau` values
- Version bumped to v5.1 to reflect slope7 implementation

## Reference
- Specification: [documentation/specs/kcor_slope7.md](documentation/specs/kcor_slope7.md)


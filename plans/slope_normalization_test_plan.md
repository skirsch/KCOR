# Create Slope Normalization Test Script

## Overview
Create `test/slope_normalization/test.py` that extracts and applies three slope normalization methods from `code/KCOR.py` to `booster_d0_slope.csv`, outputting results to an Excel file. Also add `slope-test` target to main Makefile.

## Implementation Steps

### 1. Extract normalization functions from KCOR.py
Copy the following functions and dependencies to `test.py`:
- `fit_linear_median()` (lines 609-642) - Linear median quantile regression
- `fit_slope7_depletion()` (lines 706-823) - TRF bounded depletion-mode normalization  
- `fit_slope7_depletion_lm()` (lines 826-891) - LM unbounded depletion-mode normalization
- Constants: `EPS = 1e-12`, `SLOPE6_MIN_DATA_POINTS = 5`, `SLOPE6_QUANTILE_TAU = 0.5`
- Required imports: `numpy`, `pandas`, `statsmodels.api`, `scipy.optimize.least_squares`

### 2. Read and prepare input data
- Read `booster_d0_slope.csv` from the same directory
- Parse the `date` column (ISO week format: 2022-06, 2022-07, etc.)
- Create sequential time index `t` starting from 0 (0, 1, 2, ...)
- Extract `h(t) raw` column as `hazard_raw`
- Keep existing `h(t) adj` column for comparison

### 3. Apply all three normalization methods
For each method:
- **Linear (Method 1)**: 
  - Fit using `fit_linear_median(t, log(hazard_raw))`
  - Apply normalization: `h_adj_linear = hazard_raw * exp(-b_lin * (t - t_mean))`
  
- **TRF Depletion (Method 2)**:
  - Fit using `fit_slope7_depletion(t, log(hazard_raw))`
  - **Process**: Our code (KCOR.py) computes initial parameter estimates using heuristics, then passes them to `scipy.optimize.least_squares()` for numerical optimization
  - **Initial estimates** (computed in OUR code, lines 744-773, used as starting guess `p0`):
    - C_init: mean of first 3 log(hazard) values
    - k_0_init: slope from first two points  
    - delta_k_init: max(later_slope - k_0_init, 0.001)
    - tau_init: (max(t) - min(t)) / 3.0
  - **Note**: Poor initial guesses can affect convergence (local vs global minima, convergence failures). The heuristics are data-driven but still guesses.
  - **Bounds**: Δk ≥ 0, tau > 1e-3 (C and k_0 unbounded) - enforced by OUR code before calling scipy
  - **Fitting parameters**: max_nfev=1000, ftol=1e-8, xtol=1e-8 (passed to scipy)
  - Apply normalization: `h_adj_trf = hazard_raw * exp(-C - kb*t - (ka - kb)*tau*(1 - exp(-t/tau)))`
  
- **LM Depletion (Method 3)**:
  - Fit using `fit_slope7_depletion_lm(t, log(hazard_raw))`
  - **Process**: Our code computes initial parameter estimates, then passes them to scipy's optimizer
  - **Initial estimates**: Same heuristics as TRF (computed in OUR code)
  - **Note**: Same caveat about initial guess quality affecting results
  - **No bounds** (unbounded Levenberg-Marquardt)
  - **Fitting parameters**: max_nfev=1000, ftol=1e-8, xtol=1e-8 (passed to scipy)
  - Apply normalization: `h_adj_lm = hazard_raw * exp(-C - kb*t - (ka - kb)*tau*(1 - exp(-t/tau)))`

### 4. Create Excel output
Write to `test/slope_normalization/slope_normalization_test_output.xlsx` with two sheets:

**Sheet 1: "data"**
- Columns: `date`, `h(t) raw`, `h(t) adj` (existing), `h_adj_linear`, `h_adj_trf`, `h_adj_lm`
- All rows from input CSV

**Sheet 2: "parameters"**
- Columns: `method`, `parameter`, `value`, `description`
- Rows for each method's fitted parameters:
  - Linear: `a_lin`, `b_lin`, `t_mean`
  - TRF: `C`, `k_inf` (kb), `k_0` (ka), `tau`
  - LM: `C`, `k_inf` (kb), `k_0` (ka), `tau`
- Also include fitting configuration rows:
  - TRF: bounds info (Δk ≥ 0, tau > 1e-3)
  - Both nonlinear: max_nfev=1000, ftol=1e-8, xtol=1e-8
  - Note that initial parameter estimates are computed in OUR code (not scipy) using heuristics
  - Include initial estimates in parameters sheet for comparison with final fitted values

### 5. Error handling
- Handle cases where fits fail (return NaN parameters)
- Use try/except blocks around each fit
- Log warnings if fits don't converge
- Print initial parameter estimates used for nonlinear fits to console for debugging
- Check `result.success` from scipy and warn if optimization failed

### 6. Add Makefile target
- Add `slope-test` to `.PHONY` declaration in `Makefile` (line 13)
- Add target: `slope-test: $(VENV_DIR)` followed by `cd test/slope_normalization && $(abspath $(VENV_PYTHON)) test.py`
- Add help text in `help` target (around line 166): `@echo "  slope-test     - Run slope normalization test on booster_d0_slope.csv"`

## Files to modify
- Create: `test/slope_normalization/test.py`
- Modify: `Makefile` - Add `slope-test` target and help text

## Dependencies
- numpy
- pandas  
- statsmodels
- scipy
- openpyxl (for Excel writing)


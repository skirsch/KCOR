# Improve Slope8 Initial Guess Algorithm

## Overview
Replace the fragile single-point finite difference initial guess logic in `fit_slope8_depletion` with a robust windowed OLS approach. This will dramatically reduce invalid fits and prevent pathological tau values by using small OLS fits over early and late windows instead of single finite differences.

## Implementation Steps

### 1. Create Helper Function `make_slope8_initial_guess` ([code/KCOR.py](code/KCOR.py))
- **After `format_initial_params` (around line 171)**: Create new helper function `make_slope8_initial_guess(s_valid, logh_valid)` that:
  - Takes sorted arrays `s_valid` and `logh_valid` (time and log-hazard values)
  - Returns tuple `(C_init, k_0_init, delta_k_init, tau_init)`
  - Implementation based on `documentation/specs/slope/slope8_initial_guess.md`:
    - **Step 0**: Sort data by `s_valid` (ensure ordering)
    - **Step 1**: Compute global linear fit as baseline/fallback using `np.linalg.lstsq`
    - **Step 2**: Compute early and late window OLS fits:
      - Window size: `w = max(n // 3, 3)` (at least 3 points, ~1/3 of data)
      - Early window: first `w` points → `b0, a0` (slope and intercept)
      - Late window: last `w` points → `b_inf, a_inf` (slope and intercept)
    - **Step 3**: Derive initial guesses:
      - `k_0_init = b0` (early slope)
      - `k_inf_init = b_inf` (late slope)
      - Pattern check: if `k_inf_init < k_0_init`, collapse to global slope (non-depletion case)
      - `delta_k_init = max(k_inf_init - k_0_init, 1e-4)` (ensure ≥ 0)
    - **Step 4**: Compute tau:
      - `span = max(s_valid) - min(s_valid)` (with fallback to 10.0 if span ≤ 0)
      - `tau_init = span / 3.0` clamped to `[2.0, 52.0]` weeks
    - **Step 5**: Compute C from early intercept:
      - `C_init = a0` (intercept from early window fit, evaluated at s=0)
    - Handle edge case: if `n < 5`, return almost-linear fallback:
      - `C_init = a_global`, `k_0_init = b_global`, `delta_k_init = 1e-4`, `tau_init = 10.0`
  - Use `np.linalg.lstsq` with `rcond=None` for OLS fits
  - Return all values as `float` for consistency

### 2. Update `fit_slope8_depletion` Function ([code/KCOR.py](code/KCOR.py))
- **Location**: Lines 1038-1190
- **Replace initial parameter estimation block (lines 1078-1105)** with call to helper:
  - After filtering invalid values (line 1076), ensure data is sorted:
    - Add: `order = np.argsort(s_valid)`; `s_valid = s_valid[order]`; `logh_valid = logh_valid[order]`
  - Replace lines 1078-1105 with:
    ```python
    # Use improved initial guess algorithm
    C_init, k_0_init, delta_k_init, tau_init = make_slope8_initial_guess(s_valid, logh_valid)
    ```
  - Keep existing bounds and clipping logic (lines 1113-1128) - these remain unchanged
  - Keep existing return format and error handling

### 3. Update Comments/Documentation ([code/KCOR.py](code/KCOR.py))
- **Line 1038**: Update docstring for `fit_slope8_depletion` to mention improved initial guess algorithm
- **After helper function**: Add docstring explaining the windowed OLS approach and rationale

## Key Design Decisions

1. **Windowed OLS over finite differences**: Uses OLS fits over early/late windows (~1/3 of data) instead of single-point differences, dramatically reducing sensitivity to noise.

2. **Pattern enforcement**: Checks for depletion pattern (k_∞ ≥ k_0) and gracefully falls back to global linear slope if pattern is violated (non-depletion case).

3. **Reasonable tau bounds**: Clamps initial tau guess to [2, 52] weeks to prevent pathological values (e.g., 30,000+ weeks) while still allowing optimizer to explore full [MIN_TAU, MAX_TAU] range.

4. **Consistent C estimate**: Uses intercept from early window fit (evaluated at s=0) instead of arbitrary mean of first 3 points, ensuring consistency with k_0 estimate.

5. **Backward compatibility**: Maintains same function signature and return format; only improves internal initial guess logic.

## Files to Modify

- `code/KCOR.py`:
  - Add `make_slope8_initial_guess` helper function (after line ~171)
  - Update `fit_slope8_depletion` function (lines 1078-1105)
  - Update docstrings and comments

## Testing Considerations

- Verify improved initial guesses reduce invalid fit rate (check debug CSV for fewer NaN results)
- Verify tau initial guesses are reasonable (between 2-52 weeks, not extreme values)
- Verify depletion pattern enforcement works (non-depletion cases fall back to global slope)
- Verify edge cases handled (n < 5, span ≤ 0, etc.)
- Verify sorting ensures proper ordering before OLS fits
- Compare fit success rates before/after improvement

## Expected Benefits

- **Dramatically reduce invalid fits**: Windowed OLS is much more robust to noise than single-point differences
- **Avoid insane tau values**: Clamping prevents pathological initial guesses that lead to optimizer failures
- **Respect depletion pattern**: Pattern checking ensures initial guesses match expected depletion behavior
- **Graceful degradation**: Non-depletion cases automatically fall back to near-linear initialization

## Reference

- Specification: [documentation/specs/slope/slope8_initial_guess.md](documentation/specs/slope/slope8_initial_guess.md)


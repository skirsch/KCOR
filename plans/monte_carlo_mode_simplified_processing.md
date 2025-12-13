# Monte Carlo Mode Simplified Processing

## Problem Analysis

Current issues in MC mode:
1. **YOB=-10 appears**: AGE_RANGE bucketing (line 3442-3443) converts YOB=-2 to -10: `(-2 // 10) * 10 = -10`
2. **Slope8 attempted**: YOB=-10 < 1940, so code tries slope8 instead of linear
3. **Wrong cohorts processed**: Code processes YOB=-10 instead of only YOB=-2

## Solution: Separate MC Processing Path

Create a dedicated MC processing branch in `process_workbook()` that:
- Skips AGE_RANGE bucketing for MC mode
- Filters to only YOB=-2
- Forces linear fits only
- Uses shared functions with MC-specific control flow

## Implementation Steps

### 1. Fix AGE_RANGE Bucketing (Immediate Fix)
**File**: `code/KCOR.py` (around line 3441)
- Skip AGE_RANGE bucketing when `MONTE_CARLO_MODE` is True
- Ensure YOB=-2 remains -2

### 2. Add Force Linear Flag to compute_slope6_normalization
**File**: `code/KCOR.py` (function starting at line 1560)
- Add optional parameter `force_linear_mode=False`
- When `force_linear_mode=True`, skip slope8 attempts and use linear fits only
- Modify logic at line 1897 to respect this flag: `use_slope8 = (yob < SLOPE8_MAX_YOB and yob != -2) and not force_linear_mode`

### 3. Create MC-Specific Processing Branch
**File**: `code/KCOR.py` (in `process_workbook()` around line 3356)
- After reading sheet data, check `if MONTE_CARLO_MODE:`
- In MC branch:
  - Ensure YOB=-2 (add if missing, filter to only -2)
  - Skip AGE_RANGE bucketing
  - Skip SA mode (not applicable to MC)
  - Call `compute_slope6_normalization()` with `force_linear_mode=True`
  - Process with simplified flow (no dynamic slopes, direct normalization)
  - Call shared functions: `build_kcor_rows()`, `build_kcor_ns_rows()`, `build_kcor_o_deaths_details()`
- Keep existing non-MC path unchanged

### 4. Filter YOB in MC Mode
**File**: `code/KCOR.py` (around line 3363)
- After adding YOB=-2 if missing, filter dataframe: `df = df[df["YearOfBirth"] == -2]`
- Add debug message confirming only YOB=-2 is present

### 5. Update Normalization Application
**File**: `code/KCOR.py` (around line 3683)
- In MC mode, ensure normalization parameters are linear mode only
- Verify no slope8 parameters are stored for MC iterations

## Key Changes Summary

1. **Line ~3441**: Skip AGE_RANGE bucketing in MC mode
2. **Line ~1560**: Add `force_linear_mode` parameter to `compute_slope6_normalization()`
3. **Line ~1897**: Respect `force_linear_mode` flag
4. **Line ~3363**: Filter to YOB=-2 only in MC mode
5. **Line ~3356**: Create MC-specific processing branch with simplified flow

## Testing Considerations

- Verify MC mode only processes YOB=-2
- Verify all fits are linear (no slope8 attempts)
- Verify output matches expected format
- Ensure regular mode is unaffected

## Implementation Todos

- [ ] Skip AGE_RANGE bucketing when MONTE_CARLO_MODE is True (around line 3441)
- [ ] Add force_linear_mode parameter to compute_slope6_normalization() function
- [ ] Modify use_slope8 logic to respect force_linear_mode flag (line 1897)
- [ ] Filter dataframe to only YOB=-2 in MC mode after adding column (around line 3363)
- [ ] Create separate MC processing branch in process_workbook() with simplified flow
- [ ] Verify MC mode only produces linear fits and YOB=-2 output


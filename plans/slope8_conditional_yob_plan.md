---
name: Conditional slope8 based on YearOfBirth
overview: Modify slope8 normalization to apply only for cohorts born before 1940 (YOB < 1940). For cohorts born in 1940 or later (YOB >= 1940), use the linear exponential fit method (fit_linear_median) instead. Both methods use the same fit window (enrollment to SLOPE_FIT_END_ISO, skipping HVE weeks and highest dose delay), but linear uses time-centered approach while slope8 uses s=0 at enrollment.
todos: []
---

# Conditional Slope8 Application Based on YearOfBirth

## Overview

Modify `compute_slope6_normalization` in [`code/KCOR.py`](code/KCOR.py) to:

1. **Conditionally apply slope8** (four-parameter depletion fit) only for cohorts born **before 1940** (YOB < 1940)
2. **For cohorts born >= 1940**, use **simple exponential** (linear fit with `fit_linear_median`)
3. **Unified fit window**: Both methods use the same fit window = adjustment window:

- Start: enrollment_date + skip weeks (see below)
- End: SLOPE_FIT_END_ISO
- **Skip weeks**: 
 - All doses: Skip first DYNAMIC_HVE_SKIP_WEEKS (2 weeks) since enrollment
 - Highest dose: Additionally skip SLOPE_FIT_DELAY_WEEKS (15 weeks), so total 17 weeks for highest dose
 - Other doses: Skip 2 weeks total

4. **Time centering**:

- **Linear fit**: Time-centered (t_c = t - t_mean, where t_mean is center of fit window)
- **Slope8 fit**: t=0 at enrollment (s=0 at enrollment, not centered)

## Changes Required

### 1. Add Configuration Constant

- **Location**: Near top of file with other slope configuration parameters (around line 120)
- **Add**: `SLOPE8_MAX_YOB = 1940` constant
- This defines the maximum YearOfBirth for which slope8 will be applied

### 2. Modify Linear Fit Data Collection

- **Location**: [`code/KCOR.py`](code/KCOR.py) around lines 1704-1727
- **Current behavior**: Linear fit uses `fit_weeks` from `baseline_window` (2022-01 to SLOPE_FIT_END_ISO)
- **New behavior**: Linear fit should use same fit window as slope8:
- **Fit window start**: 
 - Highest dose: enrollment_date + DYNAMIC_HVE_SKIP_WEEKS + SLOPE_FIT_DELAY_WEEKS (2 + 15 = 17 weeks)
 - Other doses: enrollment_date + DYNAMIC_HVE_SKIP_WEEKS (2 weeks)
- **Fit window end**: SLOPE_FIT_END_ISO
- **Time centering**: Compute `t_mean` as center of the fit window (mean of t values in fit window)
- **Fit**: Use `fit_linear_median` on fit window data with time-centered approach (t_c = t - t_mean)

### 3. Modify Slope8 Fit Data Collection

- **Location**: [`code/KCOR.py`](code/KCOR.py) around lines 1875-1913
- **Current behavior**: Slope8 skips DYNAMIC_HVE_SKIP_WEEKS implicitly (only checks SLOPE_FIT_DELAY_WEEKS for highest dose)
- **New behavior**: Explicitly skip DYNAMIC_HVE_SKIP_WEEKS for all doses:
- **Fit window start**:
 - Highest dose: enrollment_date + DYNAMIC_HVE_SKIP_WEEKS + SLOPE_FIT_DELAY_WEEKS (2 + 15 = 17 weeks)
 - Other doses: enrollment_date + DYNAMIC_HVE_SKIP_WEEKS (2 weeks)
- **Fit window end**: SLOPE_FIT_END_ISO
- **Time axis**: s=0 at enrollment (not centered)

### 4. Modify `compute_slope6_normalization` Function

- **Location**: [`code/KCOR.py`](code/KCOR.py) starting at line 1556
- **Change**: In the cohort processing loop (around line 1661), add a check before attempting slope8:
- If `yob >= SLOPE8_MAX_YOB` (or `yob == -2` for all-ages), skip slope8 entirely and proceed directly to linear mode
- If `yob < SLOPE8_MAX_YOB`, proceed with current slope8 logic

### 5. Implementation Details

**Fit Window Calculation (for both linear and slope8):**

- Determine highest dose: `max_dose = max(max(pair) for pair in dose_pairs)`
- Check if current dose is highest: `is_highest_dose = (dose == max_dose)`
- Calculate fit window start offset:
- If highest dose: `fit_start_weeks = DYNAMIC_HVE_SKIP_WEEKS + SLOPE_FIT_DELAY_WEEKS` (2 + 15 = 17)
- Otherwise: `fit_start_weeks = DYNAMIC_HVE_SKIP_WEEKS` (2)
- Fit window: `enrollment_date + fit_start_weeks` to `SLOPE_FIT_END_ISO`

**Linear Fit (YOB >= 1940):**

1. Collect data from fit window (enrollment + skip weeks to SLOPE_FIT_END_ISO)
2. Compute `t_mean` as mean of t values in fit window
3. Fit: `fit_linear_median(t_values - t_mean, logh_values)` (time-centered)
4. Store parameters: `mode="linear"`, `a`, `b`, `t_mean`, `c=0.0`

**Slope8 Fit (YOB < 1940):**

1. Collect data from fit window (enrollment + skip weeks to SLOPE_FIT_END_ISO)
2. Convert to s (weeks since enrollment): `s = weeks_since_enrollment`
3. Fit: `fit_slope8_depletion(s_values, logh_values)` (s=0 at enrollment)
4. Store parameters: `mode="slope8"`, `C`, `ka`, `kb`, `tau`

**Application (Normalization):**

- **Fit region = adjustment region**: Same window used for both fitting and applying normalization
- **Linear mode**: Apply `h_norm = h * exp(-b * t_c)` where `t_c = t - t_mean` (time-centered)
- **Slope8 mode**: Apply `h_norm = h * exp(-kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))` where `s = t` (s=0 at enrollment)

### 6. Specific Code Changes

**Around line 1704** (linear fit data collection):

- Replace `for week in fit_weeks:` loop with logic that:

1. Determines fit window: `slope8_end_dt = _iso_to_date_slope6(SLOPE_FIT_END_ISO)`
2. Checks if highest dose: `is_highest_dose = (dose == max_dose)`
3. Calculates fit start offset: `fit_start_weeks = DYNAMIC_HVE_SKIP_WEEKS + (SLOPE_FIT_DELAY_WEEKS if is_highest_dose else 0)`
4. Iterates through cohort_data and collects data where:

 - `weeks_since_enrollment >= fit_start_weeks`
 - `date_died <= slope8_end_dt`

5. Computes `t_mean` as mean of t values in fit window
6. Fits using time-centered approach: `fit_linear_median(t_values - t_mean, logh_values)`

**Around line 1875** (slope8 fit data collection):

- Modify existing slope8 data collection to explicitly skip DYNAMIC_HVE_SKIP_WEEKS:

1. Calculate fit start offset: `fit_start_weeks = DYNAMIC_HVE_SKIP_WEEKS + (SLOPE_FIT_DELAY_WEEKS if is_highest_dose else 0)`
2. In the loop, change condition from:

 - Current: `if not is_highest_dose or weeks_since_enrollment >= SLOPE_FIT_DELAY_WEEKS:`
 - New: `if weeks_since_enrollment >= fit_start_weeks:`

**Around line 1871** (before slope8 fit section):

- Add conditional check: `if yob < SLOPE8_MAX_YOB and yob != -2:`
- Wrap the entire slope8 attempt block (lines 1871-2118) in this condition
- For cohorts with `yob >= SLOPE8_MAX_YOB`, skip directly to linear mode assignment

**Around line 2120** (where mode is determined):

- Ensure that cohorts with `yob >= SLOPE8_MAX_YOB` get `mode="linear"` with proper parameters from the linear fit
- Update log messages to indicate when linear mode is used due to YOB threshold vs. fallback

### 7. Logging Updates

- Update log messages to indicate:
- When linear mode is used due to YOB threshold: `mode=linear(yob_threshold)`
- When linear mode is used as fallback: `mode=linear(fallback_slope8_not_attempted)`
- Log should indicate fit window used (enrollment_date + skip_weeks to SLOPE_FIT_END_ISO)
- Log should indicate skip weeks: `skip_weeks={fit_start_weeks}` (2 for normal doses, 17 for highest dose)

## Testing Considerations

- Verify that cohorts with YOB < 1940 use slope8 with correct fit window (skip weeks applied)
- Verify that cohorts with YOB >= 1940 use linear mode with correct fit window (skip weeks applied)
- Verify that highest dose skips 17 weeks (2 + 15) in both linear and slope8 fits
- Verify that other doses skip 2 weeks in both linear and slope8 fits
- Verify that linear fit uses time-centered approach (t_mean computed from fit window)
- Verify that slope8 fit uses s=0 at enrollment (not centered)
- Verify that fit region = adjustment region (same window for both)
- Verify that all-ages cohort (YOB=-2) uses appropriate method
- Ensure normalization parameters are correctly stored and applied in `apply_slope6_norm` function

## Notes

- The linear fit method (`fit_linear_median`) already exists and is used as fallback
- Linear normalization formula: `h_norm = h * exp(-b * t_c)` where `t_c = t - t_mean` (time-centered)
- Slope8 normalization formula: `h_norm = h * exp(-kb*s - (ka - kb)*tau*(1 - exp(-s/tau)))` where `s = t` (s=0 at enrollment)
- The `apply_slope6_norm` function (line 3581) already handles both modes correctly, but may need updates to ensure fit window = application window
- DYNAMIC_HVE_SKIP_WEEKS = 2 (skip first 2 weeks since enrollment for all doses)
- SLOPE_FIT_DELAY_WEEKS = 15 (additional skip for highest dose, so 17 weeks total for highest dose)


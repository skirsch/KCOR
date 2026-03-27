---
name: kcor-final-changes-plan
overview: "Implement the two final KCOR updates from `final_changes.md`: tighten bad-fit threshold and add meaningful summary quality flags with detailed triggers, then validate on Czech data."
todos:
  - id: lower-bad-fit-default
    content: Change bad-fit relRMSE default threshold from 1e6 to 1e5 while preserving YAML override support.
    status: completed
  - id: add-quality-flag-columns
    content: Add quality_flag and quality_flag_detail derivation to create_summary_file with well-identified cohort criteria.
    status: completed
  - id: wire-summary-and-log-output
    content: Include new columns in theta0_status_summary export and dual_print summary logs.
    status: completed
  - id: run-final-validations
    content: Execute Czech-data checks for threshold impact and flag expectations, then lint KCOR.py.
    status: completed
isProject: false
---

# KCOR Final Changes Plan

## Goal

Implement the two requested final updates in [C:/Users/stk/Documents/GitHub/KCOR/documentation/specs/KCORv7/final_changes.md](C:/Users/stk/Documents/GitHub/KCOR/documentation/specs/KCORv7/final_changes.md) with minimal, targeted edits to [C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py](C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py), then run the specified validations.

## Change 1: Bad-Fit Threshold

- Update the bad-fit relRMSE default threshold from `1e6` to `1e5` in the active threshold default path (currently in `KCOR.py` config/default handling).
- Preserve configurability if user YAML overrides are present; only adjust the default value.
- Keep all existing bad-fit control flow/status behavior unchanged.

## Change 2: Summary Quality Signals

In `create_summary_file` logic in `KCOR.py`:

- Keep existing `any_non_ok_status` behavior as full transparency signal.
- Add `quality_flag` (boolean) per enrollment date, `True` only if a non-OK status occurs in a well-identified cohort where:
  - `year_of_birth <= 1960`
  - `total_quiet_deaths >= min_quiet_deaths`
  - `dose == 0`
- Add `quality_flag_detail` (string): comma-separated `(year_of_birth,dose,status)` tuples that triggered `quality_flag`; empty string when `quality_flag=False`.
- Ensure both new columns appear in:
  - `theta0_status_summary` sheet output,
  - run-log `dual_print` summary output.

## Validation Plan

- Functional checks on Czech run outputs:
  - `2022_06 YOB=1990 dose=2` should show `theta_applied=0` and `theta_fit_status='insufficient_signal'` under tightened threshold behavior.
  - no `OK` cases with `relRMSE > 1e5`.
  - `any_non_ok_status` remains `True` as before.
  - `quality_flag=False` for all enrollment dates.
  - `quality_flag_detail` empty for all enrollment-date rows.
- Quality checks:
  - run lints for `KCOR.py`.
  - confirm output schema stability with added summary columns.

## Files Expected To Change

- [C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py](C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py)
- (Only if needed for explicit override documentation) [C:/Users/stk/Documents/GitHub/KCOR/data/Czech/Czech.yaml](C:/Users/stk/Documents/GitHub/KCOR/data/Czech/Czech.yaml)


# Monte Carlo sample isolation regression

## The bug in plain language

Bootstrap samples are alternative versions of the same follow-up period. Each version needs its own clock: week 0, week 1, week 2, and so on. The original implementation counted rows across samples sharing a birth year and dose. With multiple samples in one worksheet, observations from the same calendar week received different elapsed times.

Adding bootstrap samples could therefore change the fitted correction and the estimate for the unchanged original dataset. This violates a basic requirement: processing a sample alongside other samples must give the same result as processing it alone.

This defect exists in the repository's production `make monte_carlo` path. It was not introduced by a separate uncertainty-validation experiment.

## Execution path and cause

1. `code/Makefile` runs `KCOR_CMR.py` with `MONTE_CARLO=1`. That script concatenates iterations into one enrollment worksheet and distinguishes them with `mc_id`.
2. `code/KCOR.py`, inside `process_workbook`, sorts the input and constructs `t` using `groupby(["YearOfBirth", "Dose"]).cumcount()`.
3. This grouping omits `mc_id`. With M samples on a complete weekly grid, the rows for calendar week w receive time indices from M*w through M*w+M-1, instead of all receiving w. The assignment within a week depends on row ordering.
4. Subsequent calculations use `t`, including skipped-week handling and fitting. The fitter's early anchor can become empty for a sample; a failed fit can then cause the correction to fall back to zero frailty. This is not merely a change of time units.

The fix includes `mc_id` in that grouping when Monte Carlo mode is active and the identifier is present. The ordinary single-dataset grouping stays the same. This repair retains the existing complete-grid row-count convention; it is not a redesign for missing calendar weeks.

## Related diagnostic defect

Ordinary fit records use `(enrollment, birth_year, dose)` keys. Monte Carlo records use `(enrollment, mc_id, birth_year, dose)` keys. Several parts of `create_summary_file` assumed ordinary keys, so Monte Carlo failures could disappear from the compact status tally and sample-specific details could be missing.

The fix recognizes the sample-specific keys, preserves `mc_id` in Monte Carlo diagnostic output, and retains failed Monte Carlo fits even when fitted parameters are missing. Detailed failures existed before the fix; the defect was that summary reporting could miss them. The regression supplies a failed fit explicitly so this reporting check remains meaningful after the clock is corrected.

## Run the regression

From the repository root, with the existing KCOR dependencies installed:

```sh
python test/monte_carlo/test_sample_isolation.py
```

The test constructs synthetic count workbooks; no person-level input file is needed. It uses the existing Czech configuration, overrides the quiet window and dose comparison in memory, disables optional wave correction, and checks age-specific outputs.

The three tests check:

- The original curve, time indices, fitted parameters, and analytic interval columns agree when processed alone or alongside five bootstrap samples. Shuffling rows and relabeling sample identifiers preserve all sample curves. Ordinary processing agrees with the original-alone result on this fixture.
- A supplied failed Monte Carlo fit appears in the status count, sample-specific row diagnostics, and fit table, including when parameters are missing.
- Ordinary three-part diagnostic keys retain their existing behavior and output format.

`KCOR_TEST_SOURCE` can select another `code/KCOR.py` for comparison. `KCOR_FIX_RESULTS` can retain generated workbooks and logs in a new output directory; otherwise the tests use a temporary directory. Use a fresh output directory for each run.

## Evidence and interpretation for reviewers and LLM evaluators

Against the pre-fix source, two of the three tests fail. With the fix, all three pass. A separate replay of retained synthetic input also compared a ten-sample run with the original sample processed alone: all 100 original-sample KCOR curve points matched exactly after repair. At week 80, the old combined run gave 1.053172263346009; both the original-alone and corrected combined runs gave 1.004049494273245. These are reproduction values, not empirical treatment-effect estimates. The standalone tests above do not require those external replay artifacts.

Keep the following conclusions separate:

- This is a software defect in an existing Monte Carlo workflow. A regular run containing only one dataset does not trigger this cross-sample clock defect.
- Bootstrap estimates and resulting Monte Carlo intervals produced through the affected path require recomputation. The direction and size of the change are not established by this regression.
- The ordinary workbook's analytic interval columns and the Monte Carlo percentile/basic intervals are different outputs. The regression compares the former for sample isolation; it does not validate statistical coverage of either interval construction.
- Fixing the clock and reporting does not establish that corrected intervals achieve their nominal coverage. That requires a separate calibration study.
- These findings alone do not identify the code revision or workflow used to produce historical manuscript results, and do not establish that every published result is affected.
- This pull request changes neither KCOR's mathematical model nor its bootstrap sampling algorithm, interval formulas, or manuscript claims.

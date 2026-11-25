# Objection 3 production verification

The revised pipeline uses the sample-isolation repair included with PR 5. Each bootstrap sample advances its own follow-up clock. Diagnostic summaries use the sample ID to look up fits, record failures and attribute flags. An additional fix to the Monte Carlo summary handles an original-only run or a group without finite bootstrap draws. It reports zero draws and omits the interval. The original point estimate is never substituted for a resample.

## Regression and real-data replay

The packaged sample-isolation regression failed two of three tests before the repair. Afterward all three passed, including ordinary single-dataset diagnostic compatibility. To repeat it, set KCOR_TEST_SOURCE to the absolute revised code/KCOR.py path and run the test_sample_isolation.py script in the PR 5 monte_carlo_sample_isolation package.

For the real-data check, the input was data/Czech/KCOR_CMR_MC_audit_25.xlsx, enrollment 2021_24. Run the revised workbook entry point once on its original rows (mc_id zero) and once on the entire workbook, with MONTE_CARLO=1, DATASET=Czech, MC_ENROLLMENT_DATE=2021_24, MC_YOB_START=1940, MC_YOB_END=1945 and MC_INCLUDE_AGGREGATES=0. Supply distinct output directories named alone and combined with output.xlsx and run.log in each. Run check_workbooks.py with their parent directory.

All 525 original cohort-week/dose-pair rows were identical. The three reported percentile intervals agreed at output precision with the 2.5th and 97.5th percentiles recomputed from the 24 resamples. The combined diagnostic report identified the insufficient-signal fit for sample 17, birth-year group 1940, dose zero. The original-only run reported no interval.

## Calibration

Run check_coverage.py to reproduce the archived unequal-frailty scenario with the selected revised fitter using the existing study adapter. All 600 refitted original estimates matched the archived values, and all 600 passed the study gate. All 999 person-resampling estimates for the first dataset were also refitted and matched the archive. The interval endpoints for every dataset were recomputed from the archived draws. The nominal 95% percentile interval covered the true ratio in 502 of 600 datasets (83.7%). The wave module was inactive in this study. This was a replay of the retained study, not 600 new independent simulations.

The revised paper records this result alongside the separate study using the paper's joint initial fit. The next specification decision concerns the early fixed baseline rate in the production fitter versus the joint rate/frailty fit in section 2.4. Changing the fitter is separate from the sample-isolation repair.

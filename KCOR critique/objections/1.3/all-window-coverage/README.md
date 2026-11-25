# Coverage validation with the all-window joint fitter

This is an isolated follow-up to PR #5 objection 1.3. It evaluates the candidate from [2.7, all-window joint fitting](../../2.7/low-frailty/all-window-joint/README.md). It does not adopt the candidate or change the official paper. The exact weekly nonlinear-wave survival inversion remains a separate experiment pending lead-author approval.

## Paired main designs

The two archived [multi-window coverage scenarios](../multiwindow-coverage/README.md) are reproduced with the same data-generating law, seed namespace (20261124), 100 independent populations of one million people per scenario, 199 person-bootstrap draws per dataset, four age-by-dose cells, baseline rates 0.002 and 0.001, and generating frailty variances 4 and 0.5. Quiet windows follow enrollment weeks 2–26 and 35–52 (zero-based). The no-wave and nonlinear-wave (supplied alpha=1.18) scenarios both have exact pooled KCOR target 1. We retain the existing optional approximate wave correction, reporting endpoint, early normalization, geometric age pooling and diagnostic fallbacks. The only changed component is the experimental common-k, theta, profiled-gap two-start joint fitter. Archived reference estimates are not recomputed; matching bootstrap-batch pooling weights are checked against the archived arrays.

The deterministic expected-count outcome is exactly 1 for no wave, 0.994723 with the nonlinear wave. The approximately -0.53% nonlinear deterministic error belongs to the existing approximation combined with fitting a gamma working model to the survivor population; it is not a sampling-coverage effect.

## Workbook verification

The candidate fast replay matches the actual Monte Carlo workbook for the original and five resamples in each of three scenarios. Pooled and both age-specific reporting endpoints and the reported age-specific percentile intervals match, including the pathological stress dataset 34.

| Scenario | Maximum absolute endpoint discrepancy |
|---|---:|
| No wave | 4.44e-16 |
| Nonlinear wave | 5.26e-14 |
| Short-first-window stress | 2.09e-14 |

[Verification script](check_candidate_workbook.py) and [machine-readable check](workbook-verification.json).

## Original contamination experiment coverage check

The [2.7 contamination follow-up](../../2.7/all-window-contamination/README.md) identified finite-sample bias under zero added hazard, despite exact expected-count recovery. We therefore additionally test the interval coverage in its original single-window design at 50,000 and 5,000,000 people per cohort. All 200 original independent datasets at each size use the same archived original streams, mortality calibration, fifth accumulating week normalization, and generating theta=0.5 versus 4. Generate 99 new person-bootstrap draws per dataset by resampling each fixed-size cohort's death-week/survivor counts multinomially. Draw seed components are (20261125, cohort size, replicate). Resampling within cohort conditions on the fixed enrollment sizes in that simulation. This is the historical simulation's single-window pipeline, not a new intervention or the pooled Czech workbook. Original endpoints are checked against the prior contamination replay.

## Earlier low-information multi-window stress

The all-window trial already reran 50 independent 50,000-person two-window stress datasets with 99 person-bootstrap draws. Its 48/50 coverage (96%; exact 95% binomial interval 86.3–99.5%) versus 50/50 under the previous joint fitter was paired with a reduction in median interval width from 213.056 to 0.3146, a mean endpoint 1.03676 [1.01453,1.05899] and zero failed subgroup fits. The original joint fitter covered with pathological intervals. This result is retained in the combined comparison, not rerun with a different seed.

All confidence intervals for coverage are exact binomial intervals across independent datasets; reported mean point and width intervals are Monte Carlo mean intervals, distinct from the within-dataset person-bootstrap intervals.

## Completed results

All 200 matched one-million-person datasets and 400 single-window datasets completed. Archived pooling weights match the reproduced weights, and all 400 original single-window endpoint values reproduce the 2.7 replay. Candidate results below include unconditional diagnostic behavior. The two main scenarios use 199 person draws; the two single-window and earlier stress designs use 99.

### Main paired comparison

| Scenario | Fitter | Covered / 100 [exact 95% coverage CI] | Mean KCOR [95% MC CI] | Mean interval width [95% MC CI] | RMSE |
|---|---|---|---|---|---:|
| No wave | Fixed | 78 [68.6%,85.7%] | 1.0098 [1.0080,1.0116] | 0.0409 [0.0395,0.0424] | 0.0134 |
| No wave | Previous joint | 94 [87.4%,97.8%] | 1.0025 [0.9987,1.0063] | 0.0744 [0.0733,0.0756] | 0.0194 |
| No wave | All-window | 97 [91.5%,99.4%] | 0.9996 [0.9955,1.0037] | 0.0847 [0.0835,0.0860] | 0.0206 |
| Nonlinear wave | Fixed | 88 [80.0%,93.6%] | 1.0064 [1.0036,1.0093] | 0.0522 [0.0512,0.0533] | 0.0156 |
| Nonlinear wave | Previous joint | 92 [84.8%,96.5%] | 0.9980 [0.9936,1.0023] | 0.0779 [0.0768,0.0791] | 0.0220 |
| Nonlinear wave | All-window | 90 [82.4%,95.1%] | 0.9956 [0.9907,1.0006] | 0.0877 [0.0864,0.0891] | 0.0251 |

Compared with the prior joint fitter, the all-window candidate gained coverage in three no-wave datasets and lost none (paired exact p=0.25); with the nonlinear wave it gained two and lost four (p=0.6875). This does not establish a significant change in coverage between the two joint approaches. Widths increased by 0.01029 [0.00924,0.01134] and 0.00979 [0.00874,0.01084], respectively. No insufficient-signal flags or failed subgroup fits occurred in either main scenario. The no-wave mean bias is practically negligible. The nonlinear-wave coverage estimate of 90% is marginally compatible with 95% (upper exact CI 95.1%); it remains the weakest scenario and is not proof of nominal calibration. Its deterministic expected-count KCOR differs from 1 by about -0.53%; this is evidence of a model/approximation contribution to the error, not a claim that it explains every uncovered interval.

### Single-window original null

| Population per cohort | Covered / 200 [exact 95% coverage CI] | Mean KCOR [95% MC CI] | Mean interval width [95% MC CI] | RMSE |
|---:|---|---|---|---:|
| 50,000 | 187 [89.1%,96.5%] | 1.0494 [1.0305,1.0684] | 0.5494 [0.5366,0.5621] | 0.1442 |
| 5,000,000 | 192 [92.3%,98.3%] | 1.0013 [0.9993,1.0033] | 0.0539 [0.0532,0.0546] | 0.0142 |

No failed subgroup fits, one insufficient-signal flag among the small-cohort draws, no flags among large-cohort draws. The original small-cohort null is biased despite reasonable coverage; broad intervals acknowledge poor resolution. The tenfold reduction in observed width after increasing population 100-fold follows the expected approximate inverse-square-root sample-size behavior.

### Across all five designs

The new candidate's observed coverages are 97%, 90%, 96%, 93.5%, 96% for no wave, nonlinear wave, short-window multi-window stress, small-cohort single-window null and large-cohort single-window null. Every exact coverage confidence interval contains 95%, but the nonlinear-wave result is marginal, the designs share modeling assumptions and target a null, and the coverage checks are conditional on fixed alpha, configured windows, and the existing fallback convention. These results substantially strengthen the evidence for person resampling and the experimental all-window fitter in the observed settings. They do not establish universal coverage or justify changing existing reporting rules without approval. In particular, the tested full in-wave numerical inversion has not been combined with the candidate; the remaining approximation must be discussed under 1.2. Additional non-null and misspecification studies would be separate validation extensions.

[Main results](results.json), [all candidate draws](coverage-draws.npz), [paired interval records](intervals.csv), [single-window results](contamination-coverage.json), [single-window draws](contamination-draws.npz), [single-window interval records](contamination-intervals.csv). Reproduce with Python 3.11 using replay.py, check_candidate_workbook.py and contamination_coverage.py.

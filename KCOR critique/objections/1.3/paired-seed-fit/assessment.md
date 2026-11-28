# Objection 3: paired test of the initial baseline-rate fit

The question is whether adopting section 2.4's joint initial fit improves coverage of the existing production percentile interval in the model-consistent scenario where it undercovered.

## Result

| On the same 200 datasets | Production early rate | Joint initial fit |
|---|---:|---:|
| Nominal 95% intervals covering the true ratio | 160/200 (80%) | 194/200 (97%) |
| Mean interval width | 0.0478 | 0.1488 |
| Mean point estimate (truth 1) | 1.00975 | 1.00587 |
| Root mean squared point error | 0.01357 | 0.03751 |
| Original fits passing the study gate | 200/200 | 200/200 |

The joint fit gained coverage on 38 datasets and lost it on four (two-sided exact paired test, p = 5.65e-8). The observed 17 percentage point gain came with intervals 3.11 times as wide. The joint mean was closer to the target, while typical point error as measured by RMSE was larger. All 39,800 experimental draw estimates were finite with no reported fit failure. Both versions matched the workbook entry point on 600 weekly rows per version. These results apply to the specified scenario and do not establish nominal coverage across other cohorts or designs.

## Comparison

Use the first 200 archived datasets in the one-million-person unequal-frailty study, in archive order. Use the first 199 pooled person-bootstrap samples of each dataset for both methods. The frailties are 4 and 0.5, quiet weeks are 6 through 40, the target is the week-52 ratio anchored at week 5, and its true value is 1. The optional wave correction is inactive.

The control is the revised production fitter. Its original estimates are refitted and its archived bootstrap draws are replayed by seed, with the first, middle and last production draw verified in each dataset. The experimental version replaces the initial single-parameter fit with the existing manuscript Step 1 implementation on the nearest quiet window. The jointly estimated rate supplies the baseline to the subsequent production refit, which still uses the original weekly approximation and its quiet and early anchor weeks. The same normalization, safeguards and 2.5th/97.5th percentile calculation apply in both arms. This experiment does not substitute a different pooled refit or change the production edition.

The study compares coverage within datasets, relative interval widths and error of the original point estimates. The paired test counts datasets covered only by one method. On exact expected counts, the revised production estimate was 1.00461455 and the estimate after substituting Step 1 was 0.99993566.

## Reproduction

Run `python compare.py` to regenerate `results.json`. This uses the saved counts and seeds from the corrected-validation archive. Run `python check_workbook.py` to compare the study adapter with the actual workbook entry point for the original sample and five person-bootstrap replicates under each method.

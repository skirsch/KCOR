# Combined all-window fitter and exact weekly wave inversion

Status: experimental, not adopted. This trial combines the [all-window multistart fitter](../../2.7/low-frailty/all-window-joint/README.md) with the [exact weekly survival inversion](../weekly-survival/README.md). It keeps the existing fitting objective, anchor, pooled weights, wave windows, fixed supplied alpha=1.18 and trial deficit attribution. The 3.2 first-entry correction and quiet-only diagnostic are not included, so the comparison isolates the change to the wave correction. The non-wave and proportional cases retain ordinary inversion.

## Designs and verification

- Main nonlinear-wave scenario: same 100 independent one-million-person datasets and 199 person-bootstrap draws per dataset as the [previous all-window coverage validation](../../1.3/all-window-coverage/README.md). Four age-by-dose cells have generating frailty variances 4 and 0.5 in each age, true pooled KCOR 1. Both original and bootstrap estimates retained.
- Short-window stress: same 50 independent 50,000-person datasets and 99 draws as the matched multistart stress study. True KCOR 1.
- Table 8: same archived input, normalization, pooling, forced alpha, and 999 person-bootstrap draws per enrollment. The prior all-window points were independently reproduced at specified matched draws; all six new original points independently match ordinary workbook processing, max discrepancy 8.44e-15.
- All applied fitted theta values and fit diagnostics across both synthetic studies are exactly identical to the archived all-window approximate-wave arms. The change is downstream of the fit. The maximum absolute weekly inversion cumulative survival residual is below 3.8e-15 across all three replays. No additional failures occurred.
- No-wave expected-count outputs still recover the true ratio exactly.

## Repeated-sample coverage

All confidence intervals on coverage are exact binomial intervals; intervals on mean KCOR and mean width are Monte Carlo intervals across independent datasets.

| Scenario | Wave correction | Coverage [95% CI] | Mean KCOR [95% MC CI] | Mean interval width [95% MC CI] | RMSE |
|---|---|---|---|---|---:|
| Main 1M | Approximate | 90/100 [82.4%,95.1%] | 0.99565 [0.99072,1.00057] | 0.08773 [0.08637,0.08908] | 0.02506 |
| Main 1M | Exact weekly | 90/100 [82.4%,95.1%] | 0.98982 [0.98493,0.99471] | 0.08708 [0.08573,0.08843] | 0.02654 |
| Stress 50k | Approximate | 48/50 [86.3%,99.5%] | 1.03676 [1.01453,1.05899] | 0.32496 [0.30443,0.34548] | 0.08572 |
| Stress 50k | Exact weekly | 48/50 [86.3%,99.5%] | 1.03068 [1.00878,1.05258] | 0.31934 [0.29903,0.33965] | 0.08222 |

Main scenario: two datasets gain coverage, two lose it, paired exact p=1. Stress: all coverage dispositions unchanged. Main width decreases by 0.000645 [0.000542,0.000748]; stress by 0.005620 [0.004975,0.006265]. No main-scenario insufficient-signal or failed fits, no stress failures. The stress insufficient-signal flags are exactly unchanged. The exact numerical correction does not improve overall coverage in either scenario. In the high-information case it makes the mean modestly less accurate, whereas in the stress case it makes the mean closer to the target.

## Why does an exact inversion give an imperfect endpoint?

The exact inversion uses the supplied baseline rate and enrollment frailty law. The current all-window fitting objective still describes the quiet survivor population with the proportional gamma working law, including after alpha differs from one. A nonlinear wave generally changes survivor frailty to a non-gamma distribution. Thus the fit and correction may use different survivor models.

The deterministic expected-count controls and counterfactual input substitutions quantify this discrepancy. True target=1. No sampling intervals are applicable to these deterministic quantities.

| Inputs to wave inversion | Approximate, main | Exact weekly, main | Approximate, stress | Exact weekly, stress |
|---|---:|---:|---:|---:|
| Fitted k, fitted theta | 0.994723 | 0.988967 | 0.995889 | 0.990133 |
| Fitted k, generating theta | 1.005546 | 0.999943 | 1.005608 | 0.999994 |
| Generating k, fitted theta | 0.994792 | 0.989023 | 0.995896 | 0.990139 |
| Generating k, generating theta | 1.005615 | 1.000000 | 1.005615 | 1.000000 |

The main noiseless fitted frailties are 4.2473, 0.5366, 4.2848, 0.5375; their generating values are 4, 0.5, 4, 0.5. The stress results are 4.2221, 0.5313, 4.2492, 0.5319. Substituting true theta alone almost completely restores the exact inversion's endpoint; substituting true k alone does not. This strongly implicates the fitted frailty/model approximation in the remaining deterministic error, rather than the numerical inversion. The existing approximate wave correction has a positive error of about 0.56% with true inputs and partially cancels the error in the fitted frailty by coincidence in this design. That cancellation is not a justification for an approximate equation generally.

The exact weekly inversion is mathematically valid conditional on the chosen baseline/wave attribution and correct inputs; the full pipeline does not yet estimate its parameters using a survival model consistent with nonlinear selection. This is related to 3.2's mixed-period problem, not solely low-variance sampling uncertainty. Further investigation requires a distinct decision about the fitted model. No operational KCOR run should be supplied generating parameter values; the substitution is a known-truth diagnostic only.

## Table 8: all six matched pointwise 95% person-bootstrap intervals

| Enrollment | Pair | All-window + approximate | All-window + weekly survival |
|---|---|---|---|
| 2021 | 1/0 | 1.01293 [0.95567,1.06846] | 1.01279 [0.95552,1.06823] |
| 2021 | 2/0 | 1.03343 [0.97066,1.10595] | 1.03310 [0.97032,1.10552] |
| 2021 | 2/1 | 1.02024 [0.95791,1.09426] | 1.02005 [0.95768,1.09403] |
| 2022 | 1/0 | 1.01190 [0.96751,1.05505] | 1.01195 [0.96848,1.05587] |
| 2022 | 2/0 | 1.02766 [0.98506,1.07906] | 1.03028 [0.98735,1.08406] |
| 2022 | 2/1 | 1.01558 [0.97035,1.06781] | 1.01811 [0.97103,1.07380] |

All six intervals contain 1; the largest point change is +0.002614. Fitted enrollment frailty and baseline are unchanged.

Deficit handling was not exercised in the synthetic waves (617,096 evaluated subgroup-weeks in the main run, 116,160 in stress), but was exercised in the empirical controls. In 2021, 690/127,344 evaluated bootstrap subgroup-weeks (0.54%) were below the fitted conditional baseline; in 2022, 6,322/44,955 (14.06%). Evaluated-week denominator excludes zero-frailty shortcut cases. At the trial attribution convention, observed mortality is retained and each deficit gives a smaller fitted baseline increment, with previous wave exposure preserved. Lead-author approval covers this attribution as well as the exact inversion equation.

## Disposition

The exact numerical in-wave equation is verified and has little effect on Table 8; it does not by itself improve nonlinear-wave coverage. In this comparison, the remaining deterministic discrepancy arises predominantly from the use of the proportional gamma fitting approximation after nonlinear selection. Recommend considering the exact wave equation on mathematical grounds, while explicitly recognizing the dependence on the fitted inputs and deficit convention. Do not describe the combined method as fully validated for nonlinear waves or adopt it merely to improve the point estimate. Discussion of a consistent nonlinear fitting model and mixed periods is pending with 3.2. No paper or official pipeline change.

Reproduce with Python 3.11: replay.py main, replay.py stress, replay.py table, parameter_diagnostic.py, check_combined_workbook.py. All matched intervals, diagnostics and parameter counterfactuals are saved alongside the scripts.

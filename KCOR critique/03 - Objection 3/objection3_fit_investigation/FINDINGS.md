# Why the quiet-window fit shifts the one-million-person result

## Main finding

The corrected implementation's residual population offset is caused chiefly by **fixing the baseline mortality rate from four early observed weeks before fitting frailty**. Those observed rates already reflect selection: people with higher frailty die earlier, so the observed rate is below the underlying unselected baseline rate. The direct early estimate is then held fixed in the quiet-window frailty fit. In this scenario the fixed value is lower than the rate that best fits the same quiet-window hazard model, especially in the high-frailty arm. The frailty parameters consequently fit low, shifting the normalized output above its model-defined value of 1.

This is an estimation/approximation issue, not floating-point drift or a failed optimizer. The original and saved bootstrap calculations are internally reproducible. A counterfactual joint fit nearly removes the **deterministic** offset but increases the variation of estimates across sampled datasets, so it is not a validated replacement.

## Exact model and production approximation

For week `w`, let `b_w = k exp(gamma (w - skip))` be the generating baseline hazard increment, and `B_w` its cumulative sum from enrollment. An arm with gamma frailty `theta` has survival

`S_w = (1 + theta B_w)^(-1/theta)`.

The exact observed weekly cumulative hazard increment is

`h_w = [log(1 + theta B_w) - log(1 + theta B_(w-1))] / theta`,

with the exponential limit at `theta = 0`. The simulation generates deaths from this survival model. Its exact expected death counts reproduce that formula with maximum weekly hazard discrepancy below `1.1e-16` in the fitted window.

The production fitter instead computes `k_fixed` as the mean of `h_w / exp(gamma (w - skip))` over the first four post-skip weeks. It fits `theta` against the approximation

`h_w ~= k_fixed exp(gamma (w - skip)) / [1 + theta H_gom_before(w)]`,

where `H_gom_before` accumulates its own post-skip Gompertz increments. This is the actual source's continuous-style hazard formula applied to discrete weekly data, with the early `k_fixed` treated as known. The exact survival model includes depletion from enrollment, including the two skipped weeks. The early observed `h_w` is therefore already below the generating `k exp(gamma (w - skip))` when frailty is positive. These formulas explain why the early rate read is systematically low; the precise magnitude also depends on the discrete approximation.

The true frailty values are 4.0 and 0.5, and the generating `k` is 0.001 in both arms. On **exact expected population counts**, with no sampling noise:

| Fit | Arm 0: rate, frailty | Arm 1: rate, frailty | Normalized output |
|---|---:|---:|---:|
| Actual production fit | 0.00098428, 3.76701 | 0.00099801, 0.47200 | **1.00461455** |
| Same hazard approximation, `k` and frailty fitted jointly | 0.00099012, 4.00327 | 0.00099875, 0.50041 | **0.99993565** |
| Exact discrete gamma hazard, jointly fitted | 0.00100000, 4.00000 | 0.00100000, 0.50000 | **1.00000000** |

The actual source's fixed-rate fit is independently reproduced by scalar minimization of its quiet-plus-anchor residuals to within `5e-8` in frailty. Changing its optimizer starting value from 0 to 20 changes the resulting frailty by at most `2.4e-9`. Thus, the 0.00461455 offset is not a local-solver or initialization accident. The approximate hazard model can fit the exact population very closely when its two parameters are estimated together: its maximum quiet-window hazard residual is `1.4e-12`. Fitting the exact discrete formula recovers the generating parameters at numerical precision.

Simply substituting the generating `k = 0.001` while leaving the rest of the original fitter unchanged produces 0.99211, an overcorrection. The production model's baseline-rate choice and its hazard approximation interact. The joint-fit rows above are controlled probes, not approved changes.

With true frailty parameters in the correction, the normalized output is 1 to numerical precision. The post-skip cumulative observed hazard is `log[(1+theta B_w)/(1+theta B_(skip-1))]/theta`. Inverting it with that `theta` gives `(B_w - B_(skip-1))/(1+theta B_(skip-1))`. The arm-specific denominator cancels when the ratio is normalized at week 5. This checks the prespecified model target independently of fitting.

## What happens across the 600 saved datasets

The production point estimate averages **1.00986**, with standard deviation **0.01058**. Refit each of the same saved original datasets jointly under the production hazard approximation: the mean becomes **1.00391**, but the standard deviation rises to **0.02613**. The root mean squared error against truth 1 changes from **0.01445 to 0.02639**. A joint fit using the exact discrete hazard gives almost the same distribution (mean 1.00397, standard deviation 0.02612). All 600 joint refits converged in both arms.

The joint approximate fit therefore removes most of the *mean* upward offset but makes the estimate about 2.5 times more variable across these datasets. This is a genuine bias-variance tradeoff. The present production percentile interval averages only **0.04962** wide; reusing it after a joint-fit change would be invalid. A new resampling and coverage design would be required before judging any replacement.

The smaller-frailty arm is often estimated near its zero boundary: 202/600 production originals have fitted frailty below 0.01, compared with 131/600 under the exploratory joint fit. The original operational gate accepts these fits because its explicit bound flag checks the upper bound. This lower-bound behavior can make the estimator and its bootstrap distribution asymmetric; it is not the whole explanation, since the deterministic population offset occurs with neither arm at a bound.

To inspect the original bootstrap rather than invent new samples, three retained 999-draw batches were replayed from their original seeds. Every estimate matched its saved value exactly. In a covered dataset, 80/999 bootstrap fits put the smaller-frailty arm below 0.01. In two missed datasets that count is 542/999 and 435/999; their percentile intervals are respectively `[1.00361, 1.06415]` and `[1.00576, 1.06079]`. These examples illustrate the boundary and upward tail, but three selected datasets do not estimate the population frequency of that mechanism. The full 600-dataset coverage result remains in the corrected-validation module.

## Implication for objection 3

This investigation makes the remaining issue more specific: the quiet-window procedure fixes a baseline rate read from already selected mortality, which biases the fitted frailty parameters and the model-defined contrast in this scenario. The percentile bootstrap distribution is also shifted upward and asymmetric, particularly when the smaller-frailty fit approaches zero. At one million people, the resulting interval is too far above the target often enough to cover only 83.7% of datasets. The observed weak performance is not explained by a missing fit, random-number error, quantile arithmetic, or solver initialization.

The controlled joint fit shows a way to remove the deterministic offset on exact counts, but it increases the sampling variation of the point estimate. Therefore no production fix or switch of interval type follows automatically. A subsequent change would need an explicit target, a prespecified estimator and diagnostic rule, new production-equivalence checks, and fresh coverage validation. Historical paper outputs require separate provenance checks before these results can be attributed to them.

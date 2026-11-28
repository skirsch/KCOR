# Why the corrected large-population percentile interval covers only 83.7%

This is a post-run diagnosis of the fixed, prespecified `unequal_large` scenario. It uses retained results and deterministic population calculations; no final scenario, acceptance rule, interval, or target was changed. The primary target is the model-defined normalized contrast, exactly 1 in this scenario.

## What the 600 datasets show

The percentile interval covered 1 in 502 of 600 datasets. **All 98 misses have the lower endpoint above 1; none lie wholly below it.** The median gap between 1 and a missed lower endpoint is 0.00366, with an interquartile range of 0.00154–0.00575. All 600 original analyses passed the declared fit gate, all 599,400 bootstrap estimates were finite, and no bootstrap fit failed. Recomputing every percentile endpoint from its saved 999 draws gave **exactly the stored endpoint**. Using only the first 499 draws changed coverage from 502/600 to 500/600. The ten-sample full production workbook and array adapter previously agreed to at most 4.5e-16 on complete curves. These checks strongly disfavor a quantile implementation, floating-point rounding, or incomplete-fit explanation for the 83.7% result.

The average original point estimate is 1.00986, whereas the truth is 1. The average midpoint of the percentile endpoints lies another 0.00822 above that point estimate, or about 1.01807. It lies above the point in 584 of 600 datasets. The average percentile width is 0.04962. The upward point error and right-shifted percentile interval therefore act in the same direction. The basic interval reflects the same bootstrap quantiles around the point estimate and covers 586/600 (97.7%) in this scenario. This contrast locates an important role for percentile construction and bootstrap asymmetry; it does not establish that basic intervals are calibrated in every scenario.

### Actual missed intervals and their widths

The true value is 1 in each row. These examples come directly from the retained per-dataset JSON files:

| Dataset repetition | Point estimate | Percentile 95% interval | Width | Distance of lower endpoint above truth |
|---:|---:|---:|---:|---:|
| 575 (narrowest miss) | 1.01980 | [1.00634, 1.03458] | 0.02824 | 0.00634 |
| 171 (near median missed width) | 1.01665 | [1.00576, 1.06079] | 0.05503 | 0.00576 |
| 480 (widest miss) | 1.04365 | [1.00065, 1.09302] | 0.09237 | 0.00065 |

Across all 98 misses, the median width is 0.05489 and the mean is 0.05648. Among the 502 covered intervals, the median width is 0.04638 and the mean is 0.04828. The missed intervals are therefore **slightly wider**, not systematically tighter. The failure is one-sided placement: their lower endpoints exceed 1. At one million people the intervals are much narrower than in the 100,000-person scenario (mean widths 0.04962 versus 0.21249), which makes the residual upward offset more consequential. Width and placement both matter.

## Deterministic fitted-population offset

The generating gamma survival model's value is 1. Applying the corrected production fitter to its exact expected counts, with no random sampling, gives **1.00461455**. The fitted frailty parameters are 3.7670 and 0.4720 rather than the generating 4.0 and 0.5; the fitted early baseline rates are 0.00098428 and 0.00099801 rather than the generating 0.001. Inverting the exact population hazards with the *generating* frailty parameters gives 1 to machine precision. Thus the population offset enters through the implemented fitting/adjustment procedure, not through random-number error.

The fitter fixes its baseline rate using early observed marginal hazards and then fits frailty with a Gompertz hazard approximation. Those early hazards already reflect frailty selection, so their baseline-rate calculation need not equal the generating rate. In the exact-population probe, changing the optimizer's initial frailty value from 0 to 20 changed the fitted values by at most 2.4e-9; a bad starting value is not an explanation. A diagnostic ablation that substitutes the generating baseline rate alone makes the population output 0.99211, overshooting in the opposite direction. The offset therefore cannot be attributed solely to one baseline-rate line; the fitting approximations interact. The ablation is a mechanistic probe, not a proposed repair.

The deterministic offset matters for coverage. If the very same retained percentile intervals are checked against 1.00461455 instead of the prespecified model value 1, 564/600 cover (94.0%). **Sixty-three of the 98 primary misses** are covered by that fitted-population target. This secondary check helps locate the cause; changing the target after seeing results would not validate the interval for the model-defined contrast. The observed mean point estimate, 1.00986, is also about 0.00524 above the deterministic fitted-population output, indicating additional finite-sample upward bias in this scenario.

## Boundary and skew diagnostics

In 202/600 original datasets, the smaller-frailty arm's fitted parameter is below 0.01, despite the generating value being 0.5. Those datasets have 45 percentile misses (77.7% coverage); the other 398 have 53 misses (86.7% coverage). The current fitter flags its *upper* frailty bound, not this near-zero lower-bound pattern, so all 600 still pass the declared operational gate. The lower-bound pattern is associated with worse coverage but does not by itself explain all misses.

The bootstrap median is above the original point estimate in 518/600 datasets, with mean difference +0.00175. The mean bootstrap estimate is +0.00348 above the point. The stronger +0.00822 displacement of the percentile-endpoint midpoint reflects asymmetric tails. Exploratory recentering of each interval midpoint on its point estimate would cover 576/600, but this is a post-run diagnostic, not an endorsed interval or fresh coverage validation.

For comparison, the 100,000-person unequal-frailty scenario has mean point estimate 1.02271 and mean percentile width 0.21249, with 23 one-sided misses out of 600. At 1,000,000 people the mean point error falls, but the interval becomes more than four times narrower; the residual offset and bootstrap asymmetry become large relative to interval width. The deterministic population offset is stable at about 1.00461455 across these two population sizes.

## Conclusion and limits

The retained evidence supports a **systematic fitting/target offset plus an upward-shifted, asymmetric percentile bootstrap distribution** as the proximate explanation. The exact-population offset accounts for much of the lost coverage; bootstrap asymmetry and finite-sample bias contribute further. This is not ordinary numerical roundoff or a missing-fit artifact. The exact share attributable to each fitting approximation is not identified by this analysis. Proposing a new estimator, diagnostic gate, or interval rule would require a separate design and fresh validation; the post-run counterfactual checks here do not change the prespecified verdict.

Evidence: [`diagnosis_unequal_large.json`](development/diagnosis_unequal_large.json) and its [read-only script](development/diagnose_large_coverage.py); [`population_fit_probe.json`](development/population_fit_probe.json) and its [probe](development/population_fit_probe.py); [`anchor_ablation.json`](development/anchor_ablation.json) and its [diagnostic ablation](development/anchor_ablation.py); the [prespecified final comparison](final_2026-11-10/assessment/comparison.json), [model check](development/model_checks.json), and [completion audit](final_2026-11-10/assessment/completion_audit.json).

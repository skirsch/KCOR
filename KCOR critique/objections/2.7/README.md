# 2.7 Contamination threshold

Evaluation finding N7 from the [November 21, 2026 review](<../../iterative review/evaluations/2/README.md>).

The study has a valid equal-baseline null. What its output fails to establish is a threshold at which the estimate first reverses direction. In the comorbidity sweep, 15 of 20 estimates already exceeded one at zero added contamination.

§3.3.2; [PR #5 target audit](../1.5/INITIAL_TARGET_AUDIT.md) · [checked counts](<../../iterative review/evaluations/2/repeat-2/B-follow.md>)

## Comorbidity sweep with 200 runs per level

The rerun extends the paper's original fitter, calibration, grid and seed sequence to 200 simulated datasets at each of the five levels. All 1,000 runs returned estimates. The first 20 per level reproduced the 100 archived KCOR estimates to within 0.000000084. Each interval below describes Monte Carlo uncertainty in a mean or difference of means. Different levels used independent random streams. The contrasts use pointwise 95% Welch intervals.

| Added weekly hazard | Mean KCOR (95% interval) | Difference from zero contamination (95% interval) |
|---:|---:|---:|
| 0 | 1.050 (1.034 to 1.066) | Reference |
| 0.0001 | 1.010 (0.993 to 1.027) | −0.040 (−0.063 to −0.017) |
| 0.0003 | 1.060 (1.045 to 1.076) | +0.010 (−0.012 to +0.032) |
| 0.001 | 1.229 (1.210 to 1.249) | +0.179 (+0.154 to +0.204) |
| 0.003 | 1.457 (1.435 to 1.479) | +0.407 (+0.380 to +0.434) |

At zero contamination the simulated mean was already above one. Among the tested levels, 0.001 was the smallest whose mean exceeded the zero-contamination mean with a 95% interval for the difference entirely above zero. The grid leaves the location of any onset between levels unresolved. These results apply to the fitter used for the published sweep. Iteration 3 applied this comparison in §3.3.2 and corrected the figure and table captions. The threshold objection is addressed in the manuscript. The investigation of low-variance recovery remains deferred.

[Reproduction script](comorbidity_sweep.py) · [All runs](comorbidity_200_runs.csv) · [Full-precision estimates and intervals](comorbidity_200_summary.csv)

These results use the historical simulation's fifth accumulating week as the reference. The [paired reference-week comparison](../2.8/README.md#iteration-3-paired-comparison-of-reference-weeks) reports the same datasets using the workbook's fourth-week reference, including intervals for the change in each contrast.

## Comparing fitters on identical simulated datasets

Each pair of simulated cohort histories was passed through the same study pipeline with three choices of fitter. The original fitter reproduced all 1,000 saved endpoints to within 0.00000001. Every fitter returned an endpoint for every dataset. The following are means of the anchored endpoint KCOR across 200 runs per level. Intervals in parentheses give 95% Monte Carlo confidence intervals for the mean under the current joint fitter.

| Added weekly hazard | Original fixed baseline | Revised fixed baseline | Revised joint fit |
|---:|---:|---:|---:|
| 0 | 1.050 | 1.051 | 1.064 (1.039 to 1.090) |
| 0.0001 | 1.010 | 1.012 | 1.088 (1.061 to 1.115) |
| 0.0003 | 1.060 | 1.062 | 1.145 (1.120 to 1.170) |
| 0.001 | 1.229 | 1.231 | 1.313 (1.282 to 1.343) |
| 0.003 | 1.457 | 1.460 | 1.543 (1.511 to 1.574) |

At zero contamination, the original and revised joint means differed by +0.0143 (paired 95% interval −0.0145 to +0.0431). The iteration 1 joint fit yielded 1.06417, compared with 1.06430 for the current joint fit. Holding the revised weekly calculation fixed and switching to joint fitting raised the mean by 0.0134 at zero contamination and 0.0826 at 0.0003.

With joint fitting, 0.0003 was the first tested level whose mean exceeded the zero-contamination mean with a pointwise 95% interval for the difference wholly above zero (+0.0806, interval +0.0447 to +0.1166). The mean at zero contamination itself exceeded one. Comparisons between fitters use paired intervals because they share the same death counts. Comparisons between contamination levels use independent samples under the original seed design.

[Comparison script](compare_fitters.py) · [Run-level estimates and fit status](fitter_comparison_runs.csv) · [Mean estimates and intervals](fitter_comparison_means.csv) · [Paired fitter differences](fitter_comparison_paired.csv)

## Why the zero-contamination mean exceeded one

The null diagnostic reproduced all 400 archived zero-contamination endpoints (200 datasets, two fitters). It held deaths fixed while changing the frailty supplied to the inversion and the early ratio used to anchor the endpoint. The population anchor is the ratio at the same week computed from the generator's exact death probabilities. The comparisons below describe the same 200 runs with 50,000 people per cohort.

| Frailty used in the inversion | Early anchor | Mean endpoint KCOR (95% Monte Carlo interval) |
|---|---|---:|
| Generating values for both cohorts | Population | 1.0013 (0.9976 to 1.0050) |
| Generating values for both cohorts | Sampled | 1.0207 (1.0031 to 1.0383) |
| Joint fit for A, generating value for B | Sampled | 1.0680 (1.0454 to 1.0907) |
| Generating value for A, joint fit for B | Sampled | 1.0171 (0.9961 to 1.0380) |
| Joint fits for both | Sampled | 1.0643 (1.0387 to 1.0899) |
| Original fits for both | Sampled | 1.0500 (1.0343 to 1.0657) |

On these datasets, replacing both generating frailties with joint estimates increased the mean by 0.0436 (paired 95% interval 0.0251 to 0.0622). The principal shift came from estimating numerator cohort A's frailty. Its generating variance was 0.5. The joint fits averaged 1.817, with 84/200 below 0.000001 and a 95th percentile of 7.11. Cohort B's generating variance was 4 and its mean estimate was 4.189. All 400 joint cohort fits returned status `ok`. With about 23–24 expected deaths per week in A, its true frailty multiplier on the baseline hazard changes only from 0.9986 to 0.9864 across the fitting window. The variance is constrained to be nonnegative. Large upward errors are possible while estimates below the true value can fall only as far as zero. Substituting each cohort's true variance in turn isolates the contribution of estimating it. This experiment does not assign the full fitting effect to the nonnegativity constraint alone.

With both frailties known, replacing the population anchor with the sampled one added 0.0194 in these 200 runs (paired 95% interval 0.0016 to 0.0372). The five-week anchor includes about 114 expected deaths in A and 113 in B. It enters the endpoint as a divisor. An independent 100,000-replicate check drew multinomial survivor counts at the skip, anchor and endpoint boundaries from the original generator's survival probabilities. These counts suffice for the cumulative inversion at known frailty. The sampled-anchor mean was 1.00872 and the population-anchor mean was 1.00041, a paired increase of 0.00831 (0.00747 to 0.00914). The particular 200-run anchor contribution therefore overstates the more precisely estimated average. All intervals in this section concern simulation means or their paired differences.

| People per cohort | Runs | Joint-fit mean KCOR (95% Monte Carlo interval) |
|---:|---:|---:|
| 50,000 | 200 | 1.0643 (1.0387 to 1.0899) |
| 500,000 | 200 | 1.0109 (1.0023 to 1.0195) |
| 5,000,000 | 200 | 1.0020 (0.9994 to 1.0047) |

Replacing sampled deaths and risk sets with their exact expectations yielded an anchored endpoint of 1.000000 to six decimals for the joint fitter and 1.00382 for the original fitter. Thus the pronounced excess is a finite-sample effect in this design. Increasing the number of simulated replicates estimates the bias more precisely. Increasing each simulated cohort's size improves the information available to each fit and anchor.

[Diagnostic script](diagnose_null.py) · [Replicate estimates, fitted frailties and noiseless checks](null_diagnostic_runs.csv)

The [low-frailty numerical investigation](low-frailty/README.md) extends this analysis to matched window lengths and cohort sizes, the exact-rate diagnostic, the stress-test outlier, the optimizer tolerance check and the first quiet windows of Table 8.

[Objection register](<../../iterative review/open-objections.md>) · [All objections](../../README.md)

## All-window joint fitter follow-up

[Matched replay and information-limit diagnostic](all-window-contamination/README.md): all five contamination levels at 50k, 500k and 5M people per cohort; residual finite-sample null bias explained by frailty resolution and noisy anchor. Experimental only.

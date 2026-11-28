# Table 8 historical replay and joint-fit trial

The six pooled point estimates and twelve confidence limits in Table 8 were independently reproduced to their printed precision on November 21, 2026. Git commit `bb2d68249e8bf77c8fb07e6c8336bc790a808a8a` contains the v4.3 program in `code/KCOR.py`, the input in `test/negative_control/data/KCOR_synthetic_neg_control.xlsx` and the corresponding `test/negative_control/out/KCOR_summary.log`. The input and log have Git blob IDs `f2dbb3d7a39d47a06821048de042285df46785ba` and `26d9e8c9eb7f55aa455609feb2fa3d2747b65b16`.

For the replay, extract the program and workbook from that commit into a temporary directory. With `SA_FINAL_KCOR_MIN=0`, run `python KCOR.py input.xlsx replay.xlsx KCOR_summary_neg_control.log`. The pooled rows for December 26, 2022 are in the output's `dose_pairs` sheet under `YearOfBirth=0`. The archived program computes the standard error of the pooled log ratio from weighted count-based variance terms, including the anchor, and uses multipliers of ±1.96 before exponentiating. This replay identifies the interval calculation. It does not estimate repeated-sample coverage.

For the independent feasibility trial, use the same archived input with the revised workbook entry point in ordinary mode, both enrollment sheets, the three Table 8 dose pairs and the existing Czech configuration (including forced alpha 1.18). Substitute the experimental initial joint fit from [the paired comparison](../1.3/paired-seed-fit/compare.py) in memory. The revised program used for the trial has SHA-256 `9baa948ec391b81ca43bd5a93db398bb582b24acf2cbaeedd710a27fc76ffa01`.

| Enrollment | Pseudo-dose comparison | v4.3 replay | Historical 95% interval | Revised joint-fit trial, reconstructed point |
|---|---|---:|---|---:|
| 2021_24 | 1 vs 0 | 1.0097 | [0.992, 1.027] | 1.0091 |
| 2021_24 | 2 vs 0 | 1.0213 | [1.000, 1.043] | 1.0318 |
| 2021_24 | 2 vs 1 | 1.0115 | [0.991, 1.033] | 1.0225 |
| 2022_06 | 1 vs 0 | 0.9858 | [0.970, 1.002] | 1.0503 |
| 2022_06 | 2 vs 0 | 1.0756 | [1.055, 1.097] | 1.0575 |
| 2022_06 | 2 vs 1 | 1.0911 | [1.070, 1.112] | 1.0068 |

The revised workbook emitted pooled rows only on October 7, 2024, the last input date. In `build_kcor_rows`, the pooled-row calculation runs after the loop over dates. For the trial column, take the weighted geometric mean of the three stratum-specific KCOR values from the workbook on December 26, 2022. Recompute the workbook weights for each stratum from the first four weeks as its mean positive weekly death rate across doses times its total person-time, then normalize. The same reconstruction agreed with all six pooled rows that the workbook did emit to within 1e-14. The joint run flagged insufficient signal in four of the nine contributing 2022_06 stratum and pseudo-dose fits. No joint-fit bootstrap intervals were produced. Comparing v4.3 to the revised pipeline does not isolate the effect of changing the initial fitter.

## Follow-up for the revised Table 8

The [person-bootstrap runner](../1.3/table8_bootstrap.py) reads the original workbook directly from Git. Within each enrollment it groups the eligible people represented in the age-shift control by age band, mutually exclusive vaccination subgroup, and week of death or survival to the end of follow-up. It verifies the weekly risk-set identity and recovers the other vaccination groups by subtracting the unvaccinated and two-dose groups from the total. A multinomial draw with these cell frequencies has the same distribution as sampling the enrolled people with replacement. Each sampled person contributes to the all-persons series and to their subgroup where applicable. The bootstrap conditions on the enrollment size represented in the table and uses separate draws for each enrollment.

Each draw refits all nine contributing strata using the full fitting windows and recomputes the three pooling weights. The configured insufficient-signal fallback is retained in every draw. The intervals are marginal 95% percentile intervals for the six ratios, conditional on the configuration (including alpha 1.18). They describe sampling variability in the empirical control. Its true adjusted mortality contrast is unknown, so the table cannot measure coverage.

From this directory, run `python table8_bootstrap.py --output table8_bootstrap.json`. The [results](../1.3/table8_bootstrap.json) include the percentile endpoints, number of eligible people, fit diagnostics and point estimates obtained by changing only the initial fitter back to the production early-rate fit.

The endpoint calculation matched the six original workbook estimates to relative tolerance 1e-10. As an independent check on resampled data, draw 1 for each enrollment was converted back into a weekly input workbook. After running the full joint-fit workbook entry point, pooling its nine stratum and dose-pair rows per enrollment at the reporting date recovered 1.005984, 1.002606 and 0.996642 for 2021_24, and 0.960863, 1.045520 and 1.088106 for 2022_06. All six agreed with the bootstrap runner to the same tolerance.

The wide lower tail in 2022_06 was also checked separately through the full workbook. Draw 29, the first with a 1 versus 0 estimate below 0.4, gave 0.233433, 0.227557 and 0.974827 for the three comparisons. The workbook and bootstrap runner agreed to relative tolerance 1e-10. All 17,982 stratum fits returned successfully in the 1,998 enrollment draws, including fits that returned the configured insufficient-signal fallback.

### Which fits contributed to the Table 8 diagnostics

The nine fits per enrollment come from three pseudo-doses within each of three overlapping constructions. These use people recorded with two doses, unvaccinated people, and everyone. Pseudo-doses 0, 1 and 2 label age groups within each construction. Only the nine contributing fits are counted here.

| Original 2022_06 construction | Pseudo-dose | Reason the fitter applied zero frailty variance |
|---|---:|---|
| People recorded with two doses | 1 | Fitted frailty variance reached the upper-bound threshold. |
| People recorded with two doses | 2 | Second first-pass gap increment was negative (−0.003177). |
| Unvaccinated people | 0 | Second first-pass gap increment was negative (−0.013778). |
| All persons | 1 | Fitted frailty variance reached the upper-bound threshold. |

Each flagged series retains its deaths, risk set and contribution to the pooled ratio. No contributing fit was flagged in the original 2021_24 sample. With resampling, at least one contributing fit was flagged in 271/999 draws in 2021_24 and 998/999 in 2022_06. These are counts of draws with any flag across the nine fits.

In draw 29 the contributing two-dose subgroup's pseudo-dose 0 fit was accepted. Its applied frailty variance increased from 1.4590 to 82.4044. Its adjusted cumulative hazard at the anchor increased from 0.01851 to 0.04252 and at the reporting date from 0.17732 to 3875.06. This illustrates the sensitivity of the inversion to a large accepted variance in the shared denominator. It accounts for the very low 1 vs 0 and 2 vs 0 ratios in this draw. The draw was selected to inspect a lower-tail result and does not establish how much of the entire interval width is due to this mechanism.

### Iteration 2 check against the low-frailty simulation

In the [current joint-fit replay](../2.4/verification.md), four of nine fitted variances in 2021 rounded to zero with status `ok`. In 2022 four were set to zero by the insufficient-signal fallback. Two of those fits reached the upper bound and two had a negative gap increment. Other 2022 fits returned variances of 11.90 and 24.56. The empirical control does not supply the underlying cohort frailties.

Using the same archived input, seed and [Table 8 runner](../1.3/table8_bootstrap.py) with the wave-entry edition, draw 21 was the first 2022 draw with a 1 vs 0 ratio below 0.4. The fitted variance for pseudo-dose 0 among people recorded with two doses rose from 1.45837 in the original sample to 68.43009. Replacing just this one variance with its original-sample value, retaining the draw's deaths, fitted baseline rates, other variances and pooling weights, moved 1 vs 0 from 0.24737 to 0.94182 and 2 vs 0 from 0.27689 to 1.05423. The 2 vs 1 ratio stayed at 1.11936. This check identifies the driver of those two low ratios in the selected draw. It does not measure its contribution to the full bootstrap interval. The [simulation diagnostic](../2.7/README.md#why-the-zero-contamination-mean-exceeded-one) separately tests recovery when the generating variance is known.

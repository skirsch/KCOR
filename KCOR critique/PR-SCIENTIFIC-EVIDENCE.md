# KCOR revision: detailed scientific evidence

**Companion to the [lead-author presentation](PR-PRESENTATION.md) | 25 November 2026**

This briefing presents the scientific results of the KCOR revision, from the most consequential mathematical and numerical findings to specific implementation and manuscript corrections. It distinguishes tested candidate changes that await lead-author approval from changes already incorporated into the revised edition.

## Executive summary

The investigation of the objections in PR #5 exposed three distinct numerical and statistical issues: the fitting procedure could preserve a poor first-window estimate, low mortality can make small frailty variances statistically difficult to identify, and the nonlinear wave correction uses an approximate survivor distribution during the wave.

The most important new result is that uncertainty in frailty variance need not translate into equally poor precision in KCOR itself. In the difficult synthetic multi-window test, a candidate all-window joint fit eliminated extreme endpoint estimates and brought the observed uncertainty close to a model-based Fisher information benchmark. A separate single-window contamination experiment shows where improved optimization alone cannot remove bias.

| Finding | Evidence | Status |
|---|---|---|
| Jointly fit the same common baseline scale and frailty variance across all quiet windows | 2022 Table 8 interval widths decreased 73–90%; stress-test median interval width fell from 213.056 to 0.315 | Experimental, approval pending |
| Frailty resolution is strongly governed by mortality, cohort size and quiet-window timing | Exact multi-window Fisher analysis, checked against simulations | New scientific contribution, not yet included in paper |
| The constraint at zero explains much of the low-frailty finite-sample upward bias | Predicted near-zero fractions 44.3%, 32.6%, 7.7%; observed 43.0%, 35.0%, 7.5% | Mechanism substantially explained; weak-information bias remains |
| Contamination experiment requires separation from its biased finite-sample null | New fitter null 1.0494 at 50k; 1.0013 at 5M per cohort | Higher-information companion proposed |
| Exact nonlinear-wave survivor inversion and survivor-law fit | Exact known-truth recovery of rate, frailty and KCOR; sampled main mean 1.00093 vs 0.98982; coverage unchanged | Experimental, approval pending |
| Mortality before the first quiet window | Exact proportional entry recovery and nonlinear survivor-law candidate substantially improve marked early-wave cases | Experimental; early anchor validity remains conditional on wave labels |

The thresholds and intervals are statistical properties of specific models and experiments, not evidence of a treatment effect in Czech data.

## 1. All-window joint fitting and uncertainty coverage

In the first-window joint fitter, the baseline Gompertz scale k and frailty variance theta are estimated together in the initial quiet interval. The subsequent pooled fit holds k fixed while updating theta. This can propagate an unstable initial estimate into later reconstructed gap exposures.

A constructed 50,000-person stress test exposed the numerical mechanism. A group with true variance 0.5 had an estimated variance of 91.841, producing an extremely large inferred gap exposure and a pooled KCOR of 812.675 when the generating target was 1.

The isolated candidate fits the same common k and theta across all quiet windows and recalculates the gap offsets inside the objective for each candidate pair. It does not introduce window-specific baseline scales or different core frailty models. A second optimizer starting value, selected by residual sum of squares, avoids a remaining bad solution branch. Neither a first-window-only update nor a researcher's chosen frailty value is imposed on the final solution.

### Matched known-truth stress comparison

50 independent datasets; 99 person-bootstrap draws per dataset; 50,000 people in each full population; 22 quiet weeks split across two windows; known target KCOR=1.

| Metric | Previous joint-seed fit | All-window fit, two starts |
|---|---:|---:|
| Median 95% interval width | 213.056 | 0.315 |
| Maximum 95% interval width | 1,629.002 | 0.514 |
| Endpoint RMSE against 1 | 162.635 | 0.0857 |
| Empirical 95% coverage | 50/50 | 48/50 |
| Failed subgroup fits | 177 | 0 |
| Mean KCOR [95% MC interval] | 43.506 [-1.560,88.573] | 1.037 [1.015,1.059] |
| Pathological dataset KCOR [95% bootstrap interval] | 812.675 [0.848,744.878] | 1.016 [0.875,1.131] |
| Offending subgroup theta, true 0.5 [95% bootstrap interval] | 91.841 [0,91.639] | 0.462 [0,2.227] |

The extremely wide intervals of the previous fitter explain its high coverage. The candidate reduces the catastrophic errors; it still has positive mean bias in this difficult design. The 50 datasets do not establish general calibration.

### Matched Table 8 outcomes

Same archived data and same 999 person-bootstrap draws per enrollment; all values include 95% percentile intervals.

| Enrollment | Pseudo-dose pair | Previous joint fit | All-window candidate |
|---|---|---|---|
| 2021 | 1/0 | 1.008 [0.956,1.058] | 1.013 [0.956,1.068] |
| 2021 | 2/0 | 1.031 [0.974,1.094] | 1.033 [0.971,1.106] |
| 2021 | 2/1 | 1.023 [0.964,1.089] | 1.020 [0.958,1.094] |
| 2022 | 1/0 | 1.030 [0.322,1.236] | 1.012 [0.968,1.055] |
| 2022 | 2/0 | 1.072 [0.341,1.193] | 1.028 [0.985,1.079] |
| 2022 | 2/1 | 1.041 [0.847,1.213] | 1.016 [0.970,1.068] |

2022 interval widths decreased by 90%, 89% and 73%; 2021 widths increased by roughly 9–12%. All six intervals contain one. There were no insufficient-signal flags or failed fits in the candidate Table 8 replay. Narrower intervals alone do not prove calibration; the empirical control does not establish exact equivalence. Real Czech true frailty remains unknown.

**Recommendation:** Favor adoption of the candidate, conditional on lead-author approval and broader pipeline validation. The candidate is not yet the official method.

[Full all-window experiment, code, draws and parameter estimates](objections/2.7/low-frailty/all-window-joint/README.md)


### Broader coverage validation (objection 1.3)

The candidate was tested on the same 100 independent one-million-person datasets per scenario and 199 paired bootstrap draws as the original multi-window study. The previous short-window stress was retained, and 400 additional original single-window null datasets were tested with 99 person draws each. Three actual workbook entry-point checks reproduced the candidate's fast array replay to numerical precision.

| Scenario | Candidate coverage [exact 95% CI] | Mean KCOR [95% MC CI] | Mean interval width [95% MC CI] |
|---|---|---|---|
| No wave, 1M total | 97/100 [91.5%,99.4%] | 0.9996 [0.9955,1.0037] | 0.0847 [0.0835,0.0860] |
| Nonlinear wave, 1M total | 90/100 [82.4%,95.1%] | 0.9956 [0.9907,1.0006] | 0.0877 [0.0864,0.0891] |
| Short-window stress, 50k total | 48/50 [86.3%,99.5%] | 1.0368 [1.0145,1.0590] | 0.3250 [0.3044,0.3455] |
| Single-window null, 50k/cohort | 187/200 [89.1%,96.5%] | 1.0494 [1.0305,1.0684] | 0.5494 [0.5366,0.5621] |
| Single-window null, 5M/cohort | 192/200 [92.3%,98.3%] | 1.0013 [0.9993,1.0033] | 0.0539 [0.0532,0.0546] |

The old fixed / previous joint / candidate coverages are 78% / 94% / 97% without a wave and 88% / 92% / 90% for the nonlinear wave. The candidate main-study widths are about 13–14% wider than those of the previous joint fit, with no failed fits or insufficient-signal flags. Its no-wave expected-count KCOR is exactly 1; the nonlinear-wave expected-count result is 0.994723. All five coverage confidence intervals contain 95%, marginally for the nonlinear-wave design. Coverage remains conditional on the tested null designs, supplied alpha and existing correction; the exact in-wave trial is not combined with this fitter. The small single-window cohort retains a biased mean despite reasonable coverage. No claim of universal calibration is made.

[Complete coverage validation, paired results, actual workbook verification and draws](objections/1.3/all-window-coverage/README.md)

## 2. Theoretical limits of frailty and KCOR precision

For one contiguous quiet window with low cumulative hazard and approximately constant baseline weekly hazard, the local information after accounting for unknown baseline scale gives the approximation

$$
\operatorname{SE}(\hat\theta)\simeq
\sqrt{\frac{12}{N k^3 L(L^2-1)}}.
$$

Here N is the number of people, k the weekly baseline hazard, and L the number of quiet weeks. In this weak-hazard limit, halving k requires roughly eight times the cohort size for the same frailty precision. Doubling quiet-window duration can have a similar impact to increasing N eightfold. It is the rate and cumulative exposure, not simply the number of individuals, that determine the available information.

The more general analysis uses all configured quiet windows, uncertain gap exposures, the observed survival transitions through gaps and exact nonlinear-wave survival with supplied alpha. For each interval, integrated observed hazard q, death probability p=1-exp(-q), and expected risk Y give

$$
I(\beta)=\sum_i Y_i\frac{1-p_i}{p_i}
\nabla q_i\nabla q_i^{\mathsf T}.
$$

The parameter vector includes log k, theta and unknown gap exposures. The local variance of the log KCOR endpoint follows from the corresponding gradient and inverse information. This is a model-conditional regular bound, not a guarantee of finite-sample confidence coverage. Near theta=0, likelihood profiles help interpret the boundary. The calculations fix alpha; additional alpha uncertainty would reduce the available information. Overlapping Table 8 pooling strata need a joint dependence model for a strict pooled information bound.

### Theoretical versus observed uncertainty

In the same difficult synthetic multi-window design, the exact-model theoretical pooled KCOR 95% interval at the true ratio of 1 is **[0.877,1.140]** (width 0.2634). The all-window empirical median bootstrap width was 0.3146. The theoretical log-ratio standard error was 0.0670; the observed log-point standard deviation was 0.0748. This suggests that the candidate approaches the attainable endpoint precision in this particular design even when individual frailty variance estimates remain poor.

Actual Czech schedules were evaluated under reference theta=0.5. Example theoretical local 95% half-widths for the 2021 dose-0 cohorts are 0.364 (1940s; N=151,775), 1.023 (1950s; N=280,822), and 4.259 (1960s; N=423,254). The implied population for +/-50% frailty resolution rises from about 321k to 4.7M to 123M, conditional on those schedules, rates and supplied alpha=1.18. These are hypothetical low-variance resolution limits, not empirical estimates of Czech frailty.

**Proposed contribution:** A mathematical framework for knowing when frailty resolution is intrinsically weak, and how this relates to resolution of KCOR itself.

[Full derivation, numerical verification, cohort tables and observation-duration planning grid](objections/2.7/low-frailty/theoretical-limits/README.md)

## 3. Why the original contamination experiment had KCOR above one

The earlier 200-run experiment averaged KCOR 1.050 with zero added weekly mortality under the original fitter. The two cohorts have equal generating baseline rates, so the target is 1. Neither shifting the intended contamination nor supplying true parameters to the normal fitter is a solution to the statistical question.

The all-window replay preserves the original single contiguous quiet window, early fitting anchors, contamination and fifth accumulating week normalization. It returns 1.0494 at the original population size. The improvement compared with the previous joint-seed fitter is real, but the remaining weak-information bias persists. In a single-window design there are no additional quiet windows from which to recover information.

### Size dependence and the full contamination sweep

The following are means across 200 independent datasets per level with pointwise 95% Monte Carlo intervals. These are intervals for the mean across simulations, not individual patient-bootstrap intervals.

| Added hazard/week | 50k people/cohort | 500k/cohort | 5M/cohort |
|---:|---|---|---|
| 0 | 1.0494 [1.0305,1.0684] | 1.0090 [1.0027,1.0153] | 1.0013 [0.9993,1.0033] |
| 0.0001 | 1.0415 [1.0230,1.0601] | 1.0270 [1.0211,1.0328] | 1.0210 [1.0191,1.0229] |
| 0.0003 | 1.0853 [1.0671,1.1035] | 1.0836 [1.0771,1.0900] | 1.0878 [1.0858,1.0899] |
| 0.001 | 1.2394 [1.2169,1.2619] | 1.2492 [1.2423,1.2561] | 1.2537 [1.2513,1.2560] |
| 0.003 | 1.4676 [1.4428,1.4923] | 1.4887 [1.4810,1.4963] | 1.4859 [1.4836,1.4882] |

Exact expected deaths give null KCOR=1 to numerical precision. With five million per cohort, the observed null mean is consistent with one and the smallest tested positive injection differs from this null by +0.01969 [0.01698,0.02240]. That establishes direction at tested levels; it does not locate a universal threshold.

### Boundary-bias mechanism

If an approximately Gaussian unconstrained estimate is X~N(theta,sigma^2), clipping at zero gives

$$
E[\max(0,X)] =
\theta\Phi(\theta/\sigma)+\sigma\phi(\theta/\sigma).
$$

Using the model-based local Fisher standard error and true theta=0.5, this predicts the following:

| People per cohort | Theoretical SE | Predicted mean | Fitted mean [95% MC interval] | Predicted at zero | Observed near zero |
|---|---:|---:|---|---:|---:|
| 50k | 3.503 | 1.662 | 1.546 [1.251,1.841] | 44.3% | 43.0% |
| 500k | 1.108 | 0.736 | 0.717 [0.607,0.828] | 32.6% | 35.0% |
| 5M | 0.350 | 0.512 | 0.516 [0.469,0.563] | 7.7% | 7.5% |

The correspondence is strong evidence for finite-sample boundary bias, not proof that the clipped Gaussian approximation exactly represents the constrained nonlinear fitter. The sampled early normalization ratio also contributes: in an independent 100,000-replicate known-frailty check, sampling the anchor increased the mean ratio by 0.00831 [0.00747,0.00914]. At 50k, estimating both frailties rather than supplying generating values added +0.02877 [0.01265,0.04489] on the same 200 sampled-anchor datasets.

**Recommendation:** Retain the original 50k stress case and consider adding the higher-information 5M experiment to demonstrate the contamination effect. Do not present increasing N as changing KCOR or eliminating the finite-sample limitation in real data. Agreement with the true null is not a criterion for selecting or deleting the original test.

[Complete contamination replay, paired decompositions, information calculation and all run-level results](objections/2.7/all-window-contamination/README.md)

## 4. Nonlinear-wave inversion and mortality before the initial quiet period

For alpha=1 the original gamma inversion is exact. With alpha different from one, the surviving population generally ceases to follow a gamma distribution. The existing in-wave attribution approximates its moments and cumulative scaling. An isolated trial solves the survival equation numerically week by week, then carries the recovered exposures into the post-wave inversion.

The trial recovered exact known-input curves in 28 cases. With the same initial k and theta, the 2021 unvaccinated pseudo-dose-0 wave-week-24 survivor variance changed from the proxy 0.887 [0.692,1.074] to 1.044 [0.782,1.323]; estimated cumulative wave hazard changed from 0.04333 [0.03731,0.04860] to 0.04433 [0.03897,0.04950]. The largest of the six tested Table 8 joint-fit point changes was approximately 0.00054 at supplied alpha=1.18. These are comparisons against the earlier first-window joint fitter, not against the new all-window candidate.

The tested weekly inversion retained prior wave exposure, assigned no additional wave exposure and allowed a smaller baseline increment for an observed weekly deficit relative to the fitted baseline. This preserves observed mortality but changes its attribution. The numerical inversion and this attribution convention remain candidates for lead-author approval.

[Exact weekly wave trial and week-by-week comparisons](objections/1.2/weekly-survival/README.md)

### Combined all-window fitter and weekly survival inversion

On the same nonlinear-wave simulation datasets, combining the exact weekly inversion with the all-window fitter leaves coverage unchanged. The main scenario uses 100 independent one-million-person datasets and 199 bootstrap draws; the stress scenario uses 50 independent 50,000-person datasets and 99 draws.

| Scenario | Approximate wave: coverage and mean KCOR [95% MC CI] | Exact weekly wave: coverage and mean KCOR [95% MC CI] |
|---|---|---|
| Main | 90/100; 0.99565 [0.99072,1.00057] | 90/100; 0.98982 [0.98493,0.99471] |
| Stress | 48/50; 1.03676 [1.01453,1.05899] | 48/50; 1.03068 [1.00878,1.05258] |

Mean interval widths change from 0.08773 to 0.08708 in the main design and 0.32496 to 0.31934 in stress. Coverage confidence intervals are [82.4%,95.1%] and [86.3%,99.5%], respectively. Every fitted frailty and fit diagnostic is identical between wave corrections.

**Why the exact correction alone does not remove bias:** the fitted quiet-window model still uses a proportional gamma survivor law after nonlinear selection. A noiseless counterfactual input test isolates the error.

| Inputs (main design, true KCOR=1) | Approximate wave | Exact weekly wave |
|---|---:|---:|
| Fitted baseline and frailty | 0.994723 | 0.988967 |
| Fitted baseline, generating frailty | 1.005546 | 0.999943 |
| Generating baseline, fitted frailty | 0.994792 | 0.989023 |
| Both generating values | 1.005615 | 1.000000 |

The exact inversion recovers the correct result conditional on the correct inputs. The approximate wave equation partially cancels the fitted frailty error in this specific design. This is evidence for a consistent survivor model in the upstream fit, not for replacing estimated parameters with generating parameters.

All six matched Table 8 intervals contain one. The new largest point movement is +0.002614, in the 2022 comparison 2/0, from 1.02766 [0.98506,1.07906] to 1.03028 [0.98735,1.08406]. The maximum reconciliation error of the weekly solver is below 3.8e-15; all six original points match ordinary workbook processing. With the all-window fitter, deficits occur in 0.54% of evaluated 2021 and 14.06% of evaluated 2022 bootstrap subgroup-weeks. Exact inversion and deficit attribution remain experimental pending approval.

[Combined validation, all six intervals and known-truth parameter substitutions](objections/1.2/all-window-weekly-survival/README.md)

### Mortality preceding the first quiet window

A prior proportional shock changes the rate identifiable from quiet-window mortality. With cumulative proportional excess exposure a, the first quiet interval may identify r=k/(1+theta a) rather than k. Under the proportional gamma model, observed survival to the beginning of that window gives

$$
k=\frac{r\exp(\theta H_{\mathrm{entry}})}{1+\theta r F_{\mathrm{entry}}}.
$$

The entry-aware all-window candidate carries this exposure into the first window, alongside the existing offsets for subsequent windows. In the noiseless proportional-shock test the unmodified all-window fit gives KCOR 1.207432 and the entry-aware fit gives exactly 1, recovering both generating frailty variances and k. In 200 matched samples its mean moves from 1.20386 [1.19425,1.21347] to 0.99774 [0.99321,1.00228], improving absolute endpoint error in 198/200 datasets. This establishes the proportional-shock correction independently of sampling variability. The 2022 Table 8 values change by at most 0.000253, with essentially identical intervals. The first-entry correction is a candidate for lead-author approval.

An additional diagnostic shows the importance of the four early rows used to fit the baseline. If these rows contain wave mortality, treating them as quiet baseline observations can distort the rate and frailty. This is distinct from the KCOR normalization anchor. Excluding nonquiet rows solely from the fitting residual restores a deterministic proportional early-pulse null from 1.118 to 1.000. In the sampled partial nonlinear-wave null, the mean moves from 1.14566 [1.14045,1.15087] to 0.99321 [0.98188,1.00453]. Removing clean early rows, however, reduced precision in a separate proportional-shock control. In Table 8 the corresponding 2022 sensitivity increases the interval widths by 30-35%; all intervals still contain one. There is no general recommendation to discard the early fitting anchors: their suitability depends on the baseline model, and the interpretation of mixed weeks remains separate from the proportional entry identity.

[Known-truth controls, matched Table 8 and fitted parameter comparisons](objections/3.2/all-window-entry/README.md)

### A consistent survivor model for fitting and wave inversion

The remaining nonlinear fitting error has now been corrected experimentally by fitting integrated quiet-week hazards directly from the same survival law used during wave inversion,

$$
S(B,W)=E[\exp(-ZB-Z^\alpha W)].
$$

For every candidate pair (k, theta), reconstruct the marked wave's cumulative exposures and subsequent proportional gaps, then predict quiet mortality using the actual survivor distribution. Marked wave weeks contribute to exposure reconstruction rather than quiet-baseline residuals, including when they overlap the early fitting anchors. The KCOR normalization and all supplied dates remain unchanged.

**Noiseless recovery:** both pooled synthetic designs recover the generating k and all four frailty variances (4, 0.5, 4, 0.5) to numerical precision; KCOR is exactly 1, versus 0.988967 and 0.990133 with the previous gamma working fit and exact wave inversion. The proportional route is unchanged.

| Main cohort | True theta | Gamma working mean [95% MC CI] | Exact survivor mean [95% MC CI] |
|---|---:|---|---|
| 1940 dose 0 | 4 | 4.263 [4.232,4.295] | 4.015 [3.985,4.045] |
| 1940 dose 1 | 0.5 | 0.543 [0.516,0.570] | 0.506 [0.481,0.532] |
| 1950 dose 0 | 4 | 4.345 [4.258,4.431] | 4.057 [3.976,4.138] |
| 1950 dose 1 | 0.5 | 0.517 [0.456,0.578] | 0.481 [0.424,0.538] |

Full 199-draw matched main coverage remains 90/100 [82.4%,95.1%]; mean KCOR improves from 0.98982 [0.98493,0.99471] to 1.00093 [0.99596,1.00591]. Mean interval width changes from 0.08708 to 0.08853. The 50k short-window stress coverage remains 48/50 [86.3%,99.5%], mean KCOR changes from 1.03068 [1.00878,1.05258] to 1.04019 [1.01789,1.06249]. Thus mathematical consistency improves deterministic accuracy without resolving all statistical limitations.

In 200 independent sampled short first-window datasets with all nonlinear pulses marked, the mean moves from 1.21131 [1.20543,1.21718] to 1.00399 [0.99295,1.01503]; 197/200 improve absolute error. With partially marked waves, the mean moves from 1.20438 [1.19857,1.21019] to 0.99987 [0.98785,1.01188] (190/200 improve). A long-window case whose marked wave dates also cover clean early fitting rows loses precision: its mean improves from 1.02220 [1.01787,1.02653] to 1.01018 [0.99159,1.02877], yet only 39/200 improve individual absolute error. The fitter cannot identify unmarked nonlinear exposures from mortality counts alone. The evidence supports a consistent model with known wave dates rather than retrospective window selection.

| Enrollment | Pair | Gamma working + exact wave [95% CI] | Exact survivor + exact wave [95% CI] |
|---|---|---|---|
| 2021 | 1/0 | 1.01279 [0.95552,1.06823] | 1.01277 [0.95551,1.06827] |
| 2021 | 2/0 | 1.03310 [0.97032,1.10552] | 1.03309 [0.97038,1.10560] |
| 2021 | 2/1 | 1.02005 [0.95768,1.09403] | 1.02006 [0.95776,1.09408] |
| 2022 | 1/0 | 1.01195 [0.96848,1.05587] | 1.01255 [0.95757,1.07539] |
| 2022 | 2/0 | 1.03028 [0.98735,1.08406] | 1.05566 [0.99627,1.12107] |
| 2022 | 2/1 | 1.01811 [0.97103,1.07380] | 1.04257 [0.97537,1.10948] |

All six matched Table 8 intervals contain 1. The 2022 widths increase 29-35%, consistent with the loss of early baseline fit rows within the marked wave, while the 2021 intervals remain essentially unchanged. The six original endpoints match ordinary workbook processing and no failed fits or insufficient-signal flags occur across 18,000 subgroup fits. True Czech frailty and nonlinear exposure are unknown.

**Recommendation:** consider the exact survivor-law fitter and exact weekly inversion as a mathematically coherent pair, subject to lead-author approval including wave-row eligibility and the deficit attribution convention. The weak-information bias, marginal main coverage and uncertainty in wave labeling remain explicit limitations.

[Complete survivor-law fit study, all matched coverage and Table 8 fitted-parameter intervals](objections/3.2/nonlinear-survivor-fit/README.md)

### Identification of the nonlinear exponent

The original real Czech pooled analysis was reproduced exactly using its historical workbook and configuration (three birth decades, doses 0-3 and five overlapping enrollments). The source registry contains 12.60 million data records. The original selected cohorts comprise 10.36 million overlapping enrollment memberships, not independent people. Forty-eight cohorts pass initial eligibility, but the original positive observed excess filter retains only 62 cohort-week rows across eight cohorts. The pairwise and collapse objectives use 84 pairs and 41 points, respectively.

| Model | Pairwise alpha minimum | Collapse alpha minimum | Bootstrap stability range (both) |
|---|---:|---:|---|
| Original | 1.190 | 1.185 | [1.00, 1.30] |
| All-window joint fit | 1.000 (boundary) | 1.000 (boundary) | [1.00, 1.30] |
| Exact nonlinear survivor fit and moment | 1.000 (boundary) | 1.000 (boundary) | [1.00, 1.30] |

The historical curvature checks fail. With the exact model, all leave-one-cohort-out minima are 1.00 on the original grid. On a wider diagnostic grid extending to 0.70, both exact-model minima move to the new lower boundary, including after exclusion of two insufficient-signal cohorts. The bootstrap stability ranges are not confidence intervals; the original cohorts overlap. Improved fitting does not identify the nonlinear exponent. The historical empirical objective uses raw excess above dose 0 rather than independently established wave exposure and does not supply the external intervention-effect scale specified for identification in the method. Therefore the replay does not establish a physical exponent at or below one. The value 1.18 remains sensitivity-only.

The exact survivor law also makes later quiet mortality potentially informative through survivor selection. A separate ideal model-conditional Fisher diagnostic, using the more recent restricted dose-0/2 selection, gives local alpha standard errors about 0.92-1.35 in the four 2021 enrollments and extremely weak information in 2022. Neither this temporal-shape diagnostic nor the historical empirical objective replay supports identification from the available design.

[Historical pooled replay, numerical tables and profiles](objections/1.4/alpha-pooled-replay/README.md) | [Temporal alpha information diagnostic](objections/1.4/alpha-information/README.md)

## 5. Implemented mathematical, software and validation corrections

The following changes are already in the revised local edition. The experimental fitter and full in-wave inversion described above have not been adopted. The original model and the earlier editions remain available.

| Objection | Original issue | Revision |
|---|---|---|
| [1.2](objections/1.2/README.md) | Nonlinear wave equation could apply extra adjustment or misrepresent continuing survivor selection. | Preserve ordinary inversion at alpha=1, condition post-wave inversion on attributed exit exposure, and correct the initial survivor moment for the opening of the wave. |
| [1.3](objections/1.3/README.md) | Resampling weekly blocks did not preserve person histories; ordinary Monte Carlo processing had sample-clock and diagnostic discrepancies. | Resample people, retain death weeks, reconstruct risk sets and refit each draw; repair the Monte Carlo processing and diagnostic summaries. |
| [1.6](objections/1.6/README.md), [2.2](objections/2.2/README.md) | The fitter used instantaneous model hazard for an integrated weekly observation, and the inversion used a discrete approximation. | Use exact integrated weekly hazards and weekly gamma inversion, including the baseline reference in the optional wave calculation. |
| [2.3](objections/2.3/README.md) | The final week of a gap was omitted from the offset. | Read the reconstructed and expected cumulative exposures at the opening of the next quiet week. |
| [2.4](objections/2.4/README.md) | The earlier documented convention assigned first-wave-week deaths to the reference instead of evaluating them as part of the wave. | Include the entire modeled first wave week in the correction, as approved. |
| [3.1](objections/3.1/README.md) | A true zero additional gap exposure could trigger a fallback erasing a correct positive frailty fit. | Continue the fit with zero added gap exposure where the offset is nonpositive; flag material negative offsets separately rather than changing the KCOR calculation. |
| [1.4](objections/1.4/README.md) | Historical Table 8 interval provenance was uncertain. | Reproduce the estimates and calculate matched person-bootstrap intervals for all six comparisons. |
| [2.5](objections/2.5/README.md) | The old quiet-window scan used a different fitter and excluded failures from the pass-rate denominator. | Refit all 78 attempted placements and count every attempt. All 78 pass the revised diagnostics; the fitted variances still vary across windows. |
| [2.6](objections/2.6/README.md), [3.3](objections/3.3/README.md) | Positive-control generator and described individual-level intervention did not match; historical endpoints could be reconstructed without fitting. | Describe the actual cohort-level injection and verify Table 9 using the fitted workbook pipeline. |
| [2.7](objections/2.7/README.md), [2.8](objections/2.8/README.md) | The contamination threshold claim was unsupported because the zero-addition mean was already above 1. The anchor convention was also inconsistent. | Retain the experiment's intended fifth-week normalization and compare added hazard against the observed zero-addition mean. Report the fourth-week sensitivity separately. |

## 6. Manuscript clarifications and smaller rephrasings

These edits tighten mathematical interpretation and align the text with the implemented method and supporting evidence. They do not introduce a new KCOR algorithm.

| Objection | Text or presentation change |
|---|---|
| [1.1](objections/1.1/README.md) | Qualify the claims that contamination must be diagnostically visible or that distortion is necessarily conservative. Prognosis-dependent crossover does not have a guaranteed bias direction. |
| [1.4](objections/1.4/README.md) | State the independent evidence needed to distinguish a treatment-induced change in frailty from pre-existing selection. |
| [1.5](objections/1.5/README.md), [2.1](objections/2.1/README.md) | Specify that the Cox comparison's treatment-effect null is conditional on fixed frailty; distinguish that benchmark from the cumulative-hazard target. Disclose simulator-supplied frailties and add pointwise interval bars. |
| [1.6](objections/1.6/README.md) | Distinguish scaling observed cohort hazards from scaling individual baseline hazards. Identify frailty using the level and derivative under the fixed Gompertz slope, rather than equating positive frailty with any cumulative-hazard curvature. The diagnostic reference follows the Gompertz cumulative shape. |
| [2.6](objections/2.6/README.md) | Specify the treated cohort's own unperturbed survivor hazard as the reference for the original positive-control injection. |
| [2.8](objections/2.8/README.md) | Clarify that normalization uses one prespecified cumulative KCOR value at the reference week. |
| [2.9](objections/2.9/README.md) | Distinguish fitted comparison groups from person-resampling strata and explain Table 8 subgroup pooling. |
| [2.10](objections/2.10/README.md) | Disclose the empirical illustration selection criteria and use the term configured quiet windows for the worked example. Retain prespecification in the general method. |
| [Supporting evidence disclosures](<iterative review/open-objections.md#supporting-tests-figures-and-study-provenance>) | Disclose expected rather than sampled deaths and the exact baseline rate in the historical multiwave example; identify historical procedures in the older figure and simulation grid; replace the deterministic Poisson pilot's empty coverage table and interpretation with an accurate description of that pilot. |

The individual objection records identify exact manuscript passages and supporting tests. The cumulative redline shows the actual edits, with links to their objections.

**Manuscript evidence:** [Clean revised manuscript](<iterative review/KCOR-clean.pdf>) ? [Cumulative redline](<iterative review/KCOR-redline.pdf>) ? [Original PR #5 evidence](PR5-evidence.md) ? [Numbered corrections](revision-records.md).

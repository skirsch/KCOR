# Objection 3.2 · Survival at entry to the first quiet window

This is an experiment on an alternative first-window calculation. The reference is the joint fitter from iteration 3 after 3.1. All method, paper and PDF changes require a separate decision.

## What the test changes

Let the first quiet week start at time s after the stabilization skip. The observed cumulative hazard H ends at the **start** of that week and excludes deaths within it. Write F(s) = (exp(gamma s) - 1)/gamma. The existing joint fit returns a quiet-window rate r and seed variance theta. For additive pre-window excess exposure a, the quiet rates identify r = k/(1 + theta a), and observed entry survival gives exp(theta H) = 1 + theta [kF(s) + a]. Together they give

**k = r exp(theta H) / [1 + theta rF(s)].**

The trial uses this conversion when there are modeled weeks before the first quiet window. It keeps the existing rate bounds, applies the nonnegative-excess convention (a negative inferred excess leaves the seed rate at r), and includes the first quiet boundary among the boundaries used to reconstruct gap offsets for the pooled refit. This is an additive-model identity. Using it when alpha = 1.18 tests an approximation. The trial uses the existing wave attribution and the supplied exponent. It does not fit an exponent or assign individual deaths to causes.

## Comparisons

- Synthetic populations start with mean-one gamma frailty. Observed survival is generated as the population average of exp[-zB(t) - z^alpha W(t)]. The fitter receives all-cause weekly hazards, quiet-window dates and, for sampling experiments, weekly deaths. The subsequent KCOR wave correction receives the stated wave-window dates and exponent. Generating rates, frailties and pulse magnitudes are withheld from both fitters. The known endpoint is the ratio of the cohorts' growth in B + W from the end of modeled week 3 to the end of modeled week 74.
- The longer seed starts in modeled week 12 and ends in week 40. The shorter seed spans weeks 14–17 plus a later fitting window at weeks 50–70. The partially labeled wave is active through week 11, while the correction is supplied only weeks 0–4. The unequal-rate control raises the first cohort's baseline rate by 25%. Each stochastic design has 200 independent datasets of 50,000 people per cohort. Both fitters use identical weekly counts within each dataset.
- Table 8 uses the original Czech input and the same 999 person-bootstrap draws per enrollment as 3.1. The comparison preserves the anchor, pooling weights, quiet and wave dates, and alpha = 1.18. Each interval is the 2.5th to 97.5th percentile of all 999 draws, including configured fallbacks. The interval for a paired change is calculated from differences on matching draws. The 2021 control starts its quiet fit at the modeled origin. The 2022 fit has 14 modeled weeks before the first quiet week.

From the repository root, run the three Python 3.11 scripts in this directory. [Known-truth tests](synthetic.py) · [Table 8 replay](experiment.py) · [2022 baseline and wave attribution](components.py). Run the attribution after the replay to cross-check the endpoint for every draw.

## Table 8 with matched person-bootstrap intervals

| Enrollment | Pseudo-dose | Current [95% interval] | Trial [95% interval] | Width, current / trial | Paired change [95% interval] |
|---|---|---:|---:|---:|---:|
| 2021_24 | 1 vs 0 | 1.00832 [0.95637, 1.05843] | 1.00832 [0.95637, 1.05843] | 0.10206 / 0.10206 | 0.00000 [0.00000, 0.00000] |
| 2021_24 | 2 vs 0 | 1.03124 [0.97388, 1.09443] | 1.03124 [0.97388, 1.09443] | 0.12055 / 0.12055 | 0.00000 [0.00000, 0.00000] |
| 2021_24 | 2 vs 1 | 1.02273 [0.96398, 1.08885] | 1.02273 [0.96398, 1.08885] | 0.12488 / 0.12488 | 0.00000 [0.00000, 0.00000] |
| 2022_06 | 1 vs 0 | 1.03016 [0.32192, 1.23575] | 1.02941 [0.75831, 1.23510] | 0.91384 / 0.47678 | -0.00075 [-0.07556, 0.00295] |
| 2022_06 | 2 vs 0 | 1.07242 [0.34138, 1.19281] | 1.07156 [0.82556, 1.19239] | 0.85143 / 0.36683 | -0.00086 [-0.08272, 0.00204] |
| 2022_06 | 2 vs 1 | 1.04102 [0.84718, 1.21333] | 1.04095 [0.84625, 1.21335] | 0.36614 / 0.36710 | -0.00008 [-0.00234, 0.00171] |

All six trial intervals include 1. Every 2021 draw agreed with its current counterpart. The original 2022 sample had two insufficient-signal fits in each arm. The number of 2022 draws with any insufficient-signal fit went from 932 to 934 of 999. No trial fit reported a numerical failure.

The 2022 trials for 1 vs 0 and 2 vs 0 had 25 and 24 draws below 0.5, respectively, compared with 39 and 38 under the current fit. The nominal 2.5th percentile falls between the 25th and 26th ordered value. For 1 vs 0 the trial 2nd, 2.5th and 3rd percentiles were 0.242, 0.758 and 0.814. For 2 vs 0 they were 0.253, 0.826 and 0.869. The reported 95% widths are sensitive to the lower-tail cutoff. [Paired percentile results](empirical.json) and [trial draws](entry-draws.npz).

## Baseline and wave attribution in the 2022 control

The [subgroup estimates and 95% intervals](components-2022.csv) give the model baseline and attributed wave exposure during the five observed weeks in the configured wave (weeks 8 through 12), on the cumulative hazard scale. Both fitters used the same specified exponent. All nine trial intervals for wave exposure reached zero. For the two-dose subgroup at pseudo-dose 0, the trial estimated baseline exposure of 0.01760 [0.01436, 0.50164] and wave exposure of 0.00501 [0, 0.00886]. The corresponding current estimates were 0.01728 [0.01436, 0.50164] and 0.00530 [0, 0.00876]. The components are fitted quantities with no independently observed division into baseline and wave mortality. Extra waves outside the configured dates are not labeled or individually estimated. Independently recording the components reproduced all three endpoints in both arms for every matched draw.

## Known-truth controls without sampling noise

All numbers in this table are deterministic. The generating frailty variance of the first cohort is 4 and that of the second is 0.5. The rate columns describe the first cohort. The ratios use both.

| Wave and fitting conditions | True rate | Current rate | Trial rate | True anchored KCOR | Current KCOR | Trial KCOR |
|---|---:|---:|---:|---:|---:|---:|
| no wave | 0.004000 | 0.004000 | 0.004000 | 1.00000 | 1.00000 | 1.00000 |
| published single shock | 0.004000 | 0.003333 | 0.004000 | 1.00000 | 0.99272 | 1.00000 |
| several shocks | 0.004000 | 0.003333 | 0.004000 | 1.00000 | 0.99272 | 1.00000 |
| nonlinear, all pulses in window | 0.004000 | 0.003124 | 0.003851 | 1.00000 | 1.01028 | 0.98732 |
| nonlinear, short first fit | 0.004000 | 0.003122 | 0.003846 | 1.00000 | 0.88520 | 0.97286 |
| nonlinear, partial wave window | 0.004000 | 0.003122 | 0.003846 | 1.00000 | 0.89226 | 0.97945 |
| nonlinear, partial window, unequal rates | 0.005000 | 0.003904 | 0.004811 | 1.05381 | 0.94013 | 1.02741 |

The single and multiple additive shocks have the same cumulative exposure at quiet entry. In both, the trial recovers the generating rate and frailty in both cohorts. With exponent 1.18 and all pulses inside the supplied wave period (long seed), true baseline cumulative hazard at quiet entry was 0.04847 in the first cohort. The current estimate was 0.03786 and the trial gave 0.04667. True wave exposure at wave exit was 0.05000. The corresponding estimates were 0.05707 and 0.04996. In that example the anchored ratio moved slightly farther from its target despite better component recovery.

In the partial-window runs, half of the generating wave exposure fell outside the supplied wave dates. Wave exposure reported at exit measures only exposure through the end of that supplied window. The true value there was 0.025, while total exposure before the quiet fit was 0.050. [All cohort parameters, component estimates and endpoints](synthetic-results.json).

## Sampled synthetic cohorts

Each row compares the same 200 independently sampled pairs of cohorts. Brackets around median KCOR give the central 95% range of estimates across datasets. The final column counts datasets with a smaller, equal or larger absolute endpoint error under the trial.

| Design | Target | Current median KCOR [central 95%] | Trial median KCOR [central 95%] | Median absolute error, current / trial | Smaller / equal / larger error |
|---|---:|---:|---:|---:|---:|
| several shocks | 1.00000 | 0.9923 [0.7436, 1.313] | 1.001 [0.7492, 1.326] | 0.0869 / 0.0889 | 104 / 0 / 96 |
| nonlinear, partial wave window | 1.00000 | 0.7989 [0.0001294, 528] | 0.7947 [0.2357, 1.261e+04] | 0.2751 / 0.3454 | 42 / 3 / 155 |
| nonlinear, partial window, unequal rates | 1.05381 | 0.845 [0.205, 8492] | 0.8227 [0.2187, 3.343e+04] | 0.2971 / 0.3435 | 53 / 0 / 147 |

For the short-window nonlinear null, the largest current and trial ratios occurred in different draws. The numerator cohorts in these two draws had fitted frailty variances of 99.57 and 91.46, respectively, against a generating variance of 4. Large fitted variances amplified the cumulative inversion. These examples trace the extremes in this simulation. They do not establish the cause of a particular Czech interval. The arithmetic means are dominated by these tails. The complete results include paired Monte Carlo intervals for mean absolute error differences. [Sample-level fitted parameters and ratios](synthetic-draws.npz) and [summary](synthetic-results.json).

## Disposition

### Root-cause check on the two fitted frailty variances

The same 200 sampled nonlinear null datasets were replayed, with the current and trial fitters alternately supplied each cohort's generating frailty variance. The baseline rate was still estimated from that dataset, conditional on the supplied variance. The other cohort kept its original fit and both arms retained the supplied partial wave window. Replaying the original parameter values reproduced every saved cohort growth estimate. The target ratio is 1.

| Generating variance supplied | Current median absolute error | Trial median absolute error | Trial worse / 200 |
|---|---:|---:|---:|
| Neither | 0.275 | 0.345 | 155 |
| Denominator only (0.5) | 0.231 | 0.262 | 142 |
| Numerator only (4) | 0.098 | 0.089 | 92 |
| Both | 0.029 | 0.022 | 67 |

Across the 200 datasets, the applied denominator variance had median 0.423 [central 95% range 0, 34.34] under the current fit and 0.414 [0, 7.22] under the trial. The numerator had median 2.05 [0, 41.30] and 1.94 [0, 58.15], respectively. Estimating the low variance contributed to typical error, and estimating the higher variance had the larger effect on this measure. These substitutions include the effect of re-estimating the rate conditional on known frailty. Cohort roles and generating variances were fixed in this design.

For the Czech data, 10,000 paired resamples of the existing 999 bootstrap draws gave approximate 95% Monte Carlo ranges for the trial-minus-current width change of −0.541 to +0.089 (1 vs 0), −0.586 to +0.106 (2 vs 0), and −0.007 to +0.002 (2 vs 1). This measures how precisely the existing draws locate the percentile limits. Population-level precision and repeated-sample coverage require separate evidence. [Reproduction](variance_check.py) · [Results](variance-check.json)

The Czech negative-control points moved slightly toward 1, and two reported percentile intervals narrowed. The experiment demonstrates recovery under the additive entry identity with noiseless data. It did not improve recovery consistently under nonlinear waves and sampling variation. Objection 3.2 remains open. No change to the general method or manuscript is recommended on this evidence.

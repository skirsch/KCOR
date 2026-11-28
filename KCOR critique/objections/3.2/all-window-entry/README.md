# 3.2 follow-up: pre-window survival with the all-window joint fitter

Status: isolated experiment. No official KCOR code, paper or PDFs changed. This supersedes neither the earlier 3.2 sensitivity study nor the all-window candidate recorded under 2.7.

## Question and paired designs

The preceding 3.2 experiment translates the effective first quiet-window rate r to the modeled enrollment rate k using

    k = r exp(theta H_entry)/(1 + theta r F_entry).

This is exact under additive proportional gamma frailty; it is approximate after a nonlinear wave. The prior attempt used the first-window joint seed and then fixed k. Here we retain the existing validated all-window two-start candidate, apply the same entry conversion to its seed, and include the first quiet-window boundary among the reconstructed gap offsets when it is preceded by modeled weeks. No changes to the wave attribution, model, windows, normalization or fitting-anchor convention.

The comparison uses the same seven noiseless designs, the same three sampled scenarios with 200 independent 50,000-person-per-cohort datasets, same seed namespaces and supplied partial wave dates as the original [3.2 test](../experiment.md). Table 8 uses the original archived inputs and all 999 matched person-bootstrap draws. The 2021 first quiet window begins at the modeled origin so entry logic is inactive; five sampled replays verify exact agreement. All 1,000 2022 comparator draws reproduce the archived all-window multistart endpoints.

## Noiseless recovery

| Scenario | True KCOR | All-window | Entry-aware all-window |
|---|---:|---:|---:|
| No wave | 1 | 1.000000 | 1.000000 |
| Single proportional shock before first quiet window | 1 | 1.207432 | 1.000000 |
| Several proportional shocks before first quiet window | 1 | 1.207432 | 1.000000 |
| Nonlinear, all pulses in supplied wave window | 1 | 1.312919 | 1.013502 |
| Nonlinear, short first quiet window | 1 | 1.141670 | 1.142123 |
| Nonlinear, partially supplied wave | 1 | 1.143444 | 1.144091 |
| Nonlinear, partial wave, unequal rates | 1.053811 | 1.131759 | 1.135436 |

For the proportional-shock tests the entry-aware fit exactly recovers true k=0.004 and theta=4 and 0.5; the unmodified all-window fit gives theta 5.676 and 0.634. The discrepancy occurs even with exact expected counts, so it is distinct from the finite-sample weak-information problem. Under the model, exposure before the first quiet interval must be carried into that interval just as exposure between subsequent quiet windows is. The result does not depend on any deaths before defining the cohort.

## Sampled comparisons

200 matched independent datasets per scenario; intervals for means are pointwise 95% Monte Carlo intervals. Central 95% refers to the distribution of independent endpoint estimates, not person-bootstrap intervals.

| Scenario | True KCOR | All-window mean [95% CI] | Entry-aware mean [95% CI] | Entry-aware improved / tied / worse |
|---|---:|---|---|---:|
| Several proportional shocks | 1 | 1.20386 [1.19425,1.21347] | 0.99774 [0.99321,1.00228] | 198/0/2 |
| Nonlinear partial wave | 1 | 1.14653 [1.14308,1.14997] | 1.14566 [1.14045,1.15087] | 104/0/96 |
| Nonlinear partial wave, unequal rates | 1.053811 | 1.13489 [1.13144,1.13833] | 1.13705 [1.13197,1.14213] | 77/0/123 |

The paired mean absolute error change is -0.17863 [-0.18837,-0.16889] in the proportional case; -0.00087 [-0.00331,0.00158] and +0.00261 [0.00037,0.00485] in the nonlinear partial cases. Thus adding entry exposure clearly repairs proportional recovery but does not repair the short partially labeled nonlinear designs.

## Why the short-window nonlinear tests remain biased

The all-window fitting objective retains the four early post-skip anchor rows even when outside the configured quiet mask, an existing original fitting convention. The short-window scenarios include a large mortality pulse in modeled week index 1 (the second accumulating week), within those four rows. Treating this pulse as a quiet baseline observation distorts k and theta. This is different from the KCOR endpoint normalization anchor.

For diagnosis only, restrict just the fitting residual to configured quiet weeks (the early normalization of KCOR and exposure reconstruction remain unchanged). Retain the 3.2 entry correction and the all-window algorithm.

| Deterministic test | True KCOR | Entry-aware with fitting anchors | Entry-aware quiet-only diagnostic |
|---|---:|---:|---:|
| Proportional short early pulses | 1 | 1.118146 | 1.000000 |
| Nonlinear all pulses in window | 1 | 1.013502 | 0.988125 |
| Nonlinear short first quiet window | 1 | 1.142123 | 0.988107 |
| Nonlinear partially supplied wave | 1 | 1.144091 | 0.994800 |
| Nonlinear partial wave, unequal rates | 1.053811 | 1.135436 | 1.045903 |

This paired proportional pulse control establishes the fitting-mask effect without nonlinear approximation or sampling variation. The earlier anchor rows contain useful information when they really are quiet; removing them indiscriminately is not a general solution.

| Sampled scenario | Target | Entry-aware with anchors, mean [95% CI] | Quiet-only mean [95% CI] | Quiet-only improved / tied / worse |
|---|---:|---|---|---:|
| Several proportional shocks (anchors clean) | 1 | 0.99774 [0.99321,1.00228] | 1.00413 [0.98502,1.02324] | 24/0/176 |
| Nonlinear partial wave (early anchor contains pulse) | 1 | 1.14566 [1.14045,1.15087] | 0.99321 [0.98188,1.00453] | 178/0/22 |
| Nonlinear partial wave, unequal rates | 1.053811 | 1.13705 [1.13197,1.14213] | 1.03952 [1.02884,1.05020] | 128/0/72 |

The clean-anchor proportional central 95% range expands from [0.9414,1.0606] to [0.7529,1.3183]. Conversely, in the contaminated-anchor nonlinear null it changes from [1.0773,1.2222] to [0.8579,1.1861]. These are empirical independent-estimate ranges. The paired mean absolute error change is +0.08103 [0.07045,0.09161] in the clean-anchor control and -0.08238 [-0.09081,-0.07396] in the nonlinear null. This isolates the tradeoff between precision and including mortality that does not follow the quiet baseline model.

## Czech Table 8: matched uncertainty

All three 2021 values and intervals remain identical for all three arms; the first quiet window starts at modeled origin and includes the four fit anchors. For 2022 the first quiet window starts 14 modeled weeks after origin; the early anchors overlap the configured wave.

| Enrollment | Pair | All-window [95% bootstrap CI] | Entry-aware all-window | Quiet-only diagnostic |
|---|---|---|---|---|
| 2021 | 1/0 | 1.013 [0.956,1.068] | unchanged | unchanged |
| 2021 | 2/0 | 1.033 [0.971,1.106] | unchanged | unchanged |
| 2021 | 2/1 | 1.020 [0.958,1.094] | unchanged | unchanged |
| 2022 | 1/0 | 1.01190 [0.96751,1.05505] | 1.01215 [0.96788,1.05530] | 1.01242 [0.95750,1.07526] |
| 2022 | 2/0 | 1.02766 [0.98506,1.07906] | 1.02792 [0.98533,1.07954] | 1.05436 [0.99606,1.11877] |
| 2022 | 2/1 | 1.01558 [0.97035,1.06781] | 1.01558 [0.97033,1.06788] | 1.04142 [0.97535,1.10529] |

Entry-aware fitting changes the 2022 point estimates by at most 0.000253 and the interval widths by less than 0.00021. Quiet-only fitting broadens the 2022 widths by approximately 30–35%; all six intervals contain one. The empirical negative-control target need not be exactly one; the true Czech frailties and decomposition of mortality into baseline and wave are unknown. No failures or insufficient-signal flags across either of the entry-aware or quiet-only replays.

Example unvaccinated subgroup 2022 fitted frailty:

| Pseudo-dose | All-window theta [95% bootstrap CI] | Entry-aware theta | Quiet-only diagnostic |
|---:|---|---|---|
| 0 | 2.365 [2.042,2.728] | 2.353 [2.012,2.713] | 1.110 [0.690,1.547] |
| 1 | 6.494 [5.581,7.337] | 6.494 [5.488,7.333] | 2.553 [1.239,3.760] |
| 2 | 14.578 [11.931,16.917] | 14.578 [11.753,16.870] | 6.067 [2.440,9.741] |

## Disposition

The proportional pre-window issue now has a convincing mathematical and numerical correction candidate. The same treatment makes essentially no difference to Table 8 with the new fitter, but is necessary to recover the known proportional-shock design. Recommend the first-entry exposure correction for lead-author consideration, not yet adoption. The nonlinear short-window issue is strongly tied to whether early fitting anchors follow the quiet baseline; the quiet-only intervention is a diagnostic, not approval to abandon the original anchors. This also ties to the original nonlinear in-wave modeling objection 1.2 and the distinction between statistical information and model adequacy. Further work on mixed weeks should preserve the original KCOR intent.

Reproduce: Python 3.11 replay.py synthetic, replay.py (Table 8), anchor_diagnostic.py, quiet_only_table.py. The JSON/CSV/NPZ files retain all results and subgroup parameter intervals. The primary revision remains unchanged.

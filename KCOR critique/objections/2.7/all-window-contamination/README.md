# Contamination experiment replay with the all-window joint fitter

Status: experimental only, not adopted. No change to the experiment's purpose or the accepted paper/pipeline.

## Design

Replayed the original 200 datasets per contamination level (five levels), using the archived calibration, original cohort generator, two equal-baseline groups with generating frailty variances 0.5 and 4, weekly baseline rate 0.000456169, 130 weeks including skip, 53 configured quiet weeks (rebased 4–56 inclusive), four early fit anchor weeks (0–3), fifth accumulating week ratio normalization, and contamination imposed on numerator cohort A during the quiet window. The original generator's own discrete weekly convention is retained. No shift of the contamination or anchor. At 50,000 people per cohort all 1,000 historical and preceding joint-fit endpoints reproduce within 1e-8. The same 200 seed streams per level were run at 500,000 and 5,000,000 per cohort to diagnose population-size dependence; these are separate scenarios, not more replicates of the original experiment.

All 7,000 sampled endpoint fits and 35 expected-count endpoint fits succeeded (two subgroup fits each), with no insufficient-signal flags. The all-window trial uses the same candidate function used for Table 8, including profiling rebuilt gap offsets and a second optimizer start. In this design there is only one quiet block. Its benefit is estimating k and theta using the complete fitted weeks including the early anchor weeks, not extracting information from later quiet blocks. No gap re-entry is involved.

## All levels at the original cohort size

Pointwise 95% Monte Carlo intervals for the mean across 200 independently simulated datasets at each level.

| Added hazard/week | Original fixed | Previous joint | All-window joint |
|---:|---|---|---|
| 0 | 1.050 [1.034,1.066] | 1.064 [1.039,1.090] | 1.049 [1.031,1.068] |
| 0.0001 | 1.010 [0.993,1.027] | 1.088 [1.061,1.115] | 1.042 [1.023,1.060] |
| 0.0003 | 1.060 [1.045,1.076] | 1.145 [1.120,1.170] | 1.085 [1.067,1.104] |
| 0.001 | 1.229 [1.210,1.249] | 1.313 [1.282,1.343] | 1.239 [1.217,1.262] |
| 0.003 | 1.457 [1.435,1.479] | 1.543 [1.511,1.574] | 1.468 [1.443,1.492] |

At zero the trial changes the current joint result by -0.01485 [-0.02432,-0.00539], but differs from the original fixed result by -0.00053 [-0.02129,0.02023]. Thus it does not eliminate the low-information bias at 50,000 people. Compared to the trial's own zero level, added hazards of 0.0001, 0.0003, 0.001 and 0.003 give differences -0.00792 [-0.03436,0.01853], +0.03585 [0.00966,0.06204], +0.18995 [0.16066,0.21925], +0.41815 [0.38707,0.44922]. These are pointwise Welch intervals because different contamination levels use independent streams.

## Population-size dependence under zero added hazard

| People per cohort | Previous joint mean [95% CI] | Trial mean [95% CI] | Trial empirical middle 95% of individual simulated endpoints |
|---:|---|---|---|
| 50,000 | 1.0643 [1.0387,1.0899] | 1.0494 [1.0305,1.0684] | [0.8323,1.3590] |
| 500,000 | 1.0109 [1.0023,1.0195] | 1.0090 [1.0027,1.0153] | [0.9205,1.0984] |
| 5,000,000 | 1.0020 [0.9994,1.0047] | 1.0013 [0.9993,1.0033] | [0.9735,1.0327] |

The final column is a distribution of independent estimates, not person-bootstrap uncertainty intervals. The exact expected-count replay gives the trial theta_A=0.499591 and theta_B=3.996732 and KCOR 1.000000000 to numerical precision, at all three cohort sizes. Thus the remaining mean bias is a sampling/estimation effect in this design, not a deterministic failure of the candidate equation.

## Why the finite-size bias persists

At 50,000, substituting generating frailties and/or the population anchor isolates the contributions:

| Counterfactual | Mean KCOR [95% MC interval] |
|---|---|
| Generating frailties, population anchor | 1.0013 [0.9976,1.0050] |
| Generating frailties, sampled anchor | 1.0207 [1.0031,1.0383] |
| Fit A, generating B, sampled anchor | 1.0554 [1.0367,1.0741] |
| Generating A, fit B, sampled anchor | 1.0147 [0.9970,1.0323] |
| Fit both, sampled anchor | 1.0494 [1.0305,1.0684] |
| Fit both, population anchor | 1.0367 [1.0196,1.0538] |

Paired changes with population versus sampled anchor for the oracle give +0.01936 [0.00156,0.03717]; the earlier independent 100,000-replicate oracle check estimated the average sampled-anchor contribution at +0.00831 [0.00747,0.00914]. This particular sample of 200 overstates that contribution. The paired re-estimation contributions using the same sampled anchor are +0.03474 [0.02464,0.04483] for A alone, -0.00599 [-0.01808,0.00610] for B alone, and +0.02877 [0.01265,0.04489] for both. These are diagnostic substitutions, not proposals to supply the true frailty or anchor to operational KCOR.

Quantitative check against the recently derived Fisher limit: under the exact proportional-Gompertz-gamma model for the original 57 fitted weeks (including anchors), theta_A=0.5 has theoretical local SE 3.503 at 50,000. For an illustrative hypothetical unconstrained Gaussian estimate X~N(theta,sigma^2), constrained estimate max(0,X) has expected value theta*Phi(theta/sigma)+sigma*phi(theta/sigma) and probability of zero Phi(-theta/sigma). The local normal approximation is not exact at the boundary, but explains and predicts much of the asymmetry.

| People per cohort | Theoretical SE of theta_A | Predicted clipped-normal mean | Trial mean theta_A [95% MC interval] | Predicted zero fraction | Observed near-zero fraction |
|---:|---:|---:|---|---:|---:|
| 50,000 | 3.503 | 1.662 | 1.546 [1.251,1.841] | 44.3% | 43.0% |
| 500,000 | 1.108 | 0.736 | 0.717 [0.607,0.828] | 32.6% | 35.0% |
| 5,000,000 | 0.350 | 0.512 | 0.516 [0.469,0.563] | 7.7% | 7.5% |

With the original schedule and mortality, local +/-50% precision at theta=0.5 would require about 37.7M per cohort and +/-25% about 150.9M. This is a model-conditional local Wald benchmark, not a finite-sample minimum or a requirement for recovering the KCOR ratio. Correspondence of the clipped-normal approximation and the observed means/frequencies is evidence supporting low-information boundary bias, not a proof that all error is due to truncation.

## Full sweep at larger sizes

| Added hazard/week | Trial, 500k people per cohort | Trial, 5M people per cohort |
|---:|---|---|
| 0 | 1.0090 [1.0027,1.0153] | 1.0013 [0.9993,1.0033] |
| 0.0001 | 1.0270 [1.0211,1.0328] | 1.0210 [1.0191,1.0229] |
| 0.0003 | 1.0836 [1.0771,1.0900] | 1.0878 [1.0858,1.0899] |
| 0.001 | 1.2492 [1.2423,1.2561] | 1.2537 [1.2513,1.2560] |
| 0.003 | 1.4887 [1.4810,1.4963] | 1.4859 [1.4836,1.4882] |

At 5M, the first tested nonzero addition differs positively from the zero-addition mean by 0.01969 [0.01698,0.02240]. The test retains the expected direction with negligible observed null bias. It does not show a specific contamination threshold below the grid. A high-information version supplements, but should not conceal, the original small-cohort experiment.

## Decision

The threshold wording is already corrected. The originating bias is now substantially explained: weak frailty information plus the nonnegative parameter boundary and noisy sampled anchor. The all-window fitter reduces bias compared to the previous joint fit and stabilizes the multiple-window tests, but cannot create information in the original contiguous-window design. The 5M version demonstrates the intended contamination experiment in a higher-information regime. The 50k version remains an important stress case. Any choice to include the larger-cohort study, or adopt the new fitter, requires discussion before the paper or method is changed. No adopted pipeline work was modified.

Reproduce with Python 3.11: replay.py, information_check.py, distribution_check.py. Files: runs.csv (all sampled and exact records), endpoint-summary.csv, parameter-summary.csv, paired-changes.csv, null-decomposition.csv, information-comparison.csv.

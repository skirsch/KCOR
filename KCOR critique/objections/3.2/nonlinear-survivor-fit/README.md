# Exact nonlinear survivor-law fitting across quiet windows

**Experimental candidate only.** This is a deeper fitting change authorized for isolated testing, not adopted into the method or paper. It follows the finding in [the combined 1.2 trial](../../1.2/all-window-weekly-survival/README.md) that a mathematically exact wave inversion could not recover an exact null when its fitted frailty inputs were biased by the proportional-gamma working fit after nonlinear selection.

## Mathematical model and conventions

The target parameters remain the common Gompertz scale k and enrollment gamma frailty variance theta. The original fixed Gompertz slope and supplied alpha are retained. The model for survivor selection is the same as the exact weekly wave trial:

    S(B,W) = E[exp(-Z B - Z^alpha W)], Z ~ Gamma(1/theta,theta).
    q_i = -log S(B_{i+1},W_{i+1}) + log S(B_i,W_i).

Fit the observed integrated weekly hazards with the same unweighted least-squares objective, recomputing cumulative exposures inside the residual at each candidate (k,theta). In quiet weeks W is constant and B increases by the integrated fitted Gompertz baseline. The prior wave exposure remains part of the survivor law when fitting later quiet weeks. Earlier nonwave gaps, including a gap before the first quiet interval, are profiled from observed cumulative survival and use the nonnegative proportional gap-exposure convention. At the modeled wave opening, the observed pre-wave survival determines the proportional accumulated exposure. During marked wave weeks perform the same survival reconstruction as 1.2, using a baseline increment and a nonnegative added nonlinear exposure, or a smaller baseline increment when observed mortality falls below the planned conditional baseline. This preserves observed deaths under the trial attribution. When every observed weekly wave hazard exceeds the no-wave conditional forecast, a positive alpha implies that prior wave selection further reduces surviving mean frailty; all increments are nonnegative and an aggregate wave-exit solve is equivalent to the weekly recursion. Otherwise the exact weekly recursion is used.

The original early fit anchors are retained when outside the marked nonlinear wave. Marked wave weeks do not enter the quiet baseline residual, but all their mortality enters the exposure reconstruction. This is not a change to the KCOR normalization anchor. An unmarked nonlinear shock cannot be identified as nonlinear from all-cause counts alone; the model treats unmarked gaps as proportional, as before. No-wave/alpha=1 routing returns the existing all-window candidate; the separate proportional pre-first-window correction remains in its own 3.2 trial.

The experimental fitter uses the all-window working fit as an initial seed and a second start at theta=0.5, selecting the lower residual solution. It retains the existing minimum-death and upper-bound safeguards, with explicit flag/failure reporting. Only the fitted model and marked wave eligibility in the residual are changed. The downstream ratio uses the exact weekly correction from 1.2 unchanged.

Numerical integration uses normalized 64-node generalized Laguerre quadrature constructed using a scaled Jacobi matrix, gamma exponential tilt and two analytical moment terms to improve accuracy, with the tested 1.2 probability-space integral as a fallback in less stable regions. The zero-variance limit and an expansion near zero avoid singular gamma shape. The pilot compares the numerical hazard against the 1.2 reference for 90 combinations spanning theta 1e-6 to 100, alpha 0.7/1.18/2 and a range of cumulative exposures. All agree within the specified numerical tolerance. The noiseless generating survival itself is independently integrated by the original synthetic test generator.

## Known truth: fully specified nonlinear model

The two pooled synthetic designs reproduce the exact generating k and all four variances (4,0.5,4,0.5) to numerical tolerance. Main expected-count pooled KCOR 0.99999999995; stress 0.99999999996. This corrects the previous combined candidate's noiseless main ratio 0.988967 and stress ratio 0.990133 without supplying true parameters to the fitter.

## Earlier mixed-window designs (3.2)

Both arms use exact weekly wave inversion. The comparison arm is the previous entry-aware all-window gamma-working fitter. The second arm is the experimental exact survivor-law fitter above. Noiseless:

| Scenario | True KCOR | Gamma working fit | Nonlinear survivor fit |
|---|---:|---:|---:|
| All pulses in supplied wave dates, longer first window | 1 | 1.018984 | 1.000000 |
| Short initial window, all pulses marked | 1 | 1.203943 | 1.000000 |
| Short initial window, partial wave dates | 1 | 1.202782 | 0.998951 |
| Partial wave dates, unequal baseline rates | 1.053811 | 1.176512 | 1.050497 |

The correctly labeled cases recover both generating frailty variances and rates to numerical precision. In partial-wave cases the model still treats unmarked pulses as proportional, so complete recovery is not expected.

Matched 200 samples per scenario (50k people per cohort) use original seed streams and generator. Intervals refer to Monte Carlo means across independent datasets, not individual bootstrap intervals.

| Scenario | True KCOR | Gamma working fit mean [95% CI] | Nonlinear survivor fit mean [95% CI] | New improves absolute error / 200 |
|---|---:|---|---|---:|
| All pulses marked, longer first fit | 1 | 1.02220 [1.01787,1.02653] | 1.01018 [0.99159,1.02877] | 39 |
| Short first fit, all marked | 1 | 1.21131 [1.20543,1.21718] | 1.00399 [0.99295,1.01503] | 197 |
| Nonlinear partial wave | 1 | 1.20438 [1.19857,1.21019] | 0.99987 [0.98785,1.01188] | 190 |
| Nonlinear partial wave, unequal rates | 1.053811 | 1.17843 [1.17279,1.18407] | 1.04609 [1.03479,1.05739] | 165 |

There are no failures in these 800 samples. In the long-window case the original supplied wave starts at modeled week 0 even though the first actual pulse is week 5; the existing early fit anchors are clean in this particular generating history. Treating the marked early rows as nonlinear-wave nuisance observations loses useful information. The central 95% sample range widens from [0.9665,1.0832] to [0.8041,1.3357] and paired mean absolute error increases by 0.07230 [0.06075,0.08385]. In the short-window cases where the early rows contain a pulse, removing that contamination from the baseline fitting objective and using the exact survivor law substantially reduces the systematic bias. This illustrates accuracy/precision tradeoffs under a supplied window, not a recommendation for outcome-dependent window selection.

[Full mixed-window results](mixed-results.json) and sampled parameter arrays (mixed-draws.npz).

## Actual Table 8 validation

Same archived inputs, same 999 person-bootstrap draws per enrollment, same forced Czech sensitivity exponent 1.18 and normalization. Each endpoint uses the exact weekly wave inversion in both arms. The new fit differs only in survivor law and in treating marked wave rows as wave exposure observations instead of quiet baseline rows. All six point estimates matched the ordinary workbook entry point to maximum absolute discrepancy 2.22e-16.

| Enrollment | Pair | Gamma working fit [95% CI] | Nonlinear survivor fit [95% CI] |
|---|---|---|---|
| 2021 | 1/0 | 1.01279 [0.95552,1.06823] | 1.01277 [0.95551,1.06827] |
| 2021 | 2/0 | 1.03310 [0.97032,1.10552] | 1.03309 [0.97038,1.10560] |
| 2021 | 2/1 | 1.02005 [0.95768,1.09403] | 1.02006 [0.95776,1.09408] |
| 2022 | 1/0 | 1.01195 [0.96848,1.05587] | 1.01255 [0.95757,1.07539] |
| 2022 | 2/0 | 1.03028 [0.98735,1.08406] | 1.05566 [0.99627,1.12107] |
| 2022 | 2/1 | 1.01811 [0.97103,1.07380] | 1.04257 [0.97537,1.10948] |

2022 widths increase approximately 34.8%, 29.1%, 30.5%. All six intervals contain 1; the negative-control target is not proven to be exactly one. No failed fits or insufficient-signal flags in any of the 18,000 Table 8 subgroup fits. The 2022 changes are consistent with the preceding quiet-only anchor diagnostic, because the four early fitting rows fall inside the configured wave. The normalization still uses the same original week. In 2021, the early fit anchors lie in quiet weeks, so the point and interval changes are tiny.

Unvaccinated 2022 subgroup (pseudo-dose order), fitted theta:

| Dose | Gamma working fit [95% CI] | Nonlinear survivor fit [95% CI] |
|---:|---|---|
| 0 | 2.365 [2.042,2.728] | 1.112 [0.691,1.558] |
| 1 | 6.494 [5.581,7.337] | 2.557 [1.240,3.766] |
| 2 | 14.578 [11.931,16.917] | 6.075 [2.443,9.759] |

The original parameters and all subgroup bootstrap intervals are preserved. The observed Czech cohort frailties are not known, and these results do not establish that the new values are the true ones.

[Endpoint intervals and paired change intervals](table8-comparison.csv), [fitted baseline and frailty intervals](table8-parameters.csv), [ordinary workbook check](workbook-verification.json).

## Sampled main and stress coverage

The main study originally used 199 bootstrap draws across 100 independent populations of one million people. The new fitter's first 99 matched draws were compared with the identical prefix of the prior exact-wave/gamma-working fit, using the same full-batch weights. Both covered 89 of 100 datasets (exact coverage CI 81.2–94.4%, below nominal 95%). However the main mean KCOR improves from 0.98982 [0.98493,0.99471] to 1.00093 [0.99596,1.00591]. The full 199 matched draws are assessed separately below. In the 50-replicate 99-draw short-window stress, coverage is unchanged at 48/50 [86.3%,99.5%]. Mean KCOR moves from 1.03068 [1.00878,1.05258] to 1.04019 [1.01789,1.06249]; mean interval width from 0.31934 [0.29903,0.33965] to 0.32735 [0.30688,0.34782]. There are no failed subgroup fits; number of stress draws with insufficient-signal flags (241) is unchanged. The remaining finite-sample bias is not cured simply by using the correct survival equation.

| Main cohort | Generating theta | Working-model mean theta [95% MC CI] | Exact-survivor mean theta [95% MC CI] |
|---|---:|---|---|
| 1940 dose 0 | 4 | 4.263 [4.232,4.295] | 4.015 [3.985,4.045] |
| 1940 dose 1 | 0.5 | 0.543 [0.516,0.570] | 0.506 [0.481,0.532] |
| 1950 dose 0 | 4 | 4.345 [4.258,4.431] | 4.057 [3.976,4.138] |
| 1950 dose 1 | 0.5 | 0.517 [0.456,0.578] | 0.481 [0.424,0.538] |

The main model-consistent fit substantially improves most of the frailty estimates on average, agreeing with the exact known-truth result. The sampled estimates and their uncertainty remain conditional on the fixed alpha, wave dates, and chosen weekly attribution. These studies do not establish efficacy of a treatment in real cohorts.

## Full matched 199-draw main comparison

After the 99-draw pilot, the remaining 100 matched draws were computed with the same full-batch weights, preserving every original result. This makes the main comparison directly comparable to 1.2. All 100 independent datasets completed.

| Scenario | Fitter (both exact wave) | Coverage [exact 95% CI] | Mean KCOR [95% MC CI] | Mean interval width [95% MC CI] | RMSE |
|---|---|---|---|---|---:|
| Main, 1M, 199 draws | Gamma working | 90/100 [82.4%,95.1%] | 0.98982 [0.98493,0.99471] | 0.08708 [0.08573,0.08843] | 0.02654 |
| Main, 1M, 199 draws | Exact survivor | 90/100 [82.4%,95.1%] | 1.00093 [0.99596,1.00591] | 0.08853 [0.08718,0.08988] | 0.02496 |
| Stress, 50k, 99 draws | Gamma working | 48/50 [86.3%,99.5%] | 1.03068 [1.00878,1.05258] | 0.31934 [0.29903,0.33965] | 0.08222 |
| Stress, 50k, 99 draws | Exact survivor | 48/50 [86.3%,99.5%] | 1.04019 [1.01789,1.06249] | 0.32735 [0.30688,0.34782] | 0.08747 |

Main: five datasets gain coverage, five lose it (paired p=1), mean interval width increases by 0.001449 [0.001309,0.001590]. Both 90% coverage estimates are marginally compatible with nominal 95%, not proof of calibration. The exact survivor model removes known-truth deterministic bias and substantially improves the main mean, but leaves coverage and weak-information finite-sample limitations unresolved. The prior insufficient-signal flags in stress remain; no additional failures. No main flags or failures.

The pilot's explicit proportional route check verifies that alpha=1 gives exactly the existing all-window fit. The optimized profile quadrature agrees with the prior probability-space integration on 18 actual fitted Table 8 subgroup curves, including deficit cases, to maximum absolute weekly hazard discrepancy 1.51e-15. These are numerical model checks rather than validation of the chosen real-data wave dates or alpha.

## Disposition

The mathematical consistency issue between nonlinear wave inversion and subsequent quiet-window fitting has a concrete correction candidate. The noiseless results and sampled main mean support it; 2022 Table 8 changes are material enough to report and remain near null with intervals containing 1. Sampled clean-anchor and low-information controls expose real accuracy/precision tradeoffs, and the marginal coverage and partial-wave assumptions remain important. Recommend consideration of exact survivor-law fitting and exact weekly inversion as a coherent pair, subject to lead-author approval, including the marked wave fitting mask and deficit attribution. Do not claim uniform superiority, general calibration, recovery of unlabeled nonlinear waves, or adopt a generalized redesign. The earlier proportional-entry candidate is a separate routing decision. No official code or paper change.

Reproduction: Python 3.11 pilot.py; replay.py main; extend_main.py; replay.py stress; mixed.py; replay.py table; check_workbook.py; verify_profiles.py. Full evidence: main-full-results.json, main-full-draws.npz, main-full-intervals.csv, stress-results.json, stress-draws.npz, mixed-results.json, mixed-draws.npz, table8-draws.npz, table8-comparison.csv, table8-parameters.csv, workbook-verification.json, quadrature-verification.json. The 99-draw main prefix and its summary remain separately identifiable for reproducibility.

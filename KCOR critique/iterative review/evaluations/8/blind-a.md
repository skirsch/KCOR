# Independent KCOR review

**Question.** Please examine KCOR closely. What is wrong, if anything, and how would you fix it?

The core gamma-frailty identity is correct. If individual hazards are proportional to a starting mean-one gamma frailty with variance θ, marginal survival is (1 + θG) raised to the power −1/θ, and inversion recovers the reference-frailty cumulative exposure G. The fixed-slope identification proposition in §2.4.4 also holds under its stated proportional survivor history. The main scientific qualifications concern which selection differences that identity removes, a claim of structural non-identification in the optional module, and the difference between the fitted parameter and the parameter actually used for some reported contrasts. I also found two reproducible errors in ordinary workbook reporting.

## 1. Weekly aggregate survival can identify the wave exponent and a constant efficacy multiplier together

The abstract, §§2.7.2 and 5.4, and SI S2.1 say the exponent α cannot be separately identified from a cohort-specific intervention multiplier without an externally fixed effect scale. A single cross-sectional ratio supplies one equation in two unknowns. A sequence of weekly survival observations under a common wave supplies additional equations because the survivors' frailty distributions change. The distinction matters for the paper’s exposition of what the data can identify.

Consider a permitted special case. Quiet history establishes the baseline cumulative exposure G(t) and variances 0 in reference cohort R and 1 in cohort A. Both use G(t) = 0.001 expm1(0.085t/52)/(0.085/52). For R, cumulative observed hazard is G(t) + W(t), so aggregate survival determines the cumulative common wave W(t) at each weekly boundary. For A take an exponential mean-one frailty Z and an unknown constant c = 1 − x multiplying its wave. Its observed cumulative hazard is

\[
H_A(t)=-\log\int_0^\infty \exp[-(1+G(t))z-cW(t)z^\alpha]\,dz .
\]

At weeks 22 and 35 take W = (0.025, 0.065), c = 0.7 and α = 1.18. Numerical integration gives reference cumulative hazards (0.04740036, 0.10102057) and A cumulative hazards (0.04049789, 0.08143710). The Jacobian of the two A observations with respect to (c, α) has rows (0.02587292, 0.00883113) and (0.06371760, 0.01961961). Its determinant is −0.0000550821. Solving the two equations from (0.8, 1.1) returns (0.7, 1.18). An analytical check at α = 1 gives the derivative ratio (∂H/∂α)/(∂H/∂c) = c[ψ(2) − log(1 + G + cW)], which differs between observation times. The inverse function theorem yields local identification. Only cumulative survival at week boundaries and the quiet-model parameters enter this example.

**Fix.** Describe external efficacy as an input to the proposed localization procedure. Distinguish freely varying cohort-by-time multipliers, a single cross-sectional contrast, and a constant multiplier in longitudinal data. Keep the Czech conclusion that its analysis did not identify α. This example establishes neither global uniqueness nor useful precision in those data.

## 2. The exported pooled series loses every date except the last

In `code/KCOR.py:4784`, each date starts new lists of stratum values and weights. The pooling block at line 4826 has the same indentation as the date loop. It appends a row at line 4963 only after the last date has been processed. A separate loop computes per-date rows under a different code, but line 5460 explicitly omits that collection from the returned output.

I passed the frozen `analysis/table8/input.xlsx` sheet `2021_24` through ordinary `process_workbook` using its supplied point-only wrapper, module off and Table 8 layout. Dose 1 versus 0 had **174 dates** in each of 1940, 1950, 1960 and the direct aggregate, and **one** date in the weighted pool (birth-year code 0). The lone pooled result was 1.020941 on October 7, 2024. There was no pooled row for December 26, 2022, Table 8’s reporting date. The verification in `analysis/solver_scaling/check_workbook.py:23–26` rebuilds the pooled number from three individual rows and separately computed weights. This explains how the archived table can be numerically correct while the ordinary workbook fails to deliver it. The same discrepancy occurs with a small 20-week fixture (20 individual dates, one pooled date).

**Fix.** Emit the pooled row within the date loop, including an explicit unavailable row at affected dates if needed. Compare the actual exported pooled row at the reporting date to each archived Table 8 estimate. Test the entire sequence as well as its final date.

## 3. Missing component histories can silently change the risk set and the reported contrast

§2.9.1 says an incomplete fixed-cohort history makes the affected contrast unavailable and pooled contrasts are not silently reweighted. `code/KCOR.py:6750–6796` sums sex and other component rows before checking calendar completeness. `_complete_week_history` (line 4246) checks the dates through each group's own last observation. Neither check detects all losses of a component from a continuing cohort.

Two point-only reproductions used the supplied 20-week `input_frame("2021_24")` fixture.

- I added a male component to each existing group, with four times the female weekly deaths, opening size 50,000, and subsequent risk sets calculated by subtracting prior deaths. Removing only its row at event week 10 for birth group 1940, dose 1, changed that group's anchored dose-1/dose-0 KCOR from **1.001204 to 0.987418**, with `normalization_status=ok` in both analyses. The week survives aggregation because its female row exists. The disjoint bootstrap history reader correctly rejects this interior hole, but the ordinary point estimate is reported.
- Starting with the unmodified fixture, removing only the **final** weekly row for age group 1940 and dose 1 changed the direct aggregate endpoint from **0.999967 to 1.009796**. Both the direct aggregate and the weighted pool reported status `ok`, and the bootstrap history reader accepted this truncated input. The last observed opening risk set was still positive. The aggregate had lost a component and the weighted pool had dropped a contributor.

**Fix.** Before summing, validate each expected component's calendar and fixed-cohort risk bookkeeping through the endpoint of each requested comparison. Encode exhaustion and legitimate administrative ends explicitly. For unmatched follow-up, report to the last common supported horizon or mark later contrasts unavailable. Use the same validation for the original estimate and resamples, with separate handling for explicit independent aggregate totals.

## 4. High, accurately estimated frailty is converted into zero too readily

`code/KCOR.py:1675–1705` substitutes zero when the constrained estimate is at least 99.9% of the numerical maximum. This condition can be met below the bound, with a perfect fit. Using 80 quiet weeks, G(t) as above, and weekly hazards equal to the exact differences of log(1 + θG)/θ, I called the ordinary `fit_theta0_gompertz` with four anchor weeks. For generating θ = 99.80 it returned raw and applied θ = 99.80. For generating θ = 99.95 it returned raw θ = 99.95000000000003 and **applied θ = 0**, despite weekly RMSE of 2.2 × 10⁻¹⁸. The true baseline did not change. Relative to a comparator with the same baseline and known zero variance, the second fit gives an unanchored endpoint of **0.2641** and a fourth-week-anchored endpoint of **0.3141** instead of 1. The code flags the fallback. The scientific error is characterizing the bound rule as evidence of insufficient signal and claiming in §5 (line 940) that KCOR does not impose zero for any cohort.

This affects empirical uncertainty as well. SI S6.1.2 discloses 287/999 upper-bound dose-2 draws in the youngest displayed band, and `analysis/solver_scaling/s6-all-window.json` records another 17/999 for dose 0 in that band. Those KCOR percentiles describe a discontinuous fallback procedure.

**Fix.** Distinguish a numerical bound from lack of information, using an expanded parameter range and a profile of the objective when feasible. Keep genuine unresolved fits visibly unresolved. If the historical zero-frailty fallback is retained, present it as an explicit sensitivity calculation and distinguish its distribution from uncertainty in the fitted-model contrast.

## 5. A difference in mean health need not produce different fitted frailty variances

§1.2 (line 47) describes being healthier on average as manifesting in a different θ. §2.1 (line 127) says residual differences cannot be attributed to frailty-driven selection under the working model. Suppose selection alone gives absolute frailty X = Z in one group and X = 2Z in the other, with the same mean-one gamma Z and the same individual baseline h(t) at the same absolute X. Re-expressing each group using its own mean-one frailty gives the same θ in both and baseline scales h(t) and 2h(t). Perfect inversion returns an **unanchored ratio of 2**, entirely due to baseline selection. Anchoring gives 1 and discards the constant level difference. A related sentence in §2.1 (line 141) says cumulative outcome accumulation is invariant to depletion-induced hazard-ratio fluctuations. With common baseline cumulative hazard 1 and θ values 0 and 1, the observed cumulative hazards are 1 and log 2. Integration alone plainly leaves a depletion difference.

**Fix.** Define the reference person as having the mean frailty *of their own cohort at the rebased origin*. Distinguish selection that shifts that mean from selection that changes relative heterogeneity. Retain the more careful conditional interpretation already given in Box 2. Attribute recovery of the reference curve to inversion with the correct variance and survivor law.

## 6. Some validation and robustness claims exceed the displayed evidence

- §3.3.1 (lines 772–782) says bias, variance, coverage and non-identifiability rates are reported for each of five frailty specifications, explicitly including two-point and within-subgroup correlated frailty. Neither the main text nor SI S4.5 contains that results table. The archived simulation grid describes a lognormal scenario, and the tail experiment samples from a continuous tail mixture. The archived coverage CSV has five rows for gamma null, harm, benefit, non-gamma and sparse events, with one reference-dataset target as explained in §2.9.1. It does not supply the promised five-distribution comparison. Either publish the corresponding specifications, sampling units, population targets, results and implementation identifiers or limit the claims to the experiments shown.
- The SI S4.4 figure caption (line 223) says all three panels share a color scale centered at 1. The supplied graphic has three visibly different color bars, with ticks around 1.025–1.125, 1.10–1.25, and 1.08–1.12 respectively. At zero quiet-window offset the dose-2/dose-0 value changes from **1.269 to 1.110** across the displayed reference lengths. All shown cells exceed one, but the reference length materially affects the magnitude. Correct the caption or regenerate the figure using a common scale, report the numerical range, and identify the estimator edition that produced it.

**Scope.** This review used only the frozen candidate, its manuscript sources, analysis artifacts, and small point-only replays or deterministic calculations. I did not repeat the large Monte Carlo or bootstrap studies. None of these results supplies a causal interpretation of the Czech mortality comparisons.

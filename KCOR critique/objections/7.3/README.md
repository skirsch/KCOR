# 7.3 - Absolute solver tolerance causes incorrect frailty on noiseless rare-event data

**Severity:** High for low-rate fitting  
**Status:** Addressed in m130 (approved), preserving m129.  
**Related root causes:** [2.7](../2.7/README.md), [6.11](../6.11/README.md). Distinct from low statistical information; exact input has no sampling noise.  
**Review evidence:** [Blind B](../../iterative%20review/evaluations/7/blind-b.md)

## Original defect and correction

At baseline rate $k=0.00001$ per week, true frailty variance 50, Gompertz slope 0.085/year, and 12 exact weekly hazards, the review found that the prior proportional fitter stopped at 42.03697 with status OK (42.03706 in the matched replay). The raw absolute residuals and objective gradients are tiny at such low rates. The fitting target itself is correct, but the numerical convergence criterion is reached prematurely.

The seed, conditional-variance fit and all-window joint proportional least-squares fit now divide residuals by the fixed mean absolute observed hazard in the relevant sample (floor $10^{-12}$). This does not change the mathematical least-squares minimum, fit mask, parameter bounds or raw hazard diagnostics. It only rescales the optimizer objective. The optional nonproportional survivor-law fitter is unchanged.

In the counterexample the variance is now 50.00000. The exact grid spans four baseline weekly rates (0.000001, 0.00001, 0.0001, 0.001), three variances (0.5, 4, 50), and 12 or 52 weeks. Maximum absolute frailty error falls from 29.3524 to $1.9\times10^{-10}$ in 24 cases. All exact-grid regression assertions pass. Weak-identifiability diagnostics may still fire when data are scarce; accurate noiseless recovery does not establish attainable sampling precision.

## Table 8 - full matched replay

Both primary (module inactive) and sensitivity (forced alpha=1.18) were fully recomputed on the same original workbook and all 999 matched person-bootstrap draws per enrollment, using the current shared numerical integral. All 12 original workbook values agree with direct production output to within $2.22\times10^{-16}$. The forced-alpha comparison uses the previous archived sensitivity column, so it includes the earlier integral correction and is not exclusively an isolation of this change. The primary comparison is directly paired against m129.

All 12 reported point estimates and percentile intervals are unchanged to manuscript precision, and all contain 1; neither mode has unsuccessful fits or insufficient-signal flags. The maximum absolute difference across matched KCOR draws is $5.46\times10^{-8}$ for primary and $2.54\times10^{-5}$ for forced-alpha. The former sensitivity-archive caveat is removed from the paper because it has now been fully recomputed.

| Mode | Enrollment | Comparison | Previous estimate [95% interval] | Current estimate [95% interval] |
|---|---|---|---|---|
| Primary | 2021_24 | 1 vs 0 | 1.0162 [0.958, 1.072] | 1.0162 [0.958, 1.072] |
| Primary | 2021_24 | 2 vs 0 | 1.0456 [0.982, 1.119] | 1.0456 [0.982, 1.119] |
| Primary | 2021_24 | 2 vs 1 | 1.0289 [0.966, 1.103] | 1.0289 [0.966, 1.103] |
| Primary | 2022_06 | 1 vs 0 | 1.0070 [0.962, 1.053] | 1.0070 [0.962, 1.053] |
| Primary | 2022_06 | 2 vs 0 | 1.0164 [0.970, 1.071] | 1.0164 [0.970, 1.071] |
| Primary | 2022_06 | 2 vs 1 | 1.0094 [0.957, 1.067] | 1.0094 [0.957, 1.067] |
| Sensitivity | 2021_24 | 1 vs 0 | 1.0128 [0.956, 1.068] | 1.0128 [0.956, 1.068] |
| Sensitivity | 2021_24 | 2 vs 0 | 1.0331 [0.970, 1.106] | 1.0331 [0.970, 1.106] |
| Sensitivity | 2021_24 | 2 vs 1 | 1.0201 [0.958, 1.094] | 1.0201 [0.958, 1.094] |
| Sensitivity | 2022_06 | 1 vs 0 | 1.0126 [0.958, 1.075] | 1.0126 [0.958, 1.075] |
| Sensitivity | 2022_06 | 2 vs 0 | 1.0557 [0.996, 1.121] | 1.0557 [0.996, 1.121] |
| Sensitivity | 2022_06 | 2 vs 1 | 1.0426 [0.975, 1.109] | 1.0426 [0.975, 1.109] |

## Other matched numerical checks

- S6: refit all seven age/aggregate comparisons on 999 matched draws. All displayed frailty and KCOR intervals remain identical. The 40-49 dose-2 raw variance remains poorly resolved [0, 100]; its KCOR remains 1.507 [0.824, 2.712], with the existing fallback activated in 287/999 draws. All 14 point fits remain accepted, six with residual flags. Max absolute changes in raw variance percentiles and KCOR interval endpoints are 0.0000271 and 0.000000402 respectively. This is a genuine statistical resolution limit, not fixed by numerical scaling.
- Age-shift negative controls: the two endpoint estimates remain 1.004 [0.947, 1.053] and 1.027 [0.964, 1.091]. Across 169 weeks per curve, all point/interval bound changes are <1e-8 and residual flags remain 987 and 950 of 999 draws, respectively. All draws succeed; original and draws 1 and 42 verified against production. Existing plotted curves remain identical at figure resolution.
- Original zero-contamination study: 200 matched datasets per cohort size, true low/high variances 0.5 and 4, true KCOR 1. The intervals below are 95% Student-t confidence intervals for the across-dataset mean, not the percentile ranges of individual estimates. The exact same data are fitted before/after. Additional paired differences and individual-estimate quantiles are in the evidence folder.

| People per cohort | Previous mean KCOR [95% CI] | Current mean KCOR [95% CI] | Previous low-variance mean [95% CI] | Current low-variance mean [95% CI] |
|---:|---|---|---|---|
| 50,000 | 1.0495 [1.0306, 1.0684] | 1.0495 [1.0306, 1.0684] | 1.5480 [1.2533, 1.8426] | 1.5480 [1.2533, 1.8426] |
| 500,000 | 1.0091 [1.0027, 1.0154] | 1.0091 [1.0027, 1.0154] | 0.7174 [0.6072, 0.8277] | 0.7174 [0.6072, 0.8277] |
| 5,000,000 | 1.0013 [0.9994, 1.0033] | 1.0013 [0.9994, 1.0033] | 0.5161 [0.4695, 0.5627] | 0.5161 [0.4695, 0.5627] |

The persistent mean above 1 in the 50k and 500k simulations is not the numerical defect in 7.3. It remains explained by weak frailty information, boundary behavior and the sampled reference. The point correction is therefore justified independently of the existing statistical precision discussion, and does not redesign KCOR.

## Files

[Solver patch](solver.patch) | [Regression test](test.patch) | [Paper patch](paper.patch) | [Supplement patch](supplement.patch) | [Analysis README patch](analysis-readme.patch) | [Reproduction instructions](evidence/README.md) | [Complete Table 8 comparison](evidence/table8-before-after.csv) | [Clean PDF](KCOR-clean.pdf) | [Cumulative redline PDF](KCOR-redline.pdf).

**Round 8 independent disposition:** resolved within original scope; see [resolution review](../../iterative%20review/evaluations/8/resolution.md).

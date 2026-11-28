# Independent review of KCOR

The gamma-frailty inversion in §2.4.2 is correct for mean-one multiplicative gamma frailty. The identification proposition in §2.4.4 also follows by differentiating the marginal hazard under its stated fixed Gompertz slope and proportional survivor history. Those identities support a conditional, descriptive reference-frailty contrast. The most important correction concerns how the estimator handles a period of reduced mortality between two valid quiet windows. A second correction concerns workbook output after follow-up support ends.

## 1. High priority — a protective interval between quiet windows biases the frailty fit

A mortality reduction confined entirely to a gap can be attributed to a lower frailty variance. Section 2.5 reconstructs the difference between actual cumulative exposure and Gompertz exposure at re-entry, then forces the increment in that difference to be nonnegative. The total exposure accumulated during a gap must be nonnegative. Its difference from the planned Gompertz exposure can have either sign. A protective period leaves more frail people alive at re-entry, just as an adverse period leaves fewer. Choosing quiet weeks on either side does not remove this asymmetry.

I evaluated the production proportional fitter on exact weekly integrated hazards, without sampling error. Take 120 rebased weeks, weekly initial rate 0.002, slope 0.085/52, mean-one gamma variance 4, and four early anchor weeks. Use quiet weeks 0–9 and 70–119. Multiply individual hazards in weeks 10–69 by the indicated factor and restore the original baseline thereafter. For weekly baseline exposures B, the input is the successive differences of log(1 + 4B)/4. The comparator follows unperturbed Gompertz throughout. Both true and fitted endpoints are anchored at the end of week 3.

| Gap multiplier | True frailty variance | Fitted variance | True anchored endpoint | Fitted endpoint |
|---|---:|---:|---:|---:|
| 0.3 | 4 | 2.25257 | 0.66166 | 0.58848 |
| 1.0 | 4 | 4.00000 | 1.00000 | 1.00000 |
| 1.7 | 4 | 4.00000 | 1.33834 | 1.33834 |

All three returned numerical status `ok`. The protective case recorded a raw offset of −0.09247 and applied zero. The statistic exaggerates the reduction by 0.073 on the ratio scale. This example specifically violates the sign restriction on excess exposure. The gamma law, quiet-window Gompertz law, weekly observation model and quiet-window placement all hold. The issue therefore reaches beyond the warning about treatment effects *during* a quiet window in §5.2. See `code/KCOR.py` lines 1549–1563 and 1638–1651.

**Correction.** For an unmarked proportional gap, constrain the reconstructed gap exposure to be nonnegative and permit its departure from the planned exposure to be negative. Recheck exact benefit, null and harm examples through the same routine. If KCOR is intended only for nonnegative excess gaps, keep that restriction explicit in the abstract and applicability criteria, and designate a material negative gap as an unsupported normalization in the reported comparison. A signed deficit in a diagnostic log alongside an OK fit is insufficient.

## 2. High priority — the workbook reports a changing contrast after everyone in one cohort has exited

The ordinary code supports histories with nondeath exits (§2.9.1). Once the opening risk set is zero, a zero death count contains no information about the hazard. In `code/KCOR.py` the weekly rate is missing when person-time is zero. The normalization path converts the missing hazard to zero near lines 7848–7856. It consequently extends one cohort’s curve horizontally while continuing to accumulate deaths in the comparator.

I used the 56-week ordinary workbook fixture from `test/test_source_component_history.py`. In the 1950, dose-1 cohort, set both sex components’ opening risk and deaths to zero from April 4, 2022 onward, representing the remaining people exiting after the previous week. The source-history rules accept these reductions. By July 4 the workbook reported anchored KCOR **0.732363**, unanchored KCOR **0.732382**, normalization status `ok` and numerator fit status `OK`. There had been no person at risk in the numerator since April 4. A similar flat hazard after the last event has a different meaning when people are still under observation.

**Correction.** Retain the last supported cumulative estimate. Mark the affected cohort and contrasts unavailable from the first accumulating interval with zero opening risk, and exclude those weeks from fitting. Propagate that support boundary to pooled reporting and bootstrap draws. Exhaustion of a single sex or age component remains compatible with estimating an aggregate whose opening risk is positive. This defect concerns the exit-capable implementation. It does not imply the archived Czech death-only estimates encountered it.

## 3. Moderate priority — Table 8’s archived flag counts omit known failures of the residual screen

The current primary Table 8 JSON reports `flagged_draws: 0` for each enrollment. The replay accumulates only `diag["insufficient_signal"]` and does not pass opening risk sets for the weekly residual screen (`analysis/table8/table8_bootstrap.py`, function `estimate`, and `analysis/solver_scaling/table8_replay.py`, lines 74–79). The text in §3.1.2 narrowly specifies insufficient-signal flags and unsuccessful fits, but the archive field name and table do not tell readers how often the other contributing diagnostics fired. This matters for a method whose interpretation depends on those diagnostics.

The existing archive permits a direct check. For the 2021-24 module-off analysis I compared all 1,000 sets of the three inclusive-group fitted rates and variances in `analysis/ceiling500/table8/identified_only-draws.npz` with `analysis/age_shift/diagnostics.json`. The maximum absolute difference was zero. That inclusive group contributes to the weighted Table 8 ratios. Counting its recorded pair-specific residual flags gives **at least 987/999, 950/999 and 973/999** flagged draws for 1/0, 2/0 and 2/1 respectively. The other contributing groups can raise those counts. These are exploratory lack-of-fit flags. They do not invalidate the percentile arithmetic.

**Correction.** Rename the historical field to identify what it counted. Pass risks to the replay and tally residual flags, constraints, weak-identification statuses, and failed fits for the actual contributors to each contrast. Put the original fit and draw-level counts beside each interval, in both modes. Retain the existing draws in the interval calculation.

## 4. Moderate priority — the legacy count interval uses the wrong variance for the anchored statistic

Even with known zero frailty and independent cohorts, the analytical columns use an increment variance divided by the squared **post-reference increment**. The displayed statistic divides a cumulative hazard by its reference cumulative hazard. These have different sampling variances. For one cohort, write A for the reference cumulative hazard and D for the subsequent increment. Under independent-increment first-order approximations, the contribution to the variance of log[(A+D)/A] is

```text
(1/(A+D) − 1/A)² Var(A) + Var(D)/(A+D)².
```

The code uses Var(D)/D². For A = 0.01, D = 0.09 and rare independent Poisson increments with variances A/N and D/N, the respective values are 90/N and 11.11/N. Near the reference the discrepancy can go in the other direction. See `code/KCOR.py` lines 4473–4506. The workbook’s default percentile procedure refits the estimator and is separate from this problem. Its availability mask nevertheless depends on presence of the analytical lower bound in `code/bootstrap_reporting.py` line 230.

**Correction.** Retire the analytical columns as inferential intervals, or use the delta method for the actual anchored statistic with its reference covariance and state which fitted-parameter uncertainty is omitted. Base bootstrap availability on valid estimates, requested display horizon and complete bootstrap draws, independently of the legacy analytical calculation.

## 5. Moderate priority in sparse strata — pool membership can change without a draw-level warning

Section 2.8 documents pooling over available strata. The resulting number can refer to different age compositions in the point estimate and its bootstrap replicates. The implementation skips any age lacking a positive reference ratio and renormalizes the remaining weights (`code/KCOR.py` lines 4791–4842). The pooled status is the same `pooled_available_strata` when all ages contribute and when a positive-weight age is lost. The bootstrap counts an undefined pooled anchor only when there is no valid reference left.

In the same two-age workbook fixture, I set the 1950 dose-1 deaths through the reference week to zero and rebuilt its opening risks from the new death counts. At the endpoint, the 1950 ratio was undefined and the 1940 ratio was 1. The pooled result was 1, with normalization status OK and no abnormal-fit flag. If a sparse reference has one observed death, a draw losing that person is quite plausible. A finite draw then conceals the missing contributor.

**Correction.** Declare the positive-weight contributing strata for the contrast and use them in each replicate. For a fixed-composition interpretation, flag a draw with any missing required reference and apply the same incomplete-draw rule as for individual ages. If the intended estimand explicitly changes with availability, retain that descriptive statistic but report contributing strata, retained weight, and draw-level counts of composition changes so it cannot be read as a fixed standardization.

## 6. Low priority for the published configuration — accepted frailty ceilings below 0.5 break the fit

The configuration parser accepts any finite positive `theta0_max`. In both fitters a second optimization starts at exactly 0.5. Using a ceiling of 0.25, with exact data generated at k = 0.002 and theta = 0.1, the proportional fitter returned an error and missing theta. The marked-wave fitter raised `ValueError: Initial guess is outside of provided bounds`. See `code/KCOR.py` lines 1566–1573 and `code/kcor_survivor.py` lines 343–350. The default ceiling of 500 is unaffected.

**Correction.** Clip and deduplicate each start against the configured bounds. Preserve a successful first fit even when a supplementary start fails. Test one small positive ceiling in both modes.

## Interpretation of the empirical application

The supplement appropriately discloses the S6 product rule. Its use of future doses still restricts what the resulting comparisons can establish. Of 10,779,090 otherwise eligible people at 2021-24, the archive counts 363,033 excluded solely for a product in or after the enrollment week. Receiving a later disqualifying dose requires survival to its date. Fixing group labels at enrollment does not undo this eligibility selection. Its magnitude and direction on KCOR have not been measured. For an enrollment-defined population analysis, determine product eligibility from doses recorded before entry and recompute the same S6 contrasts. Retain the full-history result under its explicitly retrospective population definition. The archived Table 8 population predates this product restriction.

The tests above used the frozen manuscript, matching combined PDF, implementation, targeted temporary inputs and existing result arrays. No national analysis or large simulation was rerun.

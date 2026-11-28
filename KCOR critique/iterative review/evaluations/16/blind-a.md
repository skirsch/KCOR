# Independent review of KCOR candidate m177

**Question.** Examine KCOR closely. What, if anything, is wrong that matters to the validity of the method or its results, and how would you fix it? Distinguish substantive flaws from minor corrections, and do not propose unnecessary redesigns.

**Verdict.** The proportional estimator is mathematically coherent as a descriptive, model-conditional ratio. I reproduced the Table 8 primary intervals, the S6 primary point results from the person-level source, and recovery under a model with known unequal baseline rates, unequal frailties and two positive gaps. The most consequential outstanding issues concern which contrast the paper actually illustrates, how the example population was selected, and which claims its controls support.

## 1. Substantive reporting issue — the primary contrast and the contrast readers see can lie on opposite sides of one

Sections 2.1.1 and 2.8 define the primary contrast as the ratio of the two inverted cumulative hazards. The S6.1.2 outcome table and figures instead present that ratio divided by its value at the fourth accumulating week. They do label the anchoring. What the results omit is the magnitude of the difference. Applying the stated gamma inverse to the archived S6 full-precision raw cumulative hazards and fitted variances gives the following dose-2/dose-0 values for the same population, date and fit.

| S6 population | Primary unanchored ratio | Ratio at the reference week | Published anchored endpoint |
|---|---:|---:|---:|
| Ages 40–49 | 0.766 | 0.508 | 1.507 |
| Ages 50–59 | 0.520 | 0.424 | 1.228 |
| Ages 60–69 | 0.388 | 0.299 | 1.295 |
| Ages 70–79 | 0.336 | 0.258 | 1.301 |
| Ages 80–89 | 0.415 | 0.345 | 1.204 |
| Ages 90–99 | 0.545 | 0.538 | 1.012 |
| All eligible ages | 2.667 | 2.349 | 1.136 |

For example, the age 70–79 cumulative contrast increased by about 30% relative to the early reference while remaining below one. The direction differs in every displayed age band. In addition, the observed-hazard comparison in the adjacent table uses the opposite dose ordering. This is an interpretation problem of practical consequence even when the reader assigns neither contrast a causal meaning.

**Fix.** Put the primary ratio, reference ratio and anchored ratio side by side in this worked table, with one dose ordering and an explanation of one row. Match uncertainty to its own statistic if shown. Label the current fitted simulation evidence for the anchored outcome accordingly. The core estimator already supplies the necessary quantities. Sources are Sections 2.1.1, 2.8 and S6.1.2, `analysis/s6/results.json` and `analysis/gap_assessment/s6/s6-all-window.json`.

## 2. Substantive application issue — post-enrollment product eligibility changes the estimated frailty

The S6 source omits anyone with a disallowed product among the first four recorded vaccinations, including vaccinations after enrollment. A later such vaccination requires surviving until it is received. Therefore full-record eligibility can change both the level and shape of subsequent mortality, especially among people with zero doses at enrollment. The paper discloses the rule and counts the exclusions. It leaves their effect on the fitted quantities unevaluated.

I independently reconstructed the 2021-24 fixed cohorts from the compressed Czech registry file, applying the converter’s infection-row, birth-band, death-date and enrollment-week conventions. Under the full-history rule I matched opening populations and total deaths for all 14 cohort fits and recovered all seven archived anchored points to within 1e-8. I then excluded disqualifying products only when recorded strictly before the enrollment week, holding the other inputs and estimation settings fixed.

| Group or estimate | Full-record eligibility | Enrollment-time product eligibility |
|---|---:|---:|
| All ages, dose 0 at enrollment | 5,861,036 | 6,215,640 |
| All ages, dose 2 at enrollment | 1,838,302 | 1,843,153 |
| All ages, dose-0 fitted variance | 31.06 | 26.40 |
| Ages 40–49, dose-0 fitted variance | 63.87 | 37.75 |
| All ages, anchored ratio | 1.1356 | 1.1181 |
| Ages 90–99, anchored ratio | 1.0122 | 0.9971 |

The selection rule materially changes some fitted variances. The all-ages anchored endpoint shifts by 0.0174. The oldest group's point crosses one within its original wide interval. These are paired design sensitivities at the point-estimate level. Uncertainty for their differences was not computed. This finding concerns S6 and the product-restricted Czech extract. The archived Table 8 control predates this exclusion.

**Fix.** For an illustration of enrollment-defined cohorts, show this enrollment-time analysis and explicitly distinguish its population from the historical full-record selection. If retaining only the historical curves, keep the population-specific interpretation and give the above sensitivity. The frailty equations need no change. The focused reproduction and full age-by-slope point grid are in `work/kcor-review-checks/16/blind-a/eligibility_points.py` and `eligibility-refits.csv` outside the candidate.

## 3. Substantive limit on validation language — the empirical controls cannot establish model adequacy

In Section 3.1.2 the sentence saying the empirical negative control “evaluates end-to-end pipeline behavior under model-consistent conditions” assumes a property the exercise cannot establish. Remapping ages removes an assigned treatment comparison. Equal normalized trajectory shapes would additionally require proportional underlying age-specific trajectories over the waves. The later paragraph acknowledges this problem. Independently taking percentiles of the 999 resampled rows of `analysis/finite_reference/table8-primary/identified_only-draws.npz` gives exactly the six published interval pairs. The pair-level exploratory residual flag is set in all 999 draws for each of the six comparisons. Those counts are dependent resamplings and an exploratory screen, so they do not amount to 999 independent rejections. They do make an unqualified assertion of model consistency untenable. The outcome-informed historical window preference further limits claims of independent empirical validation.

Similarly, Table 9's benefit contrast violates the nonnegative excess-gap assumption. The archived interval for its signed treated-group gap is −0.04223 to −0.00686. The paper properly labels it audit-only locally, while S4.3 still credits the design with assessing both direction and timing and the conclusion counts end-to-end positive controls as method validation. An endpoint measured at week 80 establishes neither onset nor cessation of an effect. The figure offered alongside is a historical raw-count result.

**Fix.** Keep the numerical results and the useful scoped statements already in Sections 3.1.2 and 3.2. Make the overview, S4.3 and conclusion agree with those statements. Current evidence supports a checked implementation, one current matched direction for harm, a recorded benefit stress response and a conditional misspecification study. If a claim of benefit recovery under the stated model is important, a single labeled in-model example with the intervention after fitting would address it. A broader simulation redesign is unnecessary.

## Conditional assumptions and minor corrections

- The gamma inversion follows directly from the Laplace transform. With no gaps the fixed-slope identification identity follows by differentiating the marginal hazard. I checked the actual fitter on 120 exact weekly hazards with baseline rates 0.002 and 0.003, variances 2 and 3, annual slope 0.085, and two positive proportional shocks outside three fitting periods. It recovered both rates and variances, with maximum cumulative-exposure error below 2e-16, a primary ratio of 1.5000 and an anchored ratio of 1.0000. Identification of biological frailty or a treatment effect from the same counts would require further assumptions. The paper generally makes this distinction.
- Section 2.4.3 gives a 40-week illustration where perturbing the shared annual slope by ±0.020 barely changes the ratio. The corresponding S6 all-ages points for slopes 0.065, 0.085 and 0.105 are 1.1691, 1.1356 and 1.1274. Ages 90–99 give 1.0411, 1.0122 and 0.9838. Showing the actual sensitivity in the example is more informative than suggesting near cancellation generalizes. It requires only point refits and does not make the fixed slope itself a mathematical error.
- Sections 4.1 and 4.3 occasionally attribute observed residual differences or unvaccinated trajectories to depletion more categorically than the data allow. A fitted gamma component is one representation of the weekly mortality pattern. Use that conditional wording throughout. Sections 2.6.2 and 5.2 already explain why a good diagnostic fit cannot distinguish observationally matching alternatives.
- Section 2.1.3 says failed quiet-window criteria should prevent reporting contrasts, while Sections 2.1.2 and 4.4 allow expressly flagged contrasts for audit. Clarify that the restriction concerns substantive interpretation. The S6 lower-bound variance piles, modest S8 coverage studies, and non-identification of the Czech nonlinear exponent are disclosed. I found no reason to discard the proportional method or to promote the forced exponent to primary.

**Scope of verification.** I inspected the authoritative manuscript and SI, the proportional and optional survivor-law paths, bootstrap reporting, and study archives. The person-level reconstruction and baseline perturbations above are fresh point refits. I recomputed archived Table 8 percentiles without rerunning the full bootstrap. I did not estimate a causal vaccination effect, run new coverage simulations or infer the sampling significance of eligibility differences.

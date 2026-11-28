# Independent review of KCOR candidate m180

**Review question.** Examine KCOR closely. What, if anything, is wrong that matters to the validity of the method or its results, and how would you fix it? Distinguish substantive flaws from minor corrections, and do not propose unnecessary redesigns.

## Assessment

The gamma inversion follows from the stated mean-one gamma survival law. The fixed-slope Gompertz identification result also holds under its proportional-history assumptions. These are claims about each cohort’s own post-skip reference-frailty scale. They do not establish that frailty generated the observed curvature or that the reference subjects in two cohorts are exchangeable. m180 largely observes this distinction. I found no arithmetic error in the central Czech tables. Two empirical design choices deserve clearer, quantified treatment. Neither requires changing the general estimator.

## Substantive application issues

1. **The S6 eligibility rule selects on later vaccination and changes the fitted frailty appreciably.** The text accurately acknowledges the mechanism and reports the number excluded. Its quantitative consequences are accessible with the supplied full registry. I independently constructed weekly counts from the compressed Czech source, using first-infection records or participants with no infection, the converter’s birth-band and death-date exclusions, frozen dose strictly before the enrollment week, and all-cause deaths. I used the same four product-code fields for both policies. In the sensitivity population a disqualifying product excludes the participant only if already recorded before enrollment. I retained the original skip, quiet windows, baseline, fitter, reference and 2024-16 endpoint. The full-history construction reproduced all seven published anchored endpoints to better than 0.000001 and the added population was exactly 363,033, including 354,604 opening dose-zero members. Numbers below are point estimates.

   | Age in 2024 | Published full-history eligibility | Enrollment-time eligibility |
   |---|---:|---:|
   | 40–49 | 1.50749 | 1.48687 |
   | 50–59 | 1.22756 | 1.20663 |
   | 60–69 | 1.29504 | 1.27854 |
   | 70–79 | 1.30135 | 1.27894 |
   | 80–89 | 1.20424 | 1.17983 |
   | 90–99 | 1.01216 | 0.99706 |
   | All eligible ages | 1.13555 | 1.11815 |

   The dose-zero fitted variance changes from 63.87 to 37.75 at ages 40–49, from 26.82 to 15.91 at ages 50–59, and from 31.06 to 26.40 for all ages. Both all-ages residual screens still flag. Later receipt of a disqualifying product requires survival until receipt, and the filter removes many more initially unvaccinated people. The point differences reflect the combined change in eligibility and composition. They cannot be interpreted as an estimate of a single causal selection-bias term. The sensitivity is moderate for the cumulative contrast and larger for the fitted variance. The crossing at ages 90–99 was already consistent with the original uncertainty. Report the enrollment-time sensitivity immediately alongside S6.1.2, specifying which population each column describes. If the scientific population is meant to be defined on enrollment day, use the enrollment-time rule for that application. The archived age-shift control uses another extract and is unaffected. Sources include `code/KCOR_CMR.py`, `analysis/eligibility/audit.py`, `analysis/s6/replay.py`, and the S6 results/draws.

2. **The February 2022 Table 8 primary fit assumes a quiet baseline during four weeks that the same application marks as a wave.** The proportional primary fit includes the first four accumulating weeks as baseline observations even when they are outside the designated quiet windows. For enrollment 2022-06 they are ISO weeks 2022-08 through 2022-11, all inside the marked wave ending 2022-12. Section 3.1.2 discusses the post-origin survivor assumption for forced alpha, though the primary column also depends directly on these wave-period baseline anchors. The primary/sensitivity column difference bundles two changes, the baseline fitting rows and the nonlinear survivor correction.

   I used the archived Table 8 input and matched original-point code, supplied exactly the same marked dates with exponent 1, and kept the reference, date, overlap weights and proportional inversion fixed. For 2022-06, the primary, marked-unit-exponent and archived forced-1.18 points were 1.00695, 1.00830, 1.01255 for 1/0. They were 1.01642, 1.02888, 1.05566 for 2/0 and 1.00940, 1.02041, 1.04257 for 2/1. For 2021-24 the marked-unit-exponent points reproduced the primary column to within 0.00000002. The marked exercise changes how the entry period is fitted and attributed, while the final gamma inversion remains identical at fixed parameters. The data do not prove excess mortality in every marked week. Explicitly flag the primary anchor assumption for the affected enrollment, and report the marked, proportional sensitivity for any comparison of the two existing columns. This is an application-specific limitation. The reported table already confines interpretation to descriptive audit.

## Precision and validation

The common age slope is a consequential modeling assumption even where the ratios are less sensitive than the variances. With the published S6 population and everything else fixed, varying the annual slope from 0.065 through 0.085 to 0.105 changed all-ages anchored KCOR from 1.16909 through 1.13555 to 1.12740, and ages 90–99 from 1.04111 through 1.01216 to 0.98378. Ages 40–49 barely changed, 1.50743 through 1.50754. These results explain when a common change cancels between groups. The 0.4% example in §2.4.3 is correctly qualified as a specific synthetic case. An empirical point range would implement the sensitivity check already recommended in S5.3. I did not fit intervals for the alternative slope, wave or eligibility specifications.

I independently took 2.5th and 97.5th percentiles of draws 1–999 in the primary and archived sensitivity Table 8 arrays. They reproduce the reported limits at display precision. The estimates near one give no evidence of equivalence. All six primary comparisons had at least one residual-flagged contributing fit in every draw. The current text correctly reports this and distinguishes it from numerical failure. The archived Table 9 fitted estimates 0.83359 [0.77412, 0.89705] and 1.16170 [1.08652, 1.24845] support a directional illustration. The signed deficit in the benefit case prevents counting it as validation under the nonnegative-gap model. The optional Czech exponent is unidentified and the forced analysis is appropriately a sensitivity. The Cox simulation uses supplied generating variances and therefore isolates inversion. It cannot establish end-to-end recovery of fitted frailty. The paper already discloses these facts. Preserving these qualifications matters more than adding comparator models.

## Minor corrections

- The S4.0 parameter table assigns a Gompertz estimator and frailty-recovery grid to S4.2.1 alongside the historical log-hazard relaxation example. Identify the separate simulation/figure for each row, as the actual S4.2.1 captions do.
- The §5.2 sentence directing readers to §3.4 for frailty misspecification and quiet-window contamination should refer to §3.3.
- The headline comparison in Table 7 would be easier to interpret if the KCOR column or caption explicitly said “generating frailty variance supplied,” as §2.11.1 already does.

No full bootstrap or manuscript/source modifications were made in this review.
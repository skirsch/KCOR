# Independent review of KCOR candidate m180

> Examine KCOR closely. What, if anything, is wrong that matters to the validity of the method or its results, and how would you fix it? Distinguish substantive flaws from minor corrections, and do not propose unnecessary redesigns.

## Verdict

KCOR defines a coherent, conditional descriptive contrast. Under mean-one gamma frailty and proportional individual hazard, the survival law gives the stated inverse exactly. Under a known quiet Gompertz slope, the relation between cohort hazard, its derivative and frailty variance also follows. The fitted rate and frailty depend on the assumed quiet baseline, and the reference frailty is the mean of each cohort at the rebased origin. Neither algebraic result establishes exchangeability or distinguishes changes in baseline risk from depletion in the registry. The main text generally respects these limits.

The published Czech point estimates reproduce. Two consequential choices in the S6 illustration can now be quantified. Both change the oldest group's point estimate across one. Its published interval already spans one. Neither finding invalidates the reported statistic for its declared population and slope. I would add the empirical sensitivity results to S6 and keep its existing interpretation as a worked example.

## Two empirical sensitivities

I reconstructed the 2021-24 dose-2/dose-0 comparison through 2024-16 using the candidate's inputs, configured windows, two-week skip, fourth accumulating week as reference, all-window proportional fitter and no optional nonlinear correction. The middle column below is an independent replay of the published specification. The slope columns hold population eligibility fixed. The last column changes only the product-eligibility rule, holding the slope at 0.085 per year.

| Age in 2024 | Full history, slope 0.065 | Published specification, slope 0.085 | Full history, slope 0.105 | Enrollment-time product eligibility, slope 0.085 |
|---|---:|---:|---:|---:|
| 40–49 | 1.50743 | 1.50749 | 1.50754 | 1.48687 |
| 50–59 | 1.26344 | 1.22756 | 1.19239 | 1.20663 |
| 60–69 | 1.33311 | 1.29504 | 1.25774 | 1.27854 |
| 70–79 | 1.33985 | 1.30135 | 1.26364 | 1.27894 |
| 80–89 | 1.23950 | 1.20424 | 1.16969 | 1.17983 |
| 90–99 | 1.04111 | 1.01216 | 0.98378 | 0.99706 |
| All eligible ages | 1.16909 | 1.13555 | 1.12740 | 1.11815 |

**1. Moderate, empirical population selection.** The primary S6 rule uses disqualifying products recorded after enrollment. A person has to survive to receive a later disqualifying product. People who die sooner can remain eligible, and the removed population is concentrated in the initially unvaccinated group. The manuscript discloses the mechanism and counts affected people, but S6.1.1 states that the alternative has not been evaluated. On the full registry I retained the original birth bands, infection-record and date-consistency filters, opening risk definition and fixed dose assignment. I compared the published exclusion for any disqualifying product in the first four recorded doses with exclusion for a disqualifying product in those doses dated strictly before enrollment. I aggregated weekly deaths and reconstructed opening risk by subtracting preceding deaths. Under the published rule my 14 opening counts agree with the archived S6 counts, and every endpoint agrees within 0.000001. Under enrollment-time eligibility the all-age opening count increased from 5,861,036 to 6,215,640 for dose 0 and from 1,838,302 to 1,843,153 for dose 2. Every anchored endpoint decreased by approximately 0.015–0.024. These differences quantify sensitivity to the choice of study population. They do not measure causal bias. Show the two specifications side by side and retain the original label for the published population. The archived age-shift extract predates the restriction.

**2. Moderate, fixed-slope interpretation.** Section 2.4.3 gives a 40-week synthetic example in which perturbing the annual Gompertz slope by 0.020 changed the endpoint by about 0.4%. SI S5.3 recommends this diagnostic. I refitted the Czech data at exactly those perturbations using data/Czech/KCOR_CMR.xlsx and the current code/KCOR.py. The 40–49 result was virtually invariant despite substantial changes in both fitted variances. In the 70–79 group, where the dose-2 variance was at the lower boundary, the point estimate ranged from 1.264 to 1.340. The oldest group's primary 95% interval was 0.919–1.114, so the change across one does not overturn a precisely resolved direction. The all-age point remained above one at all three slopes. Report these as conditional point sensitivities and leave the synthetic example scoped to its design. The existing fitter suffices for this check.

## What the validation supports

- **Table 8.** I recomputed all six primary percentile intervals from row 1 onward of analysis/finite_reference/table8-primary/identified_only-draws.npz, excluding the point estimate in row 0. Both limits for every comparison exactly matched identified_only-table8.json and agreed with the displayed rounding. Each comparison had at least one residual-screen flag in 999/999 draws. The six primary intervals include one, the control has no known exact anchored target, and absence of a significant difference is insufficient to establish equivalence or correct normalization. Section 3.1.2 and SI S10 already give the appropriate qualifications.
- **Fitted versus known frailty.** The Cox comparison uses the simulation's known variance in KCOR inversion. It demonstrates a difference in targets and checks inversion. Evidence for performance after estimating frailty comes from the separate sampled studies, including SI S9. The distinction is explicit in section 2.11.1 and should also appear in the stand-alone Cox comparison table caption. This is a minor reporting correction.
- **Gaps, uncertainty and optional wave model.** The proportional gap constraint requires nonnegative additional exposure. The Table 9 benefit injection violates it and is explicitly audit-only. A residual or gap screen with little information cannot confirm the identification assumptions. Person-bootstrap percentiles propagate sampling variability under the specified design and fitter. They do not incorporate baseline-shape choice or retrospective eligibility selection. The nonlinear-wave survival equation is coherent for supplied inputs. The approximate procedure proposed to localize the exponent from weekly proxies has no established empirical identification here. The current Czech primary analysis keeps it inactive.

## Scope of checks

I inspected documentation/preprint/paper.md, documentation/preprint/supplement.md, the proportional and nonlinear fitting/inversion paths, the converter, person-bootstrap and aggregate-history handling, the Table 8 draw arrays, and the S6 point and diagnostic artifacts. The eligibility check used the candidate's full compressed Czech registry and aggregate output only. The two sensitivity exercises are separate point-only refits. I did not recompute 999-draw intervals for changed populations or slopes, so the table supports no claim about a new uncertainty interval or significance. No candidate file was edited.

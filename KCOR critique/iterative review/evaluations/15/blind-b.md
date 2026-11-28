# Independent review of KCOR, candidate m173

**Review question.** Examine KCOR closely. What, if anything, is wrong that matters to the validity of the method or its results, and how would you fix it? Distinguish substantive flaws from minor corrections, and do not propose unnecessary redesigns.

The ordinary workbook can give a finite estimate and bootstrap interval for the wrong all-ages population in a declared partial extract. This is a substantive implementation defect. I found no corresponding error in the published Czech S6 numbers, which I independently reproduced from the source registry. The main weakness in their presentation is that the declared primary ratio is missing from the worked example and the displayed anchored ratio can fall on the opposite side of one. The method is defensible as a descriptive contrast conditional on its specified frailty and baseline models. The current evidence does not support a causal interpretation, a generally calibrated 95% interval, or an identified empirical wave exponent.

## 1. Substantive workbook error — partial extracts can misidentify the all-ages population

When an ordinary partial extract contains older disjoint birth groups and an independently supplied, wider born-1960-or-later total (`-60`), the workbook constructs an all-ages total (`-2`) from the displayed disjoint ages alone if no `-2` was supplied. This omits known younger people. If both totals are supplied, their compatibility is never checked against each other. The code compares each individually against the disjoint display. See `code/KCOR.py` in `_all_age_source_mask` and the pre-display reconstruction within `process_workbook` (around lines 4296 and 6448), and `code/fixed_cohort_history.py::aggregate_reconciliation`.

**Focused reproducible check.** I used the existing `test_round7_workbook.input_frame('2023_01')` fixture, retained its 1940 group, and used the same weekly counts to define a distinct younger group. For dose 1 in the younger group I used opening N=40,000, multiplied deaths after row six by four, and rebuilt the death-only risk sets. With this younger population presented as `-60`, disjoint partial mode produced the following week-20 anchored dose-1/dose-0 results through the ordinary Excel entry point.

| Displayed result | KCOR | Status |
|---|---:|---|
| Actual 1940 group | 1.003326 | OK |
| Supplied younger population | 3.367641 | OK |
| Automatically constructed “all ages” | 1.003326 | OK |
| All ages when the identical younger people were supplied as a disjoint 1960 group | 2.183459 | OK |

With three diagnostic bootstrap draws, the erroneous all-ages result received three valid draws and bounds [0.849748, 1.264642], exactly the older-only interval. The few-draw bounds demonstrate the code path only. When I additionally supplied an explicit all-ages row identical to the older counts, the code also accepted it, although that all-ages population could not contain both the known older group and the supplied younger group.

**Minimal fix.** In partial mode, withhold or explicitly label the displayed-subset aggregate whenever exhaustiveness cannot be established. Where the wider younger subtotal and the complete remaining partition define the full population, add those nonoverlapping populations once. Check nested totals together, including the nonnegative death/risk history remaining after subtracting the younger subtotal and known older components. Decide bootstrap support for the same population actually reported. Keep supported disjoint results. Add a regression covering the point, draw accounting and nested-total conflict. The published S6 application has a verified supplied all-ages total, and Table 8 has a different explicitly declared overlap layout. This defect provides no basis for changing either numerical table.

## 2. Material results-reporting gap — show the declared primary ratio in S6

Sections 1.6, 2.1.1 and 2.8 define the primary quantity as the unanchored ratio of normalized cumulative hazards. Anchoring divides it by the early ratio. S6 presents the anchored dose-2/dose-0 estimate plus a separate *unnormalized* raw-hazard table in the reverse direction. Neither reports the primary quantity. The labels in the existing tables do identify their own estimands, but a reader cannot see the direction and size of the primary result without doing another calculation.

Applying the specified inversion to `analysis/s6/results.json` gives the following dose-2/dose-0 values at 2024-16. For example, among ages 70–79, normalized cumulative hazard is approximately one third as high in dose 2, while the ratio increased about 30% relative to its early value. Both statements are true in the specified model.

| Population | Primary, unanchored | Anchored, as published |
|---|---:|---:|
| Ages 40–49 | 0.766 | 1.507 |
| Ages 50–59 | 0.520 | 1.228 |
| Ages 60–69 | 0.388 | 1.295 |
| Ages 70–79 | 0.336 | 1.301 |
| Ages 80–89 | 0.415 | 1.204 |
| Ages 90–99 | 0.545 | 1.012 |
| All eligible ages | 2.667 | 1.136 |

**Minimal fix.** Show the unanchored endpoint along with the anchored endpoint in at least the principal worked example, with their corresponding pointwise intervals if presented as inferential results. Compute them from the matched per-draw hazards and fits; anchored bounds cannot simply be rescaled by the original-sample anchor. Label the absolute level of the reference-frailty hazard ratio and its change relative to the prespecified early ratio. The existing estimator and the numerically correct anchored table can be retained.

## 3. Data-dependent scope of the Czech illustrations

I read the full compressed Czech source records in chunks, reproduced the enrollment rule, frozen dose, birth grouping, LPZ death week and product restriction, and independently rebuilt the weekly death-only counts for enrollment 2021-24 through week 2024-16. I used the current ordinary proportional fitter, its four configured quiet windows, two-week skip and fourth accumulating-week reference. All seven published S6 anchored point estimates reproduced at displayed precision. I then changed one design decision at a time. The numbers below are sensitivities of point estimates; no additional bootstrap was performed.

* **Retrospective product eligibility.** With products classified from the history available before enrollment instead of all first-four-dose records through follow-up, the added dose-0 population was 354,604. At the default annual slope, the all-ages anchored estimate changed from 1.135551 to 1.118148, and ages 90–99 changed from 1.012162 to 0.997062. Dose-0 fitted frailty for ages 40–49 fell from 63.87 to 37.75, while that age band's anchored ratio moved from 1.507487 to 1.486873. The manuscript correctly discloses future-based selection and quantifies membership changes. Its effect on the fitted parameters and the contrast can also be measured directly. If the intended population is defined at enrollment, use the enrollment-time eligibility rule. If the retrospective population is intended, retain that definition and report the alternative as sensitivity. These results give no evidence of a direction-changing overall causal effect.
* **Fixed Gompertz slope.** In the published population, annual slopes 0.065, 0.085 and 0.105 produced all-ages estimates 1.169085, 1.135551 and 1.127397. Ages 70–79 gave 1.339850, 1.301345 and 1.263635. Ages 90–99 gave 1.041105, 1.012162 and 0.983779. The last row's published interval includes one. Section 2.4.3's 0.4% toy-simulation result is properly scoped, but its cancellation does not quantify the empirical sensitivity. Report the observed range where S5.3 recommends checking the slope. It is not necessary to estimate the slope and frailty jointly from the same mortality curve.

## 4. What the validation supports

The gamma inverse and the quiet-interval identification formula are consistent with the working model. With a known positive Gompertz slope and an exactly proportional survivor history, the instantaneous marginal hazard obeys the stated differential equation. Counts alone cannot distinguish the corresponding depletion explanation from a different time-varying baseline. The paper discloses that assumption. The same holds for the absence of protection in unmarked gaps under the constrained fit and the need for an independently supplied wave scale in the proposed localization procedure. The optional exponent is explicitly unidentified in the Czech analysis.

I recomputed all twelve Table 8 percentile intervals from the archived matched draws after excluding draw zero. Both columns agree with their numerical JSON archives, including the disclosed preceding-seed provenance for the forced-alpha analysis. Inclusion of one is consistent with a range of departures. The overlapping pseudo-cohorts have no known exact null, and exploratory residual flags occurred throughout the primary draws. I also recomputed S9 means and coverage counts from its independent-dataset results. The six reported coverage counts, 98, 99, 94, 92, 96 and 96 out of 100 for the respective laws/resampling schemes, agree. This is conditional simulation evidence at that design.

The methodological workflow is less complete for the 2022-26 S6/S5.6 figures than for the S6 2021-24 table. The published figure replay stores numerical fit status and residual flags but omits the signed gap screen from its returned diagnostics (`analysis/s6-figures/replay.py::fit`). The configured windows leave unmarked gaps. State that the gap assumption was unassessed for those figure curves, or compute the screen on matched draws before claiming assessment. I found no evidence this omission changes the plotted points or their archived intervals.

## Minor corrections

* The S5.6/S6.1.3 figure captions say “at four weeks.” Use “end of the fourth accumulating week following the two-week skip” to fix the calendar origin. The age-shift figure caption already gives both event and calendar coordinates.
* For a bin with event fraction below one, the negative log of conditional interval survival is the integrated hazard even when the within-bin hazard varies. A constant hazard is needed to interpret an average as a constant instantaneous rate. This clarification in section 2.3 would not change the computation.
* The text calling the first-pass seed diagnostic unavailable at “a numerical bound” could specify the numerical ceiling for frailty and both rate bounds. The code allows zero frailty, which is a structural parameter boundary, for this diagnostic.

The review used candidate m173 only. Focused workbook fixtures, independent full-record point refits, and checks of stored resamples were performed. No full bootstrap or independent reconstruction of the historical forced-alpha draws was attempted.

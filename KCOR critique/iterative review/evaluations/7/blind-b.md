# Independent review of KCOR

I reviewed the frozen m128 manuscript sources, the combined 69-page clean PDF, the estimator and workbook reporting paths, and the archived analyses. I used small point-estimate replays, aggregate-count checks, a controlled error-path test and the stored bootstrap draws. I did not rerun a full simulation study or bootstrap.

## Scientific interpretation

The gamma inversion and the fixed-slope identification proposition are mathematically sound under their stated assumptions. For example, the observed instantaneous hazard in a proportional quiet interval solves h′ = γh − θh². With a known slope and an exact trajectory this identifies the gamma variance. Observational counts alone cannot tell whether that slope and frailty law generated the trajectory. Given any observed cumulative hazard H and candidate positive variance θ, the baseline B = expm1(θH)/θ reproduces exactly the same survival. A quiet-window Gompertz restriction supplies the missing information. A good fit can therefore support a conditional descriptive comparison, but cannot establish that selective mortality generated the observed shape. The Czech future-dose product eligibility rule and outcome-informed history of window choice further limit the empirical interpretation. The manuscript discloses both. The age-shift exercise has no known exact null for age-specific mortality trajectories.

I found several implementation and reporting problems that can be corrected while retaining the working model.

## 1. The ordinary workbook can report a substitute estimate after normalization fails

The normalization loop in `code/KCOR.py` (around lines 7966–8076) is inside one exception handler for the enrollment sheet. The normalized hazards are assigned to the dataframe only after the entire loop finishes. An exception logs a warning and processing continues with the previously assigned hazards. The stored fit statuses and the published ratio fields do not identify this substitution. Bootstrap workers also redirect ordinary-run output.

I tested this failure path on a scratch workbook using the 1950-labeled subgroup of `analysis/table8/input.xlsx`, sheet `2021_24`, pseudo-dose 2 versus 0, with the specified exponent 1.18. Successful normalization gave anchored KCOR **0.897540** on December 26, 2022. Making the correction routine raise an arithmetic error gave **0.887304**, with the same reported frailties (6.511445 and 0.745943), both fit statuses **OK**, and a blank abnormal-fit field. The second run logged the normalization failure. This was a deliberate test of error handling. It supplies no evidence of such a failure in the archived Czech results.

**Remedy.** On a correction or inversion failure, mark the affected normalized series and dependent contrasts unavailable, and propagate that status to the ordinary and bootstrap outputs. Retain successful cohorts. Keep any requested raw comparator in a separately named column. Add a regression test that forces one correction failure after a successful fit.

## 2. Row position can silently replace the prescribed calendar week

The ordinary processing path computes ISO enrollment from the first Monday in January (`code/KCOR.py`, around 6622–6638), then numbers rows within each group using `cumcount()` (6730). An ISO year can begin in December. A missing weekly row can also compress follow-up. Both change stabilization, baseline timing and the reference without warning.

I ran three small ordinary workbooks with the same settings and integer risk histories. A complete `2020_24` sheet started June 8, but the program discarded that week, began June 15 and labeled July 20 as a valid reference instead of July 13. For `2021_24` the complete history correctly anchored on July 19. Removing the July 12 row from both groups changed the reported reference to July 26, still with `anchor_status=ok`. The latter input is incomplete, which is precisely what the ordinary estimate needs to disclose. The bootstrap history validator does check the weekly calendar. The complete 2021 and 2022 histories used for Table 8 avoid the ISO-year error.

**Remedy.** Use `fromisocalendar(year, week, 1)` throughout and derive event time from the elapsed date. Validate the common weekly grid before fitting and accumulating. Missing mortality histories require an explicit unavailable result or a documented reconstruction from sufficient counts. Test both a December-opening ISO year and an internal gap.

## 3. The core optimizer can stop far from the known optimum on rare-event hazards

The proportional optimizer passes weekly hazard residuals in their original units to bounded least squares (`code/KCOR.py` around 1380–1388 and 1531–1566). The nonlinear fitter already rescales its objective. The absolute stopping tolerance makes the proportional solution depend on residual magnitude even when noiseless data distinguish the parameters.

For 12 consecutive quiet weeks I used γ = 0.085/52, k = 0.00001, θ = 50, four fitting anchors and exact weekly differences of log(1 + θkF)/θ, where F(s) = expm1(γs)/γ. The production `fit_theta0_gompertz` returned **θ = 42.03697**, with status **ok**. Starting at that solution and dividing the same residual vector by the mean weekly hazard gave **θ = 50.00000**. Unscaled residual sum of squares fell from 9.32 × 10⁻¹⁷ to 3.59 × 10⁻³⁹. Rounding large expected counts to integers produced the same numerical discrepancy (42.035 versus 50.0003). The count scale in that auxiliary check isolates solver error. Ordinary short-window data can also have genuine statistical imprecision, and these tests do not establish an error of this size in a published endpoint.

**Remedy.** Divide proportional seed and final residuals by a fixed positive scale calculated once from the observations, or use an equivalent scale-aware convergence check. This leaves the least-squares minimizer unchanged. Verify exact recovery at low rates and evaluate the existing points before changing published values.

## 4. The ordinary all-ages row double counts people in the supplied overlapping control

In the Table 8 workbook, the labels 1940 and 1950 are vaccination subgroups and 1960 is the group including everyone. The pseudo-dose labels encode age bands. The dedicated resampler correctly subtracts the two subgroups from the inclusive group, samples disjoint people and reconstructs all three series. The ordinary point-estimate path instead sums all positive birth-year labels when it synthesizes the direct all-ages cohort (`code/KCOR.py` around 7143–7164 and 7685–7715).

On June 14, 2021, ordinary synthesis gives opening counts **593,535, 1,731,510, 2,238,923** across the three pseudo-dose categories. The inclusive input counts are **310,466, 935,607, 1,265,854**. Under the current core fit, the reconstructed direct pseudo-dose 1/0 anchored contrast on December 26, 2022 is **1.01126** using the duplicated counts, versus **1.00394** using everyone once. The published Table 8 *pooled* contrasts and the age-shift figures use other paths. Their reported values do not depend on this extra direct row.

**Remedy.** Route direct aggregation through the layout already used for resampling. Use the supplied inclusive group once or omit this redundant row in the known overlapping layout. Explain in the Table 8 caption that its pooled ratio combines overlapping *vaccination subgroups*, including their inclusive group, using the program's weights. “Pooled/ASMR” invites readers to mistake the labels for independent age strata or a population-standardized mortality rate.

## 5. The Czech supplementary figure and its diagnostic claim exceed the evidence

**Figure S10.** In the clean PDF (p. 69), the all-ages graph runs from July 2022 into autumn 2024. The caption says the evaluation is over calendar 2023 and that the x-axis is weeks since enrollment. The actual ticks are calendar dates. The orange dose-3/dose-0 curve is above one near the end of 2023 and at its plotted endpoint. I independently grouped the supplied all-ages rows in `data/Czech/KCOR_CMR.xlsx`, sheet `2022_26`, by date and dose, and used the current module-off fit, the configured quiet mask, two skipped weeks, four-week reference and observations through October 7, 2024. It gave dose-3/dose-0 **0.97316** at the end of 2023 and **0.97615** at the last date. The corresponding dose-2/dose-0 estimates were **0.98317** and **1.00091**. Thus simply changing the axis caption does not make the figure a replay of that specification. Section S6.1.3 attributes the example to revised settings but does not supply an alternative executable specification that accounts for the discrepancy.

**Diagnostic acceptance.** Section S6.1 opens with a claim that the application satisfies all prespecified diagnostic criteria. S6.1.2 says six of fourteen fits triggered exploratory residual flags and window-omission stability was not evaluated. Section 2.6.2 explicitly includes omission stability among the checks. A residual flag is exploratory and need not reject the fit. An unevaluated criterion still prevents a claim that every requirement passed.

**Remedy.** Regenerate S10 and verify the three companion age-band figures against saved numeric trajectories. State the fitting dates, displayed dates, reference, mode and actual axis units. Alternatively identify a historical curve with its generating inputs and scope. Replace blanket diagnostic acceptance by the specific performed checks, flagged fits and unevaluated criteria. If full acceptance is needed for the application, supply the missing stability results.

## 6. Rebuilding the paper and reproducing its studies are different deliverables

Section S7 says `make paper` regenerates all artifacts. The root Makefile target at line 419 builds the combined PDF and TeX from prose and existing figures. It does not run the scientific studies. The S8 nonlinear-wave study supplies paired interval CSVs and summary JSON in `analysis/survivor_validation/`, but I could find no generating program or command for its 100-dataset main study and 50-dataset stress study in this snapshot. The directory contains data products only. By contrast, Tables 8–9 and the newer age-shift figures have identifiable analysis runners.

**Remedy.** Describe `make paper` as a rendering command. For each study claimed to be rerunnable, give its entry point and fixed settings. Supply the S8 generator with its seeds and bootstrap settings, or label S8 as archived evidence that readers can audit from the stored intervals.

## Remaining interpretation and checks

- Table 3 (PDF p. 44) marks “conditions on survival” as “No” for KCOR. Section 2.3 constructs the estimator from hazards in surviving risk sets. The final contrast is cumulative and, under mean-one gamma frailty, B is cumulative hazard for reference frailty Z = 1. It differs from the observed population cumulative risk. Clarify the table row and Box 2's “marginal” label by specifying which population or reference is being contrasted.
- The Cox experiment supplies KCOR with the generating frailty variances. A non-unit marginal Cox summary with unequal frailty distributions and a unit fixed-frailty contrast does not establish bias in RMST or in a Cox summary for its own target. Section 2.11.1 already explains much of this distinction. Align the abstract's claim about “biasing standard survival estimands” and the unqualified entries in the comparison tables with that explanation.
- The Table 8 archived point and percentile arithmetic checks out. For both modes and both enrollments I recomputed the archived point from draw 0 and the two quantiles from draws 1–999. All agreed exactly with the corresponding summary JSON at stored precision. The forced-exponent intervals remain archival sensitivity results, as the caption specifies. Precision studies using earlier fitters, simulations of selected regimes and mostly near-one empirical point estimates do not establish general coverage or eliminate application-specific selection.

The priority repairs are the normalization failure path, the calendar calculations and the low-rate solver. The empirical figures, overlapping-cohort row and reproducibility claims each have a narrower remedy. None requires replacing the gamma identity or interpreting a retrospective contrast as a treatment effect.


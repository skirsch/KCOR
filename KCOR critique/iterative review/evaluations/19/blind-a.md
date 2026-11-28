# Independent review of KCOR, frozen candidate m183

## Verdict

The core proportional gamma inversion follows from the stated survivor model. The material errors I could verify concern the record-level Monte Carlo converter and its interval reporting. One additional validation discrepancy affects the point estimator when given impossible overlapping control counts. I found no evidence that any of these errors changed the six published primary Table 8 values. I did not repeat the full bootstrap.

## 1. Monte Carlo confidence intervals can combine different follow-up weeks and omit undefined draws

The normal write path calls `create_mc_summary` for the Monte Carlo log. It selects the closest available date independently for every iteration, even when the target lies outside an iteration's follow-up. It then drops missing KCOR values, takes one finite value per iteration, and computes percentile and basic intervals on the remaining values. The grouping key omits the selected date. With a changed age signature, it can also print a percentile interval for a separate group containing draws but no original estimate. See `code/KCOR.py`, lines 8312–8369 and 9942–10069.

I ran the workbook writer on three fixed-cohort iterations using the supplied 2023-01 fixture. The original had 20 weeks, draw 1 had 12 weeks, and draw 2 had 20 weeks with no numerator deaths through the reference week. Each history satisfied death-only accounting. The all-ages point was 1.5795 on May 15. The only finite bootstrap endpoint was 1.4242 on March 20. Draw 2's anchored endpoint was undefined. The log reported a percentile 95% CI of [1.4242, 1.4242] and a basic 95% CI of [1.7348, 1.7348]. Neither is an interval for the original endpoint. The worksheet did retain the unavailable value at its proper date.

**Correction.** Choose one reporting week for the original analysis, record it, and require that week and population for every draw. Keep the full requested draw roster, including failures before a workbook row is produced. Show the counts by reason and withhold inferential bounds when the point or a required draw is unavailable. The ordinary bootstrap already implements the underlying complete-draw rule. The default `make monte_carlo` uses four iterations including the point, so a clean default run gives only three resamples. Its interval should be presented as a demonstration unless a suitable resample count is requested.

## 2. The record-level Monte Carlo converter silently drops requested birth cohorts

The master population honors `MC_YOB_START` and `MC_YOB_END` at `code/KCOR_CMR.py:648`. The weekly grid then unconditionally filters observed birth bands to 1930–1960 at line 928. The reported inclusive totals are sums of this grid. Reconciling them to its remaining constituents cannot detect the omitted people.

I executed the unchanged `process_enrollment_data` function on 12 eligible people from the 1940 and 1970 birth groups, with all three configured doses represented, six deaths during follow-up, and an intended range spanning both birth groups. Its returned enrollment population contained 12 people. The weekly output contained only the 1940 group, an opening risk set of six, and three deaths. This affects an advertised range override. It is not evidence of a change to results using the default 1930–1960 range.

A separate sparse-input failure arises in the same function. Removing the highest-dose members while keeping the same configured cap produced `ValueError: Length mismatch: Expected axis has 3 elements, new values have 4 elements`. The pivot is renamed as though every configured dose must have appeared, at lines 1000–1008. The ordinary converter already preserves actual dose-column names and supplies zeros for missing categories. The Monte Carlo driver catches failed iterations, combines whichever files exist, and still prints a completion message, lines 1529–1565. A genuine sampling zero can therefore remove a whole draw for otherwise supported comparisons.

**Correction.** Build the grid from eligible sampled records in the requested birth-band range. Reindex the pivot to the configured dose categories explicitly. Reconcile opening counts and deaths against the sampled records before constructing totals. Preserve records of failed draws or fail the run when its requested sample is incomplete.

**Scope clarification.** The Monte Carlo converter also intentionally pools doses at and above a date-specific cap, whereas the ordinary converter retains up to six doses and the ordinary reader filters to the configured comparisons. The top-dose labels therefore need to say, for example, “3 or more” wherever that rule applies. Checking the bundled 100,000-record sample with the documented product and age restrictions gave 13,965 exactly-three-dose versus 13,966 three-or-more-dose members for enrollment 2022-26. That distinction does not require changing the chosen design.

## 3. The overlapping Table 8 input can pass point validation despite impossible membership

In the `table8` layout the 1960 pseudo-group includes the 1940 and 1950 subgroups. The ordinary point calculation checks each weekly risk/death history separately. The bootstrap also subtracts the subgroup death/survivor cells from the inclusive cells and requires nonnegative remainders. Those checks disagree. See `code/fixed_cohort_history.py:143`, `code/bootstrap_reporting.py:42`, and `code/KCOR.py:6504`.

In a 20-week fixture the dose-1 opening risks were 60,000 and 20,000 in the two constituent groups but only 40,000 in their inclusive group. Each group individually satisfied death-only accounting. Point validation returned no invalid components. The workbook reported the inclusive and direct all-ages KCOR as 0.999967 and the overlapping pooled KCOR as 0.999963, each with normalization status “ok.” The bootstrap rejected the same input for invalid overlapping history counts.

**Correction.** Share the overlap-feasibility test before reporting the affected point estimates, including checks on weekly deaths and the final surviving remainder. Keep unrelated comparisons available. I separately checked the actual two `analysis/table8/input.xlsx` sheets. Both passed the disjoint-cell reconstruction, with 2,511,927 and 2,450,449 people respectively. This is a validation defect for other input, rather than evidence against those archived rows.

## Interpretation of the evidence

The Czech illustration already discloses the lack of outcome-independent window selection, the product restriction using some post-enrollment records in S6, the residual flags, and the external status of the forced exponent. The primary Table 8 bootstrap uses its own matched disjoint-cell reconstruction. The Cox illustration explicitly supplies the generating variances to KCOR. Those details limit the conclusions that each exercise supports. They do not require replacing the descriptive estimand or creating another causal model.

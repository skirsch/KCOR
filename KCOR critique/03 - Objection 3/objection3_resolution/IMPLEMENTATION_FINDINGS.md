# Production reproduction findings

These findings concern the current source checkout, identified by the SHA-256 hashes in the verification artifacts. They do not establish which source revision produced historical manuscript tables or archived workbooks.

## 1. Time is counted across bootstrap samples

`code/KCOR_CMR.py:1577` writes all iterations into a single enrollment worksheet. In `code/KCOR.py:6772`, the time index is constructed with:

```python
df["t"] = df.groupby(["YearOfBirth", "Dose"]).cumcount().astype(float)
```

The `mc_id` grouping used elsewhere is missing here. In a complete calendar grid with M samples, sample i receives `t = M * calendar_week + i`. Both the fitting time and the skipped-week rule use that index. The normalization baseline still uses row index 5 of each sample's merged curve, so this is not a harmless rescaling of time.

With at least six samples, iteration zero has no observations in the fitter's anchor range `2 <= t < 6`. Its fit returns `empty_anchor`, and the normalization falls back to theta zero. Only sample identifiers 2–5 can have an anchor point at all in this complete-grid construction. Other samples also fall back. The failure is introduced by the processing layout, even when the input population satisfies the gamma working model.

### Direct reproduction

The identical original simulated dataset was processed by the unmodified full workbook pipeline:

| Layout | Original estimate at week 80 | Original fit |
|---|---:|---|
| Original sample alone | 1.0040494943 | Both arms fitted successfully |
| Original plus nine bootstrap samples in one sheet | 1.0531722633 | Both arms not identified; theta zero applied |

Sources: `development/reproduction_single_iteration/verification.json` and `development/reproduction_10_iterations/verification.json`. Input workbooks, production output workbooks, and logs are retained beside them. This comparison diagnoses the bug; it is not a coverage experiment. The week-52 geometry used in the final design is separately reproduced in `development/reproduction_final_geometry/`.

The array adapter matches every week of every retained age-specific production curve in the 1-, 4-, and 10-sample reproduction cases, with maximum absolute error at most 4.5e-16. It uses the production fitter and inversion. It explicitly reproduces the erroneous joint time index; it does not repair it silently.

The additional full-size check matched 100,000 completed age-specific production points from 1,000 samples with **zero absolute difference**, and checked theta columns and the real production interval summary. Evidence is in `development/reproduction_1000_iterations/verification_age_scope.json`. The age-specific output was captured read-only from production's completed `out_rows` list. Source inspection confirms that the subsequent age-pooling work does not modify those frames: final assembly concatenates them, cleans column names, and selects columns. The ongoing `all_ages_asmr_rows` loop creates rows the source near line 5456 explicitly excludes from output. It was stopped after the requested age-specific calculation was saved and verified; no completed 1,000-sample workbook export is claimed. `run_disposition.json` records this scope and stop reason.

### Required correction and regression checks

Production time must represent elapsed calendar time within each sample, for example by deriving it from dates or by grouping the complete-grid count by `mc_id` as well as age and dose. Verify that the original point estimate is invariant to adding, removing, and reordering bootstrap samples, that every sample has the same intended skipped weeks and anchor weeks, and that joint processing agrees with processing each sample separately. Rerun final calibration after the repair. No production code is changed by this module.

## 2. The compact diagnostic summary misses Monte Carlo keys

Per-iteration rows in `development/reproduction_10_iterations/production.xlsx` correctly show `NOT_IDENTIFIED` for iterations 0, 1, and 6–9. Nevertheless, the companion `KCOR_summary.xlsx`, sheet `theta0_status_summary`, reports zero counts and `any_non_ok_status=False`.

The source explains this discrepancy: the summary loop near `code/KCOR.py:9667` skips parameter-map keys unless their tuple length is three. Monte Carlo fits use four-element keys `(enrollment, mc_id, age, dose)`. Consequently, an empty status tally can appear clean. The detailed per-iteration warnings exist; this is not a claim that the code has no diagnostics.

The validation module gates on the actual fit results, not this compact summary. The summary must retain the iteration dimension or otherwise count the four-element keys explicitly before it can be used to authorize interpretation of Monte Carlo output.

## 3. Resampling and interval construction are real, but differ from the prose

The actual Monte Carlo path resamples complete people with replacement from the pooled eligible population. `verify_person_aggregation.py` executes the original `process_enrollment_data` function, extracted without changing its syntax, to avoid that script's unrelated top-level input/output work. It verifies exact death and survivor counts for original people and two actual pandas resamples. The two resampled arm totals differ, confirming that this is not a fixed-arm-size bootstrap.

Grouped multinomial sampling of identical lifetime records has the same distribution as that person bootstrap. It preserves event/risk-set dependence and calendar order. It is not the independent weekly Poisson bootstrap used by the separate historical simulation-grid validation.

`create_mc_summary` excludes iteration zero from bootstrap quantiles and prints percentile and basic intervals. The module checks its interval construction against production output. The ordinary workbook's `CI_lower` and `CI_upper` columns are a separate analytic interval; they are not substituted for Monte Carlo quantiles in this study.

The current manuscript source, `documentation/preprint/paper.md:631–662`, instead describes contiguous block resampling of aggregate time bins, and `documentation/preprint/supplement.md:270` explicitly says individuals are not resampled. It also acknowledges that definitive coverage validation and formal theory are deferred and limits interpretation to accepted analyses. A fair objection is the mismatch and unfinished calibration, not the absence of uncertainty code or the failure to promise universal robustness.

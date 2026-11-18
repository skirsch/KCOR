# Objection 3: uncertainty validation and current production behavior

**Status: completed November 10, 2026. The fixed validation, production comparisons for the declared age-specific scope, interval reconstruction, and completion audit have passed.**

**Verdict: retain objection 3 as a specific implementation, documentation, and calibration objection. Do not retain the claim that KCOR has no uncertainty procedure, or reinterpret these results as an impossibility result for the intended method.**

## Framing supported by the completed reproduction checks

KCOR has a genuine person-level Monte Carlo uncertainty procedure. The objection must not be stated as “there are no uncertainty intervals.” The paper also explicitly acknowledges that definitive calibration is deferred and that interpretation is conditional on diagnostic acceptance; it does not promise valid intervals under every possible violation of its working model.

The present review nevertheless identifies a concrete implementation problem. In the current production path, processing bootstrap samples together changes the time index used to fit the correction and apply the skipped-week rule. The original dataset's estimate changes merely because bootstrap samples are added to its worksheet. A second defect makes the compact diagnostic summary omit Monte Carlo fit-status entries. These findings are independent of contamination scenarios.

This module evaluates the existing calculation as written. It does not silently substitute a repaired estimator. Production code, data, manuscript files, and previous experiments are unchanged. A completed module can conclude that the objection remains and specify the required correction; it need not force favorable results.

## What was tested

The fixed design uses eight scenarios, 600 independently generated datasets per scenario, and 999 person-bootstrap samples per dataset. It evaluates a single age-specific comparison, a known quiet window, the one-year reporting point, and no optional wave adjustment. Seven scenarios satisfy the gamma survival model, including null, boundary, low-information, and post-quiet-window changes. The eighth changes the baseline inside the quiet window and is a separately labelled assumption-violation stress test.

Each person contributes an arm label and one death week or survival beyond follow-up. A multinomial representation compresses identical records without changing their sampling distribution. Resampling is pooled across arms, so arm sizes vary as in the production person bootstrap. It is not independent resampling of weekly death counts. The original CMR aggregation function has been executed on explicit people and actual pandas resamples to verify the count reconstruction.

The reference is the model-defined, post-skip depletion-neutralized cumulative-hazard ratio, normalized at the same baseline week as the intended calculation. It is checked using exact gamma inversion. It is a descriptive working-model contrast, not an estimate of a uniquely identified causal treatment effect. The population output of the implemented algorithm is recorded separately and is not substituted for this reference to improve apparent coverage.

## Why the production comparison matters

The full production workbook path gives approximately 1.00405 for a retained original dataset processed alone and 1.05317 for the identical dataset processed alongside nine bootstrap samples. Both results are at week 80 in a development reproduction fixture. The final study's week-52 configuration has its own production reproduction fixture.

At the actual final batch size, all 1,000 completed production age-specific trajectories—100,000 points—matched the adapter exactly, as did the fitted theta columns and the production interval summary. These values were saved directly from the unmodified production function after its age-specific calculations completed. The remaining loop computed aggregate rows that the source explicitly does not export, so it was ended after verification. The 1,000-sample workbook export was not completed; age pooling was already outside the prespecified study scope. Complete workbook exports were verified for the smaller 1-, 4-, and 10-sample fixtures. See [the full-size comparison](development/reproduction_1000_iterations/verification_age_scope.json) and [its transparent run disposition](development/reproduction_1000_iterations/run_disposition.json).

The source constructs time with a cumulative row count grouped by age and dose, without the sample identifier. For M complete samples, sample i receives time `M * week + i`. With at least six samples, iteration zero has no observations in the fitter's early anchor range. The failed fit then applies theta zero. This is caused by the worksheet layout even for model-consistent populations.

For the unequal-frailty null at the one-year point, the mathematical reference is exactly 1. The deterministic population output is about 1.08029 under the current combined-sample calculation, compared with 1.00461 for the single-sample calculation. The latter also has a small implementation/model approximation difference; it must not be assumed unbiased or calibrated merely because the larger bug is repaired.

Detailed production rows identify failed fits. The compact summary nevertheless reports no non-OK fits because its tally skips four-element Monte Carlo parameter keys. Consequently, that compact summary is not a reliable acceptance gate. The module uses actual fit results instead.

See [IMPLEMENTATION_FINDINGS.md](IMPLEMENTATION_FINDINGS.md) for source locations, saved workbooks, and the exact defects.

## Final numerical results

All 4,800 final datasets completed, with 4,795,200 bootstrap estimates retained. The audit reconstructed every interval, checked that all dataset hashes differ, and reproduced 16 selected datasets and their first eight bootstrap samples from saved seeds. The final source/configuration hashes matched at the end of execution. Development and final random-number namespaces were separate.

The table reports **unconditional coverage of the intervals that the current code prints**, not coverage among accepted analyses. Every scenario has 600 independently generated datasets. Population sizes are totals across both arms.

| Scenario | Population | Model reference | Percentile coverage (95% Wilson interval) | Basic-interval coverage | Mean interval width | Mean estimation error |
|---|---:|---:|---:|---:|---:|---:|
| Equal frailty, no difference | 100,000 | 1.00000 | 93.8% (91.6–95.5%) | 84.5% | 0.3067 | +0.0076 |
| Unequal frailty, no difference | 100,000 | 1.00000 | 83.5% (80.3–86.3%) | 81.5% | 0.3298 | +0.0823 |
| Unequal frailty, larger population | 1,000,000 | 1.00000 | 13.3% (10.8–16.3%) | 21.0% | 0.1030 | +0.0804 |
| Zero frailty, no difference | 100,000 | 1.00000 | 95.3% (93.3–96.8%) | 85.5% | 0.3047 | +0.0008 |
| Increase after quiet window | 100,000 | 1.04865 | 83.5% (80.3–86.3%) | 78.8% | 0.3465 | +0.0890 |
| Decrease after quiet window | 100,000 | 0.95135 | 82.5% (79.3–85.3%) | 80.7% | 0.3133 | +0.0804 |
| Low-information population | 4,000 | 1.00000 | 95.2% (93.1–96.6%) | 87.3% | 2.4127 | +0.1899 |
| Quiet-window contamination **(stress only)** | 100,000 | 1.15179 | 84.0% (80.9–86.7%) | 80.2% | 0.3792 | +0.0957 |

The unequal-frailty, larger-population, and post-quiet-change cases establish undercoverage according to the prespecified criterion: their Wilson intervals lie wholly below 92%. These are model-consistent cases. The stress test is not needed for that finding. Equal-frailty percentile coverage is inconclusive under the stricter prespecified rule because its Wilson interval extends below 92%. Zero-frailty and low-information percentile coverage meet the numerical unconditional criterion in those particular cases, but cannot validate the correction or accepted-analysis procedure. Basic intervals under-cover in every scenario under the same criterion.

**All 4,800 original analyses failed their correction fit, and zero passed the declared diagnostic gate. Conditional coverage is therefore undefined in every scenario.** Nevertheless, all printed intervals were finite and ordered: interval-production failure was 0%. Bootstrap fit failures were approximately 99.60% in all but the low-information case, where they were 99.79%. Finite output is plainly different from a successfully fitted analysis.

The larger-population result is consistent with systematic bias: its mean error remains about +0.0804 while its intervals become narrower. The deterministic population calculation gives the same direction and approximate size of error. This explains why simply using more people or more bootstrap samples does not repair the missing frailty adjustment.

A separately labelled **post-run target sensitivity check** used the population outputs recorded during development. In the large-population case, percentile coverage was 17.2% if the reference was the single-sample procedure's population output (1.00461), and 93.2% if the reference was the combined-sample algorithm's own population output (1.08029). Thus, the main failure is not an artifact of choosing exactly 1 instead of the single-sample descriptive functional. The intervals can largely follow the output of the wrong combined-sample calculation. Neither alternative replaces the prespecified model reference or establishes conditional calibration. [Target sensitivity results](final_2026-11-10/assessment/target_sensitivity.json) preserve all scenarios and both interval types; no estimator or acceptance rule was tuned for this check.

The numerical conclusion is stable to the declared bootstrap precision check. Using only the first 499 draws changed percentile coverage by at most 1.17 percentage points and basic-interval coverage by at most 0.83 percentage points, both below the prespecified 2-point limit. This checks quantile precision at the same batch geometry; it is not a run of production with a different total number of samples.

Full interval widths, signed errors, root mean squared errors, failure rates, acceptance denominators, Wilson bounds, and precision checks are retained in [summary.json](final_2026-11-10/assessment/summary.json). See [the results table](final_2026-11-10/assessment/RESULTS.md), [per-dataset results](final_2026-11-10/assessment/replicates.csv), and [the numerical audit](final_2026-11-10/assessment/audit.json).

## Interpretation rules

The report must distinguish unconditional coverage of intervals the code prints from coverage conditional on the declared diagnostic gate. If no original analyses pass, conditional coverage is undefined. That does not refute the paper's claim restricted to accepted analyses; it means this implementation cannot provide the needed validation of that claim. Treating undefined conditional coverage as zero, 100%, or evidence of adequacy would be wrong.

The module's gate is explicit: both fits must succeed, with no degeneracy guard and none of the listed production warning flags. It is not a claim to encode every graphical or stability judgment mentioned by the manuscript. The full manuscript diagnostic protocol is not specified as a single executable acceptance rule.

Any failure in the quiet-contamination stress test is evidence about an assumption violation. It is not needed to establish the implementation defects and cannot by itself refute the method within its stated scope.

## Changes needed before lifting the objection

1. Repair sample-specific time construction. The original point estimate must be invariant to adding or reordering bootstrap samples, and joint processing must agree with processing samples separately.
2. Make the compact diagnostic summary count Monte Carlo entries and preserve their iteration-specific failure information.
3. Rerun calibration of the repaired production procedure with fresh validation seeds and a declared acceptance gate. Include both coverage and how often the gate permits interpretation. Do not reuse the current run as validation of code it did not execute.
4. Reconcile the manuscript's description of block resampling of aggregate time bins with the actual person-resampling path. Identify which interval calculation and source revision produced each reported table or figure.

The production repairs above are recommendations, not changes made by this module. The ordinary workbook's analytic intervals, the separate historical simulation-grid bootstrap, and historical manuscript tables are distinct outputs. Findings about the current person-bootstrap path must not automatically be attributed to those other outputs.

## Suggested wording for other reviewers

> KCOR includes Monte Carlo uncertainty calculations, and the manuscript acknowledges that definitive calibration is deferred and interpretation is conditional on diagnostic acceptance. The current production person-bootstrap implementation nevertheless contains a reproducible time-index defect that changes the original estimate when resamples are added and causes its correction fit to fail. Its compact diagnostic summary also omits Monte Carlo fit entries. A fresh, prespecified validation found substantial unconditional undercoverage in model-consistent cases, while no original analyses passed the declared fit gate; conditional coverage could not be assessed. Repair the implementation and diagnostic summary, align the manuscript's resampling description with the validated code, and validate the repaired procedure on fresh simulations before treating the reported intervals as adequately calibrated. This is not a demand for universal robustness or a proof that the intended method cannot support valid inference.

## Limits on the eventual verdict

This study can determine whether the tested current production path supplies adequate uncertainty evidence and can isolate reproducible reasons it does not. It cannot certify every age-pooling option, optional wave adjustment, historical code revision, or future repair. In particular, it cannot establish that the intended KCOR method is mathematically incapable of valid uncertainty intervals. The appropriate objection is about implementation, documentation, and demonstrated calibration within the claimed scope—not a renewed demand for universal identification or robustness.

Reproduction commands and file descriptions are in [README.md](README.md); fixed rules are in [DESIGN.md](DESIGN.md).

The final [completion evidence](completion_evidence_final.json) checks all 12 frozen execution/configuration hashes, six exact source/manuscript snapshots, the production comparisons, numerical audit scope, per-scenario coverage counts, conditional denominators, and target sensitivity. [COMPLETION_AUDIT.md](COMPLETION_AUDIT.md) maps this evidence to every goal requirement. Production repairs remain future work; no original source, data, manuscript, or earlier experiment was changed by this module.

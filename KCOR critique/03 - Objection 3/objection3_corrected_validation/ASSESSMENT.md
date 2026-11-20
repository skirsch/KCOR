# Objection 3 after the Monte Carlo repair

For the post-run investigation of the large-population result, see [DIAGNOSIS.md](DIAGNOSIS.md).

## Verdict

The corrected experiment is complete. The production clock and diagnostic defects explained much of the poor coverage in the earlier run. They do not explain all of it. In the prespecified large-population, unequal-frailty scenario, the production percentile interval contained the model-defined value in only 502 of 600 independent datasets (83.7%; 95% Wilson interval 80.5%–86.4%). All 600 original analyses passed the declared fit gate, every reported interval was finite and ordered, and no bootstrap fit failed. This is undercoverage under the prespecified rule. Objection 3 therefore remains as a **specific percentile-interval calibration issue in a model-consistent, accepted scenario**. It is no longer accurate to summarize the whole objection as a failure caused solely by the sample-clock bug.

This result does not show that KCOR lacks uncertainty machinery, that every scenario undercovers, that basic intervals under-cover in this scenario, or that historical manuscript intervals came from this source revision. The separate discrepancy between the manuscript's aggregate-block resampling description and the code's person resampling is not resolved by this repair.

## What ran

The run used the corrected source snapshot from [PR #4](https://github.com/skirsch/KCOR/pull/4). The original experiment's eight scenarios, model reference, 600 independent datasets per scenario, 999 person-bootstrap draws per dataset, percentile primary interval, basic secondary interval, diagnostic gate, and numerical decision rules were retained. Fresh final seeds were used after the software repair. The age-specific path was tested with optional wave correction disabled. The eight scenarios include one quiet-window contamination stress, which is not used to establish the model-consistent calibration finding.

All 4,800 datasets and 4,795,200 bootstrap estimates were retained. The summary reconstructed every interval and replayed 16 datasets and their first eight bootstrap draws from seeds. Before restart, the coordinator checked the 1,062 preserved JSON/draw file pairs and frozen source hashes. The completion audit then checked all 4,800 results, scenario definitions, the ten-sample production workbook comparison (maximum curve difference 4.5e-16), and conditional denominators. The three execution stages—run, summarize, audit—exited successfully. Evidence is in [`final_execution_resumed.json`](final_execution_resumed.json), [`completion_audit.json`](final_2026-11-10/assessment/completion_audit.json), and the detailed [`comparison.json`](final_2026-11-10/assessment/comparison.json). No production source, manuscript file, or earlier experiment was edited.

## Main comparison

These are percentile-interval coverage rates against the fixed model-defined target. Old and corrected studies use the same scenarios but different random seeds, so this is a scenario-level comparison, not a paired dataset analysis.

| Scenario | Earlier code | Corrected code | Accepted originals | Corrected assessment |
|---|---:|---:|---:|---|
| Equal frailty, no difference | 93.8% | 97.7% | 600/600 | Inconclusive or conservative |
| Unequal frailty, 100,000 people | 83.5% | 96.2% | 600/600 | Adequate in tested scenario |
| Unequal frailty, 1,000,000 people | 13.3% | 83.7% | 600/600 | **Undercoverage** |
| Zero frailty, no difference | 95.3% | 98.0% | 600/600 | Inconclusive or conservative |
| Increase after quiet window | 83.5% | 94.7% | 600/600 | Adequate in tested scenario |
| Decrease after quiet window | 82.5% | 94.5% | 600/600 | Adequate in tested scenario |
| Low-information population | 95.2% | 97.3% | 0/600 | Conditional coverage undefined |
| Quiet-window contamination **stress** | 84.0% | 93.5% | 600/600 | Assumption-violation stress; inconclusive |

The numerical rule classified a scenario as adequate only when the two-sided 95% Wilson interval for coverage lay wholly inside 92%–98%, interval failure was at most 1%, and the 499-draw precision comparison changed coverage by no more than two percentage points. For the 1,000,000-person scenario the coverage interval lies wholly below 92%, so undercoverage is clear. The other labels above follow this same rule; a point estimate of 98% alone does not establish adequacy. All corrected percentile intervals were finite and ordered. The largest absolute 499-draw percentile coverage change was one percentage point.

For the low-information scenario, the declared gate rejected all 600 originals, so conditional coverage is **undefined** despite 97.3% unconditional coverage of printed intervals. For the other seven scenarios, every original passed the operational gate, so unconditional and conditional coverage coincide. The gate checks fit success, the degeneracy guard, and specified production warning flags; it does not encode every graphical judgment mentioned in the manuscript.

## Why the large-population result matters

The large-population scenario satisfies the gamma survival model used to generate the data and has no quiet-window contamination. The true normalized contrast is 1. The corrected estimate's mean error was about +0.00986, while its mean percentile interval width was 0.0496. The deterministic fitted population output of this implementation is 1.00461 rather than the model-defined value 1; that residual approximation difference may contribute to undercoverage as intervals narrow. A secondary check against 1.00461 found 94.0% percentile coverage. That check helps locate the discrepancy but does **not** replace the prespecified model target or prove that approximation error is the sole cause.

The corrected basic interval covered the model-defined value in 97.7% of datasets in this scenario. Its 95% Wilson bounds extend above the prespecified 98% upper limit, so the stringent adequacy rule calls it inconclusive or conservative. It is inaccurate to transfer the percentile undercoverage finding to the basic interval.

## Scope of the remaining objection

The repair resolves the two demonstrated software defects in the tested current production path: mixing sample clocks and omitting sample-specific failures from compact diagnostics. It also makes meaningful conditional calibration possible in the well-informed scenarios. The full corrected run supports a narrower objection: the **primary production percentile Monte Carlo interval is not calibrated to the model-defined contrast in the large, unequal-frailty, accepted-analysis scenario**, despite adequate or conservative behavior in several other tested scenarios.

This does not establish a universal failure of KCOR or prove which uncertainty calculation generated past paper tables. Reconciling the paper's resampling description with the implemented procedure and identifying the provenance of historical reported intervals remain separate review questions. The detailed results, including basic intervals, confidence bounds, fit failures, widths, precision checks, and secondary target sensitivity, are in the assessment files linked above.

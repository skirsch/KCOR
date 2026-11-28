# Corrected Monte Carlo calibration: design version 2

This extension evaluates PR #4, commit ab5baf0, using a byte-for-byte snapshot of its corrected production source. It preserves the earlier experiment and does not modify the production checkout, PR, manuscript, or earlier results.

The eight scenarios, 600 datasets per scenario, 999 bootstrap resamples per dataset, target at week 52, two skipped weeks, normalization at index 5, quiet weeks 6 through 40, model reference, diagnostic gate, coverage criteria, and prefix precision check are unchanged from `../objection3_resolution/DESIGN.md` and its version 1 `design.json`. This incorporates those statistical definitions; statements there about reproducing the buggy clock do not apply here.

## Differences fixed before outcomes

The source now constructs each complete-grid sample's time independently using its `mc_id`. The array adapter imports the snapshotted production fitter, guard, and inversion and sets t to calendar week, regardless of sample identifier or total samples. The corrected source also retains sample-specific fit diagnostics. The gate still uses the actual fit results, so it does not depend on summary formatting.

Fresh development and final seed namespaces are 526103 and 942107. This follows the earlier design's rule to use fresh validation seeds after a repair. Comparisons with the earlier experiment are scenario-level comparisons, not paired changes in identical datasets. Neither the scenarios nor the gate will be tuned in response to final outcomes.

The analytic reference is the model-defined post-skip cumulative baseline hazard ratio normalized at index 5. The deterministic fitted population output is reported separately and must not replace that reference to improve coverage.

## Verification and execution

Before final execution, check the analytic reference and independent sampling, reproduce complete curves and production interval summaries using the corrected full workbook path with multiple samples, and run a separate development pilot. The PR's original-versus-fixed regression and saved ten-sample replay are retained in `../objection3_production_fix/pr_verification/`.

The original person-bootstrap aggregation verification remains applicable because the resampling implementation is unchanged; hashes of the source snapshots establish this. Source hashes, exact configuration, package versions, seeds, complete draws, and per-dataset diagnostics will be retained. Final jobs have fixed counts and may be resumed only if their source/configuration hashes match. Existing evidence must not be overwritten.

The primary intervals are percentile intervals; basic intervals are secondary. Report both unconditional coverage and coverage conditional on successful original-sample fits, no degeneracy guard, and none of the existing adapter warning flags. This operational gate does not purport to encode every manuscript graphical or stability check. Conditional results with fewer than 300 accepted analyses are inconclusive. Failed finite fits are retained because the production method reports them.

The numerical criterion remains a 95% Wilson coverage interval wholly inside [0.92, 0.98], interval failure no more than 1%, and a coverage change no greater than 0.02 when using the first 499 bootstrap draws. A Wilson interval wholly below 0.92 establishes undercoverage for that scenario. Other cases remain inconclusive or conservative. Conditional calibration must use its own denominator and uncertainty interval, and cannot be inferred from unconditional calibration.

Final verification reconstructs all intervals, checks dataset uniqueness, reproduces selected seeds, checks source hashes, and compares old and corrected scenario summaries. Report failures, gate acceptance, width, error, root mean squared error, and prefix sensitivity. Label quiet-window contamination separately as an assumption-violation stress test. Boundary and low-information cases must also remain identifiable.

## Scope of the conclusion

This tests the age-specific person-bootstrap path with optional wave correction disabled. It does not certify age pooling, optional wave correction, historical paper outputs, or the separate simulation-grid bootstrap. Passing regression checks establishes the software fix; only completed calibration can address statistical coverage. Documentation differences between the manuscript and the implemented resampling remain a separate question, not silently resolved by this experiment.

# Keep Monte Carlo sample clocks and diagnostics independent

When multiple Monte Carlo samples were processed together, the elapsed-time counter grouped observations by birth year and dose without the sample identifier. Adding bootstrap samples could therefore change the original sample's fit and estimate. Include `mc_id` in the Monte Carlo time grouping while preserving ordinary processing.

Diagnostic reporting also expected ordinary fit-record keys, omitting or failing to retrieve sample-specific records. Recognize Monte Carlo keys, retain sample identifiers in diagnostic output, and retain failed fits with missing parameters. This makes failure counts and row-level fit details traceable to the correct sample.

Validation: the original source fails two of three regression tests; the patched source passes all three. Tests cover sample-count independence, shuffled and relabeled samples, sample-specific failure reporting, and ordinary processing on the fixture. Replaying a retained ten-sample workbook restores all 100 original-sample curve points exactly to the original-alone result. At week 80, the combined result changes from 1.053172263346009 to 1.004049494273245, matching the original-alone estimate. Patch applicability was checked without modifying the original source.

This submission fixes software behavior. Confidence-interval coverage after correction requires a fresh validation run.

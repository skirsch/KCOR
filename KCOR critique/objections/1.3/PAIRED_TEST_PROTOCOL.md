# Paired uncertainty checks for objection 3

For each new objection 3 simulation scenario, evaluate both Monte Carlo resampling versions on the **same independently generated outer datasets**:

1. **Person resampling:** the method actually used by the current real-data Monte Carlo code. Draw people with replacement, retaining each sampled person's cohort and death time, rebuild weekly death counts and risk sets, refit, and take percentile endpoints.
2. **Aggregate-block resampling:** the manuscript's stated approach. Use an explicitly recorded block length, source-time pool, placement rule, risk-set seam rule, and percentile endpoints. Until the authors' generating code or exact rules are available, label results by the chosen operationalization rather than calling them the unique paper interval.

Keep the **point estimator identical** between these two interval arms so their difference isolates resampling. For paper-first checks, use the manuscript's joint baseline-rate/frailty fit. If testing the current production fixed-rate fitter, report that as a separate estimator comparison. Use the same true target, target week, anchor, outer dataset seeds, number of bootstrap draws, and diagnostic reporting. Retain every dataset's point, both intervals, both coverage indicators, numerical fit outcomes, and source hashes.

The [first paired study](objection3_paper_method/paired_resampling/ASSESSMENT.md) compares 200 one-million-person, model-consistent datasets at week 52. Its block counterpart is the [explicit four-week study](objection3_paper_method/block_bootstrap_operationalization/ASSESSMENT.md). The protocol is a comparison rule for future work; it does not fill the manuscript's missing block-bootstrap or diagnostic specification.

# Operational details missing from the stated interval procedure

The paper's section 2.9.1 calls for resampling contiguous blocks of event-time bins within cohort and stratum, carrying deaths and risk sets forward, refitting the estimator, and taking 2.5th and 97.5th percentiles. The supplement says individuals are not resampled. Searching the repository's production, test, scripts, manuscript, and supplement paths found the description but no executable implementation of this specific aggregate-block procedure. The real-data Monte Carlo path resamples people; the simulation-grid coverage script draws approximate weekly Poisson deaths. [Earlier provenance audit](../objection3_step2_interval_provenance_2026-11-10/assessment.txt).

The description leaves choices that change the resulting distribution:

1. What block length is used, and is it fixed or estimated separately by cohort or stratum?
2. Are sampled blocks kept at their original calendar times or inserted at new times? If inserted, how are the aging trend and prespecified quiet-window dates treated?
3. How are numbers at risk connected at seams between sampled blocks when a block's original starting risk set differs from the preceding block's reconstructed survivors?
4. Are cohort sizes fixed within each replicate, and how is administrative end-of-follow-up survival handled?
5. Which exact diagnostic rejection rule applies to bootstrap replicates and to the original dataset?

The recurrence `N(t)=N(0)-sum_{s<t}d(s)` constrains the output but does not decide these choices. Different choices can all satisfy the recurrence and yield different interval coverage. An invented implementation would answer a question about that choice, not provide measured coverage for the paper's uniquely specified method.

An [explicit operationalization](block_bootstrap_operationalization/ASSESSMENT.md) now provides such a conditional test. Four-week block resampling with broad pre-quiet, quiet, and post-quiet source pools covered 38/200 model-consistent datasets. A source-time placement check on exact expected counts showed a large change in the bootstrap center when those pools were narrowed. This strengthens the need to state the missing rule; it does not identify which rule produced historical manuscript intervals.

To complete a manuscript interval test, obtain a generating script or fully specified rules and the revision behind the reported output. Then freeze those rules, run the paper's joint fit and full diagnostics in every bootstrap sample, and assess coverage on independent model-consistent datasets. The existing 83.7% production-code result and 93.3% candidate pilot cannot substitute for this test. A new [paired test](paired_resampling/ASSESSMENT.md) found 94.5% coverage for the paper joint fit with **person** resampling on the same datasets where this explicit block rule covered 19%; it shows that the resampling choice matters but still does not define the manuscript's intended block rule.

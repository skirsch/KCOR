# 6.5 ? The 5,000-per-arm precision claim relies on deterministic simulations

**Severity:** High for precision claim

The saved cohort-size sensitivity has identical median 1.007110465349759, zero interval width and zero false-signal rate at all six sizes and 500 replicates. The generator uses expected deaths without sampling. The claim about small-sample uncertainty is unsupported by these simulations. Check intended source, then narrow or replace the claim with actual sampled evidence.

**Related issues:** 1.3, 2.7.

**Origin:** New validation defect.

**Review evidence:** [B](../../iterative%20review/evaluations/6/blind-b.md).

**Status:** addressed in m122, preserving m121. Section 5.3 replaces the unsupported approximately 5,000-per-arm claim with a reference to the existing Section 2.9 information analysis and event-count, reference-period, window-placement and frailty-resolution requirements. The original deterministic cohort-size simulation is retained as historical evidence. No new simulation or estimator change was needed.

The six saved rows in `test/sim_grid/out/cohort_size_sensitivity.csv` all have the same median 1.007110465349759 and zero percentile width, as reported by the review. The recent [S6 precision study](../6.4/all-age-precision.md) provides actual matched sampled-data evidence. [Paper patch](paper.patch).

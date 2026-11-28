# 8.10 - Reproducibility scope and Table 8 cached replay

**Severity:** Moderate for reproducibility

**Origin:** Returning 7.8 plus new command defect

**Related issues / root causes:** 7.8, 7.3, 5.12: artifact verification versus fresh reruns

**Finding:** SI S7 now accurately states archive scope but main section 2.13 still promises all studies reproducible. Table 8 runner loads checkpoints with all 2,000 flags true in each mode; documented refit schedules zero fits. Current stored estimates/percentiles checked out.

**Proposed scope (for discussion):** Qualify universal claim; make fresh refit versus archive/resume explicit and verify cache provenance (input, settings, estimator). No evidence current intervals are numerically wrong.

**Review evidence:** Resolution. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m136 (approved straightforward-correction batch; m135 preserved). Independent closure not yet evaluated.

Qualified section 2.13 reproduction scope. Default Table 8 refit now starts fresh in a separate directory and preserves archive. Explicit resume verifies matching input/config/source/settings provenance. A one-draw smoke test scheduled four fits on fresh run, zero on matching explicit resume, and rejected incompatible draw count. Existing 999-draw archives unmodified. [Runner patch](table8_replay.patch).

[Batch manuscript and code patches](../../iterative%20review/evaluations/8/post-review-batch-1/).

**Round 9 independent disposition:** partial; see [9.3](../9.3/README.md). [Resolution](../../iterative%20review/evaluations/9/resolution.md).

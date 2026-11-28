# 8.11 - Early anchor assumption contradicted elsewhere

**Severity:** Low to moderate, wording

**Origin:** Returning 7.9

**Related issues / root causes:** 7.9, 3.2, 4.5: early anchor model assumption

**Finding:** Section 2.5 now explicitly describes the early baseline assumption; section 2.4.4 still claims only designated quiet bins are fitted.

**Proposed scope (for discussion):** Make section 2.4.4 consistent with the approved definition including early post-skip fitting anchors and marked-wave exception. No fitter change.

**Review evidence:** Resolution. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m136 (approved straightforward-correction batch; m135 preserved). Independent closure not yet evaluated.

Corrected section 2.4.4 to name early post-skip anchors and marked-wave exclusion, consistent with section 2.5.

[Batch manuscript and code patches](../../iterative%20review/evaluations/8/post-review-batch-1/).

**Round 9 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/9/resolution.md).

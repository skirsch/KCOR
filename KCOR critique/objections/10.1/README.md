# 10.1 - Verified exhaustion suffix rejected by overlapping bootstrap

**Severity:** Moderate to high

**Origin:** Direct return; direct predecessor 9.1

**Related issues / root causes:** 9.6, 8.1.

**Finding:** An entire age/dose genuinely exhausts; explicit zero suffix gives unaffected 1950 point 1 and three valid draws [0.850521,1.083847], omitted zeros give same point but no valid draws. Both dedicated reader and bootstrap reject the omitted form, ordinary validator accepts.

**Proposed scope (for discussion):** After successful validation, fill only justified zero suffixes in overlapping rectangular histories; preserve positive-risk gap rejection.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/10/blind-a.md) | [Blind B](../../iterative%20review/evaluations/10/blind-b.md) | [Resolution](../../iterative%20review/evaluations/10/resolution.md)

**Status:** implemented in m153; m152 preserved. Independent closure not yet evaluated.

Both overlapping readers share a rectangular-history helper that inserts only validated zero suffixes after genuine exhaustion; explicit and omitted representations produce identical cells, matched draws, and ordinary unaffected interval 1 [0.850521,1.083847] in a three-draw smoke test. Interior and positive-risk gaps remain rejected. [Batch patches and regression](../../iterative%20review/evaluations/10/post-review-batch-1/).


**Round 11 independent disposition:** resolved in original scope on m156. [Targeted review](../../iterative%20review/evaluations/11/resolution.md).

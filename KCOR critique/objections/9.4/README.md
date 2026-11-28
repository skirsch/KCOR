# 9.4 - Near-zero fitted variance still described as weak identification

**Severity:** Low to moderate

**Origin:** Direct return; direct predecessor 8.13

**Related issues / root causes:** 1.6, 2.2.

**Finding:** SI S6.1.2 note is corrected but diagnostic bullet still equates a near-zero estimate with low curvature or weak information. H double-prime(0)=k(gamma-k theta).

**Proposed scope (for discussion):** Separate small fitted depletion from statistical information; small theta alone is not weak identification.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m143; m142 preserved. Independent closure not yet evaluated.

SI S6.1.2 distinguishes the nonnegative parameter boundary and little fitted frailty-induced depletion from strength of identification and precision. [Batch patches and verification](../../iterative%20review/evaluations/9/post-review-batch-1/).

**Round 10 independent disposition:** partial; see [10.2](../10.2/README.md). [Resolution](../../iterative%20review/evaluations/10/resolution.md).

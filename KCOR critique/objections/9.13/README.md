# 9.13 - Age-related depletion explanation confuses absolute and relative variance

**Severity:** Moderate

**Origin:** New wording defect

**Related issues / root causes:** 8.6, 8.13.

**Finding:** SI S6.1.2 attributes smaller theta at older ages to ongoing depletion; a proportional gamma tilt changes rate, not shape or normalized variance.

**Proposed scope (for discussion):** Report observed age pattern without unsupported mechanism; distinguish absolute and relative variance.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m143; m142 preserved. Independent closure not yet evaluated.

SI S6.1.2 distinguishes absolute survivor variance from mean-one relative variance, which is invariant under proportional gamma tilting. [Batch patches and verification](../../iterative%20review/evaluations/9/post-review-batch-1/).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).

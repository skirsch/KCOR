# 13.3 - Pooled geometric mean drops genuine zero ratios before reference

**Severity:** Moderate for early follow-up

**Origin:** New related boundary discrepancy; predates 12.8

**Related issues / root causes:** 12.8, 11.3, 9.8.

**Finding:** A positively weighted available zero ratio is removed from log pool, resulting in 0.997037 rather than zero in two-age fixture; one-stratum pool unavailable. Positive later anchors valid.

**Proposed scope (for discussion):** Implement nonnegative geometric-mean limit (positive-weight genuine zero yields zero), retain contributors and composition checks; preserve undefined zero reference handling.

**Review evidence:** A, R. [Blind A](../../iterative%20review/evaluations/13/blind-a.md) | [Blind B](../../iterative%20review/evaluations/13/blind-b.md) | [Resolution](../../iterative%20review/evaluations/13/resolution.md)

**Status:** implemented in m168; m167 preserved. Independent closure not yet evaluated.


True finite zero included in positively weighted geometric mean, retaining all contributors; two-age test gives zero instead of dropping the 1940 group. [Batch patches and verification](../../iterative%20review/evaluations/13/post-review-batch-1/).


**Round 14 independent disposition:** resolved in original scope on m169. [Targeted review](../../iterative%20review/evaluations/14/resolution.md).

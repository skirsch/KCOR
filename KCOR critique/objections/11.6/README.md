# 11.6 - Dedicated replay lacks sparse-stratum pooling fallback

**Severity:** Moderate for affected replay histories

**Origin:** Related implementation discrepancy beyond 10.8

**Related issues / root causes:** 10.8, 2.9, 9.8.

**Finding:** Zero early deaths in one stratum gives NaN pooled ratios despite successful fits; ordinary workbook uses minimum-positive-rate fallback and finite weights. Published original weights confirmed.

**Proposed scope (for discussion):** Share the existing workbook fallback and compare actual pooled rows.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Shared existing expected-deaths weighting fallback. Sparse test yields 0.938236, 0.912340, 0.972400, matching ordinary reference. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

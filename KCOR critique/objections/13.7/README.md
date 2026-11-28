# 13.7 - Table 9 labels anchored values as unanchored KCOR

**Severity:** Moderate reporting

**Origin:** New specific notation discrepancy

**Related issues / root causes:** 2.8, 9.8, 13.4.

**Finding:** Archived primary week-80 ratios are anchored to end of fourth accumulating week; table says KCOR(t). Benefit unanchored .831143 vs anchored .833593; harm unanchored 1.157454 vs anchored 1.161701. Intervals are for anchored values.

**Proposed scope (for discussion):** Label Table 9 and Section 3.2 explicitly KCOR(80;t0) and specify reference week; no numerical change.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/13/blind-a.md) | [Blind B](../../iterative%20review/evaluations/13/blind-b.md) | [Resolution](../../iterative%20review/evaluations/13/resolution.md)

**Status:** implemented in m168; m167 preserved. Independent closure not yet evaluated.


Table 9 and Section 3.2 explicitly label anchored week-80 statistic and fourth accumulating-week reference; values and intervals unchanged. [Batch patches and verification](../../iterative%20review/evaluations/13/post-review-batch-1/).


**Round 14 independent disposition:** resolved in original scope on m169. [Targeted review](../../iterative%20review/evaluations/14/resolution.md).

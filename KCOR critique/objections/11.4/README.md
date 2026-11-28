# 11.4 - Ordinary bootstrap loses residual-screen flags

**Severity:** Moderate

**Origin:** New specific reporting defect

**Related issues / root causes:** 9.10, 10.5, 3.1.

**Finding:** A contributing fit with sampling_flag=True and numerical status OK is not counted in CI_point_flagged or flagged draws. Dedicated Table 8 diagnostics retain the distinction.

**Proposed scope (for discussion):** Propagate separate residual-screen flags and counts for actual contributing fits, without excluding draws.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Separate residual flag and unavailable indicator for point/direct and actual pooled contributors, separate draw counts. Tested propagation and written workbook columns; no estimator change. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

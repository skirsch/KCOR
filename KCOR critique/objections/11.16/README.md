# 11.16 - Pooled row copies one stratum?s frailty parameter

**Severity:** Minor workbook labeling

**Origin:** New specific reporting defect

**Related issues / root causes:** 2.9, 10.8.

**Finding:** Weighted geometric mean row carries theta_num/theta_den from first contributor although no pooled frailty is fitted.

**Proposed scope (for discussion):** Leave pooled frailty cells empty or label source; retain directly fitted all-ages frailty.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Geometric pooled rows have blank fitted theta cells; separately fitted all-ages theta retained. Written workbook checked. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

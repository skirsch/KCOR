# 12.9 - Pooled parameter status still copied from one stratum

**Severity:** Minor output labeling

**Origin:** Related pre-existing omission beyond 11.16

**Related issues / root causes:** 11.16, 11.4, 10.8.

**Finding:** Pooled theta values now blank; neighboring status cells still copied from first finite contributor, not representative of all fits.

**Proposed scope (for discussion):** Leave pooled parameter status blank or explicitly summarize actual contributing fit statuses; retain separate flags.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** implemented in m162; m161 preserved. Independent closure not yet evaluated.


Pooled frailty parameter status cells blank; separate contributing-fit flags retained; written workbook parity regression verifies both status cells empty. [Batch patches and PDFs](../../iterative%20review/evaluations/12/post-review-batch-1/).


**Round 13 independent disposition:** resolved in original scope on m167. [Targeted review](../../iterative%20review/evaluations/13/resolution.md).

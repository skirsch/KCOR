# 11.13 - Cumulative person-time column exports weekly exposure

**Severity:** Minor numerical export

**Origin:** New specific reporting defect

**Related issues / root causes:** 9.5, 6.2.

**Finding:** Cumu_Person_Time reports opening risks 20000, 19980, 19959 instead of cumulative 20000, 39980, 59939; cumulative value already calculated.

**Proposed scope (for discussion):** Export cumulative value and verify differencing equals weekly exposure.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Cumulative opening-risk person-weeks computed within dose/age after weekly aggregation; written workbook matches cumulative risk sets. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

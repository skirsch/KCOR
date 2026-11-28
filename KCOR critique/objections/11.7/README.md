# 11.7 - Dedicated Table 8 reader truncates fractional person counts

**Severity:** Moderate for input integrity

**Origin:** Related pre-existing defect

**Related issues / root causes:** 10.1, 9.1.

**Finding:** Internally consistent half-person input is silently cast back to original integers; ordinary bootstrap rejects it. Published input contains integer counts.

**Proposed scope (for discussion):** Validate integer counts before conversion using shared history checks.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Shared reader validates raw integer counts before conversion; fractional fixture rejected. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

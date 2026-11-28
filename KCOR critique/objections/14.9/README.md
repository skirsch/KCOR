# 14.9 - Unknown layout string can bypass ordinary future-year guard

**Severity:** Low to moderate input validation

**Origin:** New related consequence of 13.2

**Related issues / root causes:** 13.2, 12.1.

**Finding:** Invalid layout Disjoint bypasses guard while ordinary aggregation still occurs, bootstrap rejects. Impossible 2099 group changes result from 1 to 1.981297.

**Proposed scope (for discussion):** Validate allowed layout before point or bootstrap processing.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m170; targeted regression or artifact verification completed. Independent round review pending.


**Approved correction and verification:** Both point and bootstrap paths validate the declared layout before processing. Invalid Disjoint spelling raises ValueError; existing round-13 valid-layout and future-birth tests pass.

[Round 14 post-review batch](../../iterative%20review/evaluations/14/post-review-revisions.md).


**Round 15 independent disposition:** Resolved in original scope by the independent round-15 review. [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

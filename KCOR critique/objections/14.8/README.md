# 14.8 - Failed presentation reference falsely flags supported unanchored interval

**Severity:** Moderate reporting regression

**Origin:** New related consequence of 13.6

**Related issues / root causes:** 13.6, 11.4.

**Finding:** Changing reference alone from week59 to exhausted week60 leaves unanchored 1.068548 [0.904564,1.115133] and 3 valid draws identical, but flags change from zero to three and point flagged despite both fits OK.

**Proposed scope (for discussion):** Separate anchored and unanchored support and flags in point and bootstrap; keep anchored failure.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m170; targeted regression or artifact verification completed. Independent round review pending.


**Approved correction and verification:** Independent anchored and raw point/replicate flags now exported. At reference indices 59 and 60, the same earlier unanchored value 1.068548 [0.904564, 1.115133] has 3/3 valid draws, zero flagged draws, unflagged point and identical percentile method. The exhausted anchored reference remains unavailable and flagged. Three draws are a regression smoke test, not inferential precision.

[Round 14 post-review batch](../../iterative%20review/evaluations/14/post-review-revisions.md).


**Round 15 independent disposition:** Resolved in original scope by the independent round-15 review. [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

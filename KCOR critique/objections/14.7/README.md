# 14.7 - Redundant opening-risk PT column disables bootstrap

**Severity:** Low for input delivery

**Origin:** New specific validation issue

**Related issues / root causes:** 9.1, 11.8.

**Finding:** Same validated disjoint histories accepted without PT but rejected when PT exactly duplicates Alive; ordinary method derives PT from Alive.

**Proposed scope (for discussion):** Accept redundant matching opening-risk PT with validated histories; do not accept person-time-only inputs lacking person histories.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m170; targeted regression or artifact verification completed. Independent round review pending.


**Approved correction and verification:** Bootstrap accepts redundant PT exactly equal to opening Alive, drops the redundant field before resampling, and rejects genuine exposure summaries. Both acceptance and rejection tested.

[Round 14 post-review batch](../../iterative%20review/evaluations/14/post-review-revisions.md).


**Round 15 independent disposition:** Resolved in original scope by the independent round-15 review. [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

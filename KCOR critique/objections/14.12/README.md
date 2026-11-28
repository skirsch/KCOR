# 14.12 - Missing S6 gap offsets serialized as nonstandard JSON NaN

**Severity:** Minor archive portability

**Origin:** New regression in 13.5 archive

**Related issues / root causes:** 13.5, 11.18.

**Finding:** 349 per-dose draw cells across 345 rows contain NaN literal not valid strict JSON. Python successfully reproduces all 42 statuses and intervals.

**Proposed scope (for discussion):** Serialize unavailable offsets as null with strict JSON and map to unassessed on reading. No numerical refit.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m170; targeted regression or artifact verification completed. Independent round review pending.


**Approved correction and verification:** Missing S6 offset values serialized as strict JSON null. All 13,986 gap arrays parse strictly; 349 changed cells, other CSV columns identical. Recomputed all 42 gap diagnostic rows including counts, limits and status; 30 no deficit detected and 12 unassessed, identical to archived summaries.

[Round 14 post-review batch](../../iterative%20review/evaluations/14/post-review-revisions.md).


**Round 15 independent disposition:** Resolved in original scope by the independent round-15 review. [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

# 14.6 - Sensitivity sweep rejects small positive references

**Severity:** Low numerical edge case

**Origin:** New related ancillary inconsistency

**Related issues / root causes:** 12.8, 9.8.

**Finding:** Ordinary reference 7.432377e-14 yields finite anchored 3.084780e12; sweep unavailable due EPS threshold. No evidence published Czech sensitivity grid affected.

**Proposed scope (for discussion):** Use checked finite-ratio helper and finite result status in sweep.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m170; targeted regression or artifact verification completed. Independent round review pending.


**Approved correction and verification:** Sensitivity sweep now uses the ordinary finite-ratio helper and unanchored support status. An integrated extreme-count sensitivity fixture with a reference below the old EPS floor returns a finite endpoint marked OK. Zero and nonrepresentable quotients follow ordinary rules.

[Round 14 post-review batch](../../iterative%20review/evaluations/14/post-review-revisions.md).


**Round 15 independent disposition:** Resolved in original scope by the independent round-15 review. [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

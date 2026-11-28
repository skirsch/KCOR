# 13.2 - Impossible future literal birth years enter inclusive population

**Severity:** Moderate for invalid input validation

**Origin:** New related validation issue after 12.1

**Related issues / root causes:** 12.1, 11.1, 9.1.

**Finding:** A 2099 birth group in a 2023 enrollment now enters reconstructed all ages as if valid; changes point from 1 to 1.981297. Valid post-2020 birth parity is fixed.

**Proposed scope (for discussion):** Validate literal birth year against actual enrollment year before aggregation; retain valid later births and explicitly declared synthetic group codes.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/13/blind-a.md) | [Blind B](../../iterative%20review/evaluations/13/blind-b.md) | [Resolution](../../iterative%20review/evaluations/13/resolution.md)

**Status:** implemented in m168; m167 preserved. Independent closure not yet evaluated.


Ordinary literal disjoint birth-year histories reject impossible future births before aggregation; valid post-2020 parity regression passes. Explicit Table 8 pseudo-age and MC conventions preserved. [Batch patches and verification](../../iterative%20review/evaluations/13/post-review-batch-1/).


**Round 14 independent disposition:** resolved in original scope on m169. [Targeted review](../../iterative%20review/evaluations/14/resolution.md).

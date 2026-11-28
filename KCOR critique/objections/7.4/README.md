# 7.4 - Double-counted overlapping age-shift aggregate

**Severity:** Moderate to high for affected workbook row

**Origin:** New layout-specific defect

**Related issues / root causes:** 6.1, 5.3.

**Finding:** The known age-shift input has two vaccination subgroups and an inclusive group. Ordinary synthesis of an extra direct all-ages row sums all three, double counting individuals. The bootstrap reconstruction correctly uses disjoint histories. The published age-shift curves and Table 8 pooled entries use other paths and are not affected by this specific extra row.

**Proposed scope:** Use the inclusive group once or omit the redundant direct row for the known overlapping layout. Explain the pooled weighting and overlapping groups in the Table 8 caption.

**Review evidence:** [Blind B](../../iterative%20review/evaluations/7/blind-b.md)

**Status:** addressed in m129, preserving m128 (approved batch of straightforward corrections).


The explicit table8 overlap layout now uses pseudo-year 1960 (the inclusive group) exactly once for the direct aggregate fit, inversion and raw comparator. The standard disjoint and supplied-total paths remain unchanged. The archived Table 8 and age-shift replays set the same layout already used by the negative-control Makefile. Actual opening histories and death counts match the inclusive source exactly; the direct curves equal the inclusive curves at all 522 comparison dates/rows. The existing pooled contrast still uses the three overlapping groups. Table 8 primary points match the archive to 2.22e-16, and forced-alpha points agree within 3.10e-7 (the earlier numerical integration change). No new bootstrap; intervals remain as previously reported.

[Combined batch code patch](../../iterative%20review/evaluations/7/post-review-batch-1/workbook.patch) | [Paper patch](../../iterative%20review/evaluations/7/post-review-batch-1/paper.patch) | [Tests](../../iterative%20review/evaluations/7/post-review-batch-1/tests.patch)

**Round 8 independent disposition:** resolved within original scope; see [resolution review](../../iterative%20review/evaluations/8/resolution.md).

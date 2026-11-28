# 19.6 - Raw comparator cutoff rejects extremely small positive ratios

**Severity:** Low numerical edge case for fractional counts

**Origin:** Newly verified specific finding in round 19; no fundamental estimator change proposed.

**Related issues / root causes:** 17.2, 13.6, 12.8, 18.3.

**Finding:** A deterministic fractional-count workbook has a finite positive reference ratio below the fixed EPS threshold. The normalized comparator correctly anchors to 1 but the raw comparator is blank. Real-size Czech counts not implicated.

**Proposed scope (for discussion):** Use existing finite-positive ratio helper in raw comparator paths; preserve undefined results for zero or nonfinite support.

**Review evidence:** [blind-b.md](../../iterative%20review/evaluations/19/blind-b.md)

**Status:** Implemented and verified in persistent m184; independent review pending.

**Approved correction:** Implemented and verified in m184. [Verification](../../iterative%20review/evaluations/19/reader-fixes/README.md).

**Round 20 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/20/resolution.md).

# 19.3 - Missing Monte Carlo dose category aborts an otherwise valid draw

**Severity:** Moderate for sparse or restricted MC samples

**Origin:** Newly verified specific finding in round 19; no fundamental estimator change proposed.

**Related issues / root causes:** 19.1, 19.2, 18.2.

**Finding:** Both blind runs reproduced a length mismatch when the configured highest dose is not present. The pivot renames columns as if all doses were observed; one empty category can abort unrelated valid comparisons.

**Proposed scope (for discussion):** Reindex the pivot to explicit configured dose categories, retaining legitimate zero counts and existing unavailable-comparison conventions.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/19/blind-a.md) | [blind-b.md](../../iterative%20review/evaluations/19/blind-b.md)

**Status:** Implemented and verified in persistent m184; independent review pending.

**Approved correction:** Implemented and verified in m184. [Verification](../../iterative%20review/evaluations/19/producer-fixes/README.md).

**Round 20 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/20/resolution.md).

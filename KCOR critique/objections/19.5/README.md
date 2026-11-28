# 19.5 - Table 8 overlapping input point validation permits impossible nested populations

**Severity:** Moderate for malformed overlapping input; main archives unaffected

**Origin:** Newly verified specific finding in round 19; no fundamental estimator change proposed.

**Related issues / root causes:** 18.2, 16.4, 15.3, 14.4.

**Finding:** The table8 layout accepts individually valid histories even if two constituent opening risks sum above their inclusive total. The point writer gives ok normalized results, while the disjoint-cell bootstrap rejects the input. Both actual archived Table 8 sheets passed disjoint reconstruction.

**Proposed scope (for discussion):** Reuse overlap-feasibility checks before affected point estimates, including deaths and final survivor remainder; retain valid unrelated outputs. Do not change the Table 8 analysis or intended overlap.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/19/blind-a.md)

**Status:** Implemented and verified in persistent m184; independent review pending.

**Approved correction:** Implemented and verified in m184. [Verification](../../iterative%20review/evaluations/19/overlap-fix/README.md).

**Round 20 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/20/resolution.md).

# 18.2 - Monte Carlo aggregate validation bypasses exhaustive source reconciliation

**Severity:** Moderate for malformed MC inputs

**Origin:** Newly verified current defect; regression status not established.

**Related issues / root causes:** 14.4, 15.3, 16.4, 17.2, 18.1.

**Finding:** An internally valid supplied all-ages history contradicts its exhaustive constituent counts. The MC branch skips source reconciliation and reports a substantially altered direct result with status ok; unaffected age and pooled results remain unchanged.

**Proposed scope (for discussion):** Apply existing feasible-history and total consistency rules independently by replicate before display filtering, withholding only affected results.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/18/blind-a.md)

**Status:** Implemented and verified in persistent m182; independent review pending.

**Approved correction:** Addressed together with 18.1-18.4 using coherent per-replicate full-source preparation, validation, raw comparator and reporting. [Verification](../../iterative%20review/evaluations/18/mc-source-consistency/README.md).

**Round 19 independent disposition:** Original scope resolved according to targeted reviewer progress before usage-limit interruption. [Recovered progress](../../iterative%20review/evaluations/19/resolution-progress.md).

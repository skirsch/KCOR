# 18.3 - Monte Carlo all-ages raw comparator loses replicate identity

**Severity:** Moderate for secondary raw comparison output

**Origin:** Newly verified current defect; regression status not established.

**Related issues / root causes:** 16.2, 17.1, 18.1.

**Finding:** The raw comparator KCOR_ns groups across mc_id and emits no replicate ID. In written disjoint-component workbooks it is blank; with totals-only inputs the same combined value is broadcast to two different replicate trajectories. Normalized KCOR is unaffected in the tested fixture.

**Proposed scope (for discussion):** Preserve mc_id in counts, cumulative hazards, pairing, anchor, and output; compare each replicate against direct arithmetic for both layouts.

**Review evidence:** [blind-b.md](../../iterative%20review/evaluations/18/blind-b.md)

**Status:** Implemented and verified in persistent m182; independent review pending.

**Approved correction:** Addressed together with 18.1-18.4 using coherent per-replicate full-source preparation, validation, raw comparator and reporting. [Verification](../../iterative%20review/evaluations/18/mc-source-consistency/README.md).

**Round 19 independent disposition:** Original scope resolved according to targeted reviewer progress before usage-limit interruption. [Recovered progress](../../iterative%20review/evaluations/19/resolution-progress.md).

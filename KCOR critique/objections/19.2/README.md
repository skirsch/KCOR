# 19.2 - Monte Carlo converter truncates configured birth-year range

**Severity:** High for nondefault MC source ranges

**Origin:** Newly verified specific finding in round 19; no fundamental estimator change proposed.

**Related issues / root causes:** 18.1, 18.4, 17.1.

**Finding:** The producer uses the requested bounds to select its enrollment master but then hard-codes 1930-1960 for the output grid. Both independent tests lost all 1970 population and deaths under a request including 1970. Inclusive totals derived from this grid also omit them. Default range not affected.

**Proposed scope (for discussion):** Build the grid from eligible sampled records within configured bounds; reconcile opening populations and deaths to sampled records. Preserve the default intended range and existing estimator.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/19/blind-a.md) | [blind-b.md](../../iterative%20review/evaluations/19/blind-b.md)

**Status:** Implemented and verified in persistent m184; independent review pending.

**Approved correction:** Implemented and verified in m184. [Verification](../../iterative%20review/evaluations/19/producer-fixes/README.md).

**Round 20 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/20/resolution.md).

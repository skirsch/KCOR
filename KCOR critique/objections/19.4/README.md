# 19.4 - Changed Monte Carlo configuration can reuse stale generated data

**Severity:** Moderate for reproducibility of altered MC settings

**Origin:** Newly verified specific finding in round 19; no fundamental estimator change proposed.

**Related issues / root causes:** 19.2, 18.4.

**Finding:** The code Makefile stamp name encodes enrollment and iterations but not the requested birth range or aggregate switch. In scratch make dry-runs, changing either setting returned Nothing to be done.

**Proposed scope (for discussion):** Track the effective full MC configuration and rebuild on a change, including switching a setting back to a previous value.

**Review evidence:** [blind-b.md](../../iterative%20review/evaluations/19/blind-b.md)

**Status:** Implemented and verified in persistent m184; independent review pending.

**Approved correction:** Implemented and verified in m184. [Verification](../../iterative%20review/evaluations/19/producer-fixes/README.md).

**Round 20 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/20/resolution.md).

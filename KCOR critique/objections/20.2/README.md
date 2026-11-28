# 20.2 - Contradictory Monte Carlo ISO week labels alter fitted results

**Severity:** High for affected malformed MC inputs

**Origin:** Newly verified specific finding in round 20.

**Related issues / root causes:** 19.1, 18.1, 18.2, 14.5.

**Finding:** Both reviewers identified a discrepancy between actual DateDied and supplied ISOweekDied. With identical dates and counts, shifting only ISO labels by eight weeks changed the fitting mask and moved an endpoint from 1.553030 to 2.116018 with fit/status OK. These are deterministic fixture points, not interval estimates. No published input shown affected.

**Proposed scope (for discussion):** Derive canonical ISO weeks from validated dates before aggregation and fitting; reject discordant labels with explicit affected-group status, accepting equivalent formatting. Test conflicting and duplicate labels and parity on canonical input.

**Review evidence:** [blind-b.md](../../iterative%20review/evaluations/20/blind-b.md) | [blind-a.md](../../iterative%20review/evaluations/20/blind-a.md)

**Status:** Implemented and verified in persistent m185; independent review pending.

**Approved correction:** Implemented and verified in m185. [Evidence](../../iterative%20review/evaluations/20/mc-diagnostics-calendar/README.md).

**Round 21 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/21/resolution.md).

# 20.1 - Monte Carlo summary conceals flagged frailty fits

**Severity:** High for interpretation of affected MC intervals

**Origin:** Newly verified specific finding in round 20.

**Related issues / root causes:** 19.1, 18.3, 16.1, 3.1.

**Finding:** A written workbook with finite fallback output but NOT_IDENTIFIED for both cohorts in every iteration was summarized as ok, with unqualified percentile and basic intervals. Numerical completion is not successful identification. Ordinary bootstrap separately preserves and reports flags.

**Proposed scope (for discussion):** Use the same ordinary fit-flag predicate, count and label flagged original/draws and audit-only intervals without removing numerical fallback draws or changing interval arithmetic. Preserve missing-draw rules and disclose available diagnostics separately.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/20/blind-a.md)

**Status:** Implemented and verified in persistent m185; independent review pending.

**Approved correction:** Implemented and verified in m185. [Evidence](../../iterative%20review/evaluations/20/mc-diagnostics-calendar/README.md).

**Round 21 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/21/resolution.md).

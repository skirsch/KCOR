# 16.1 - Partial aggregate may silently compare different age-group coverage across doses

**Severity:** Moderate for affected partial ordinary inputs

**Origin:** New related population issue after 15.2

**Related issues / root causes:** 15.2, 14.4, 11.1.

**Finding:** An omitted age-by-dose group changes the partial aggregate endpoint and still has an OK status and complete draws labeled All available ages. This is not a requirement for equal age proportions in fully observed cohorts; the issue is undisclosed missing component support.

**Proposed scope (for discussion):** Record age-group coverage separately for each dose, and distinguish an omitted group from a documented zero. Where support differs, explicitly identify the two populations or withhold a shared-population comparison; preserve independently valid groups. No core fitter change.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/16/blind-a.md) | [Blind B](../../iterative%20review/evaluations/16/blind-b.md) | [Resolution](../../iterative%20review/evaluations/16/resolution.md)

**Status:** Implemented in m179; targeted point/interval and coverage-label verification completed.


**Approved correction:** Report each dose's available supplied age-group coverage and recorded-zero enrollment groups, distinguish missing vs documented zero, and compare coverage in bootstrap draws. Original point and interval unchanged. [Verification](../../iterative%20review/evaluations/16/age-coverage/README.md).

**Round 17 independent disposition:** Resolved in original scope. Coverage reporting and bootstrap mismatch handling verified. [Verification](../../iterative%20review/evaluations/17/resolution.md).

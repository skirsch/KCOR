# 15.2 - Partial extract silently constructs an all-ages result from displayed strata

**Severity:** High for affected ordinary partial workbook output

**Origin:** New consequence of 14.4

**Related issues / root causes:** 14.4, 11.1, 12.1, 13.2.

**Finding:** Both blind reviewers reproduced wrong full-population point and three-draw interval when a partial extract contains an older displayed stratum plus a supplied wider 1960+ total, but no all-ages total. A separate exhaustive case loses a valid pair when only one dose has an explicit total.

**Proposed scope (for discussion):** Resolve aggregate coverage by dose. Withhold or explicitly identify a displayed-subset result unless full disjoint coverage is established; reconstruct missing totals from complete components in exhaustive mode. Use the same population for point and resampling.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/15/blind-a.md) | [Blind B](../../iterative%20review/evaluations/15/blind-b.md) | [Resolution](../../iterative%20review/evaluations/15/resolution.md)

**Status:** Implemented in m175; targeted regression and S6 point parity verified.


**Approved correction:** Use the explicit label All available ages for the partial disjoint aggregate. Keep supplied full totals separate, reject mixed full/partial population comparisons, reconstruct a missing dose-specific total only under exhaustive scope, and enforce population-consistent bootstrap reporting. No core fitter change. [Verification](../../iterative%20review/evaluations/15/available-ages/README.md).


**Round 16 independent disposition:** Resolved in the original scope by independent targeted round-16 review. [Resolution](../../iterative%20review/evaluations/16/resolution.md).

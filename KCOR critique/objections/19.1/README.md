# 19.1 - Monte Carlo confidence interval mixes reporting dates and drops unavailable draws

**Severity:** High for affected MC summary intervals

**Origin:** Newly verified specific finding in round 19; no fundamental estimator change proposed.

**Related issues / root causes:** 18.3, 16.2, 14.5, 8.2.

**Finding:** Both blind reviewers and targeted reviewer verified the summary silently substitutes separate closest dates and ignores undefined or absent draws; it produces percentile and basic intervals whose endpoint population/date is not shared by the original estimate. Correctly dated weekly rows are not the problem.

**Proposed scope (for discussion):** Choose the reporting date once for each original comparison; require the same date, source population and complete planned draw roster. Expose unavailable reasons; withhold interval if any required draw is absent. Preserve the existing ordinary bootstrap.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/19/blind-a.md) | [blind-b.md](../../iterative%20review/evaluations/19/blind-b.md) | [resolution-progress.md](../../iterative%20review/evaluations/19/resolution-progress.md)

**Status:** Implemented and verified in persistent m184; independent review pending.

**Approved correction:** Implemented and verified in m184. [Verification](../../iterative%20review/evaluations/19/reader-fixes/README.md).

**Round 20 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/20/resolution.md).

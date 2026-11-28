# 18.5 - Optional nonlinear fitting described as only a post-inversion correction

**Severity:** Minor but concrete manuscript inconsistency

**Origin:** Newly verified wording inconsistency.

**Related issues / root causes:** 1.2, 11.5, 12.2.

**Finding:** Sections 2.3, 2.6 and 2.7.1 refer to correction only after gamma inversion, while the specified nonlinear survivor law actually enters fitting and inversion (sections 2.5 and 2.7.3). Other passages already state that the same survivor law is used for fitting and correction.

**Proposed scope (for discussion):** Make the narrow wording consistent with the implemented optional model, preserving the distinction that raw observed hazards are not rescaled. Do not change the algorithm or alpha policy.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/18/blind-a.md)

**Status:** Implemented and verified in persistent m183; independent review pending.

**Approved correction:** Consistent survivor-law wording throughout paper and supplement, without changing code, equations or results. [Exact edits](../../iterative%20review/evaluations/18/18.5-wording-edits.json). Clean PDF 78 pages; cumulative redline 85 pages, fourteen visible links to objection 18.5 at page 85.

**Round 19 independent disposition:** Not explicitly assessed before targeted reviewer usage-limit interruption; parent verified the exact edits and redline links and neither blind report repeated the objection. Silence is not independent closure.

**Round 20 independent disposition:** Resolved in approved scope. [Targeted resolution](../../iterative%20review/evaluations/20/resolution.md).

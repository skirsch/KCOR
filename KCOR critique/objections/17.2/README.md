# 17.2 - Exhaustive fractional total reconciliation rejects floating-point roundoff

**Severity:** Low numerical input edge case

**Origin:** Newly identified, pre-existing defect; not a direct return of the previous objection.

**Related issues / root causes:** 16.3, 15.3, 14.4.

**Finding:** A complete fractional expected-count history reconciles mathematically, but its supplied total differs from the floating-point sum of components by 4.55e-13 person. Exact equality rejects a valid all-ages point in both m177 and m180. Integer Czech counts and person-bootstrap results are unaffected.

**Proposed scope (for discussion):** Use scale-aware floating-point equality for fractional exhaustive sums; maintain exact integer validation and rejection of material discrepancies.

**Review evidence:** [Round 17 report](../../iterative%20review/evaluations/17/resolution.md)

**Status:** Implemented and verified in persistent m181; independent review pending.

**Approved correction:** Implemented and verified. [Evidence](../../iterative%20review/evaluations/17/fractional-total/README.md).

**Round 18 independent disposition:** Resolved. Valid fractional total accepted and material fractional and integer inconsistencies rejected. [Verification](../../iterative%20review/evaluations/18/resolution.md).

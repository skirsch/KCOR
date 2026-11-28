# 14.4 - Inconsistent supplied all-ages total silently changes point population

**Severity:** Moderate for malformed ordinary converter inputs

**Origin:** New related representation discrepancy

**Related issues / root causes:** 11.1, 12.1, 13.2.

**Finding:** Supplied internally valid but incorrect total changes endpoint 1.780361 to 0.999944; bootstrap rejects mismatch but point uses it. Explicit partial extracts may legitimately have a wider supplied aggregate.

**Proposed scope (for discussion):** Share reconciliation for declared exhaustive converter layout; preserve explicitly partial extracts, flag invalid totals without dropping unrelated valid age strata.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m173; targeted regression and actual S6 point parity verified. Independent round review pending.


**Approved correction:** The ordinary point and person-bootstrap paths share a supplied inclusive-total reconciliation against the disjoint source histories. The default complete extract rejects mismatched totals while preserving unrelated age results. A valid wider total in a declared partial extract (KCOR_AGGREGATE_SCOPE=partial) retains its point but not an interval reconstructed from absent constituents; incompatible smaller totals are rejected. The overlapping Table 8 layout is unchanged.

[Correction and verification](../../iterative%20review/evaluations/14/aggregate-reconciliation/README.md).


**Round 15 independent disposition:** Original single-total example resolved; general aggregate coverage and nested-total consistency remain partial. See [15.2](../15.2/README.md) and [15.3](../15.3/README.md). [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

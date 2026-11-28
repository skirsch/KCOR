# 14.2 - Manufacturer converter counts deaths outside exact enrollment dose group

**Severity:** Moderate for ancillary manufacturer cohorts

**Origin:** New specific upstream defect

**Related issues / root causes:** 14.1, 9.1, 6.2.

**Finding:** Later-dose persons can contribute deaths to dose d despite being excluded from dose-d denominator. Released manufacturer sums do not reconcile with ordinary exact-dose history (2022-06 1940s dose3: 527477 vs 527414 enrollment, 44030 vs 44026 deaths); not all discrepancy attributed to this mechanism.

**Proposed scope (for discussion):** Verify intended fixed enrollment cohort definition; split consistent exact-dose risk and deaths by product, reconcile with ordinary cohort sums.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m171; targeted regression and converter verification completed. Historical full manufacturer inputs require regeneration; independent review pending.


**Approved correction:** Manufacturer histories now partition the actual ordinary converter enrollment-dose assignment, including the other/unknown category. Weeks start at enrollment and opening counts are reduced only by deaths in that fixed cohort. Manufacturer sums must reconcile with ordinary weekly birth-year/sex/DCCI counts. The test includes a person already on the next dose who is excluded, and an unknown manufacturer retained in the other category.

[Manufacturer correction and verification](../../iterative%20review/evaluations/14/manufacturer-batch/README.md).


**Round 15 independent disposition:** Partial at delivery: corrected code independently verified, but distributed historical manufacturer artifacts remain superseded. See [15.1](../15.1/README.md). [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

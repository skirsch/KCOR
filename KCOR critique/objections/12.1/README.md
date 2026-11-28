# 12.1 - Post-2020 births excluded before all-ages reconstruction

**Severity:** High for affected ordinary workbooks

**Origin:** Direct return of 11.1; unknown-year fixture fixed

**Related issues / root causes:** 11.1, 9.1, 9.2.

**Finding:** Hard-coded 2020 cutoff removes valid births from reconstructed all-ages population but retains them in supplied totals. Three independent fixtures confirm divergent endpoints. Czech S6 eligibility ends in 2005, unaffected.

**Proposed scope (for discussion):** Remove hard-coded cutoff; verify population eligibility consistently and repeat post-2020 representation parity test.

**Review evidence:** A, B, R. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** implemented in m162; m161 preserved. Independent closure not yet evaluated.


Removed obsolete calendar cutoff; retain configured age display range and MC range. Valid 2023 enrollment with post-2020 birth group gives identical all-ages point and all three matched bootstrap intervals in both input representations: 1.981297 [1.794494, 2.140051] (three-draw smoke only). Published Czech input eligibility unaffected. [Batch patches and PDFs](../../iterative%20review/evaluations/12/post-review-batch-1/).


**Round 13 independent disposition:** resolved in original scope on m167. [Targeted review](../../iterative%20review/evaluations/13/resolution.md).

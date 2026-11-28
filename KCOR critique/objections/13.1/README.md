# 13.1 - Floating-point roundoff creates spurious positive wave exposures

**Severity:** Minor for identity diagnostics

**Origin:** Direct return of 12.2; strict-deficit and denominator fixtures resolved

**Related issues / root causes:** 12.2, 11.14.

**Finding:** Exact 52-week baseline with 50 marked weeks returns 26 positive-exposure weeks solely from rounding, accumulated exposure 3.07e-16. Strict deficits zero and total curve unchanged.

**Proposed scope (for discussion):** Use a floating-point-scale equality tolerance for the identity attribution, preserve strict deficits and cumulative curves; regress all 50 exact weeks.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/13/blind-a.md) | [Blind B](../../iterative%20review/evaluations/13/blind-b.md) | [Resolution](../../iterative%20review/evaluations/13/resolution.md)

**Status:** implemented in m168; m167 preserved. Independent closure not yet evaluated.


Floating-point-scale equality tolerance; 50 noiseless marked weeks have zero positive exposure and zero strict deficits in both identity cases. Three-week true-deficit regression passes. [Batch patches and verification](../../iterative%20review/evaluations/13/post-review-batch-1/).


**Round 14 independent disposition:** resolved in original scope on m169. [Targeted review](../../iterative%20review/evaluations/14/resolution.md).

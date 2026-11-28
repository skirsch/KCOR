# 11.14 - Wave deficit frequency not exported

**Severity:** Minor diagnostic reporting

**Origin:** New specific reporting omission

**Related issues / root causes:** 4.3, 5.10, 10.5.

**Finding:** Three-week test: one strict deficit, one baseline equality, one positive excess; skipped_nonpositive_excess=2 but deficit_weeks=1. Workbook reports only skipped count.

**Proposed scope (for discussion):** Export strict-deficit count and wave-week denominator separately.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Wave diagnostics export mode, actual wave weeks, strict deficit count and zero-excess count; missing strict-count mode explicitly unavailable. Forced-alpha actual point run verified diagnostic records. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** partial; nonlinear reporting resolved; identity/zero-frailty diagnostics remain (12.2). [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

# 11.3 - Pooled geometric mean silently clips representable ratios

**Severity:** Moderate

**Origin:** New specific numerical defect

**Related issues / root causes:** 9.8, 10.8.

**Finding:** With one stratum, direct finite KCOR 2.043049e10 became pooled 1e6 despite OK normalization.

**Proposed scope (for discussion):** Remove arbitrary log/exp clipping for the pooled statistic; report out-of-range numerical failures.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Unclipped positive finite geometric mean; sole-stratum extreme test direct and pooled both 21082653773.60724; numerical-range failure unavailable. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

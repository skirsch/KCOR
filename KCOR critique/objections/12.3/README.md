# 12.3 - SI figure still labels finite deviation as asymptotic bias

**Severity:** Minor

**Origin:** Direct return of 11.15; text corrected

**Related issues / root causes:** 11.15, 2.1.

**Finding:** Embedded fig_si_kcor_bias_vs_theta.png and generator retain y-axis Bias in KCOR asymptote (KCOR minus 1), despite corrected caption.

**Proposed scope (for discussion):** Rebuild from archived endpoints with observed week-52 deviation label; verify actual image.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** implemented in m162; m161 preserved. Independent closure not yet evaluated.


Corrected generator axis and rebuilt actual embedded figure from unchanged archived endpoints; visually verified axis reads Observed week-52 KCOR minus one. [Batch patches and PDFs](../../iterative%20review/evaluations/12/post-review-batch-1/).


**Round 13 independent disposition:** resolved in original scope on m167. [Targeted review](../../iterative%20review/evaluations/13/resolution.md).

# 8.9 - Diagnostic reporting guidance and reversed SI decision-table consequence

**Severity:** Moderate for interpretation

**Origin:** Returning 7.6

**Related issues / root causes:** 7.6, 6.3, 5.8: diagnostic flags versus substantive identification

**Finding:** Czech numerical/residual/omission reporting is now correct, but general sections still say contrasts are never reported on diagnostic failure. SI S2 row I1 says that violating all-required-diagnostics-pass implies KCOR interpretable; I7 also uses inconsistent condition/consequence.

**Proposed scope (for discussion):** Correct remaining language and table logic; preserve exploratory flags, distinguish flagged descriptive audit output, failed identification and undefined computation. No threshold change.

**Review evidence:** B, resolution. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m136 (approved straightforward-correction batch; m135 preserved). Independent closure not yet evaluated.

Corrected remaining diagnostic policy wording in sections 2.1.2 and 4.4, and the reversed SI S2 I1/I7 condition/consequence. No new diagnostic threshold or estimator change.

[Batch manuscript and code patches](../../iterative%20review/evaluations/8/post-review-batch-1/).

**Round 9 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/9/resolution.md).

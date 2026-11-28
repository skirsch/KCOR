# 10.5 - Nonlinear fit inherits proportional initializer gap diagnostic

**Severity:** Moderate for optional module

**Origin:** New specific defect

**Related issues / root causes:** 9.7, 9.12, 1.2.

**Finding:** Exact nonlinear wave with every other week quiet has no ordinary gap, yet forwarded first-pass offsets at wave exit +0.221537 and +0.0701172 in two independent designs.

**Proposed scope (for discussion):** Clear initializer-only gap diagnostics; report no_gaps for complete modeled history and not_assessed for true unmarked gaps until an appropriate diagnostic is available.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/10/blind-a.md) | [Blind B](../../iterative%20review/evaluations/10/blind-b.md) | [Resolution](../../iterative%20review/evaluations/10/resolution.md)

**Status:** implemented in m153; m152 preserved. Independent closure not yet evaluated.

The marked-wave fit no longer forwards proportional-initializer gap offsets. A successful nonlinear history with no unmarked gaps reports no_gaps; unmarked gaps are explicitly unassessed by the proportional screen. Exact test recovers theta 4 in both cases. Paper clarifies scope; no survivor-model change. [Batch patches and regression](../../iterative%20review/evaluations/10/post-review-batch-1/).


**Round 11 independent disposition:** resolved in original scope on m156. [Targeted review](../../iterative%20review/evaluations/11/resolution.md).

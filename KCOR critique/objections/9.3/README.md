# 9.3 - Table 8 resume ignores anchor override

**Severity:** Moderate

**Origin:** Direct return; direct predecessor 8.10

**Related issues / root causes:** 7.8.

**Finding:** Changing SA_ANCHOR_WEEKS from 4 to 8 reuses 1.045599; fresh calculation gives 0.966288. Default fresh run behaves correctly.

**Proposed scope (for discussion):** Hash resolved effective settings including reference index; regression with changed anchor.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m143; m142 preserved. Independent closure not yet evaluated.

Resume provenance now includes SA_, MC_ and sensitivity-mode settings. A one-draw fresh run schedules four jobs, identical resume zero; changing anchor four to eight is rejected. [Batch patches and verification](../../iterative%20review/evaluations/9/post-review-batch-1/).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).

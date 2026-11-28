# 9.14 - Current Table 8 verifier reads previous numerical ceiling

**Severity:** Low

**Origin:** New specific command defect

**Related issues / root causes:** 8.10, 8.4.

**Finding:** Documented checker fails tolerance 1e-8 after cap500; max old/new point difference 2.425e-7; printed numbers and current archived percentiles match.

**Proposed scope (for discussion):** Read current ceiling500 archives; do not relax tolerance.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m143; m142 preserved. Independent closure not yet evaluated.

Verifier and archive documentation now select ceiling500 results. All 12 ordinary pooled points reproduced to 2.23e-16; archived intervals retained, not freshly rerun. [Batch patches and verification](../../iterative%20review/evaluations/9/post-review-batch-1/).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).

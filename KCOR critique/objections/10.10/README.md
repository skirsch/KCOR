# 10.10 - Fixed-cohort wording still overstates avoidance of immortal-time artifacts

**Severity:** Low wording; known application limitation

**Origin:** Recurring disclosed limitation, narrow wording issue; direct predecessor 9.16

**Related issues / root causes:** 6.14, 7.5.

**Finding:** Section 2.2 says fixed design avoids immortal-time artifacts, while later eligibility disclaimer and S6 disclose retrospective selection. 363,033 excluded solely at/after enrollment, effect size/direction unknown.

**Proposed scope (for discussion):** Narrow the general sentence to initial exposure-label assignment, retain intended Czech population and previously disclosed caveat. Do not redesign application.

**Review evidence:** A, R. [Blind A](../../iterative%20review/evaluations/10/blind-a.md) | [Blind B](../../iterative%20review/evaluations/10/blind-b.md) | [Resolution](../../iterative%20review/evaluations/10/resolution.md)

**Status:** implemented in m153; m152 preserved. Independent closure not yet evaluated.

Exposure assignment at enrollment prevents later treatment from determining initial labels; the separate retrospective eligibility disclosure is retained. No Czech population change. [Batch patches and regression](../../iterative%20review/evaluations/10/post-review-batch-1/).


**Round 11 independent disposition:** resolved in original scope on m156. [Targeted review](../../iterative%20review/evaluations/11/resolution.md).

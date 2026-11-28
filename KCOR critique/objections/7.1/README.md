# 7.1 - Unreported normalization fallback after numerical failure

**Severity:** High

**Origin:** New implementation defect

**Related issues / root causes:** 5.11, 6.12.

**Finding:** The ordinary sheet-wide catch continues with preassigned hazards after a normalization exception. In a controlled scratch run the output ratio changed from 0.897540 to 0.887304 despite identical fitted parameters and OK/blank flags. This is an error-path test, not evidence that actual archived results encountered it.

**Proposed scope:** Mark affected normalized trajectories and dependent contrasts unavailable, propagate the failure to workbook and bootstrap diagnostics, retain successful groups. Do not alter the estimator's mathematical equations.

**Review evidence:** [Blind B](../../iterative%20review/evaluations/7/blind-b.md)

**Status:** addressed in m129, preserving m128 (approved batch of straightforward corrections).


The numerical correction loop now handles failures per cohort, preserves successful cohorts, emits unavailable normalized values instead of retaining the earlier substitute hazard, and records normalization_status independently of the original fit. Dependent direct and positive-weight pooled contrasts are unavailable (not reweighted); the bootstrap flagger includes normalization failures. The existing fit/fallback policy is unchanged. Controlled error injection on archived age-shift input passes, and a two-draw ordinary workbook output test confirms unavailable point and intervals with CI_point_flagged=true. No evidence of the original error occurring in past runs.

[Combined batch code patch](../../iterative%20review/evaluations/7/post-review-batch-1/workbook.patch) | [Paper patch](../../iterative%20review/evaluations/7/post-review-batch-1/paper.patch) | [Tests](../../iterative%20review/evaluations/7/post-review-batch-1/tests.patch)

**Round 8 independent disposition:** resolved within original scope; see [resolution review](../../iterative%20review/evaluations/8/resolution.md).

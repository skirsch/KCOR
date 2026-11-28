# 8.13 - Near-zero frailty confused with linear cumulative hazard

**Severity:** Low, wording

**Origin:** New specific mathematical wording mistake

**Related issues / root causes:** 2.2, 1.6: Gompertz curvature

**Finding:** SI S6.1.2 associates near-zero frailty with linear cumulative hazard. At zero variance the observed cumulative hazard is Gompertz, not generally linear; exact linearity at positive slope can occur at k theta = gamma. Main proposition is correct.

**Proposed scope (for discussion):** Correct SI wording to low variance/weak resolution rather than literal linearity.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m136 (approved straightforward-correction batch; m135 preserved). Independent closure not yet evaluated.

Corrected SI S6.1.2: zero frailty follows Gompertz baseline, not necessarily linear cumulative hazard; weak identification assessed separately.

[Batch manuscript and code patches](../../iterative%20review/evaluations/8/post-review-batch-1/).

**Round 9 independent disposition:** partial; see [9.4](../9.4/README.md). [Resolution](../../iterative%20review/evaluations/9/resolution.md).

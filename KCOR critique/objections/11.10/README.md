# 11.10 - Cox comparator inference and interpretation under nonproportional hazards

**Severity:** Moderate

**Origin:** New specific reporting/inference question

**Related issues / root causes:** 1.5, 2.1, 7.11, 8.12.

**Finding:** Conventional model-based Cox Wald errors/p-values under deliberate non-PH are labeled 95% CIs without inferential qualification; comparison table overstates absence of interpretable output and labels departure biased HR without explicit target.

**Proposed scope (for discussion):** Distinguish fixed-frailty benchmark from follow-up-dependent working summary; label nominal model-based inference, or use suitable robust/person-resampled inference with approval.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** documented in m159; m158 preserved. Independent closure not yet evaluated.


Retained the original Cox coefficients, nominal model-based Wald limits and p-values; no new inferential calculation or simulation. Section 2.11.1, figure caption and table identify nominal inference; comparison table distinguishes marginal constant-coefficient summary and conditional shared-frailty HR. Figure regenerated from identical archived CSV.


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

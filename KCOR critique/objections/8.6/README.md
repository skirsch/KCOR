# 8.6 - Mean frailty and stabilization-skip null benchmark

**Severity:** Moderate to high for interpretation

**Origin:** New explicit counterexamples, related earlier estimand concerns

**Related issues / root causes:** 1.1, 1.5, 6.9, 7.11: equal reference-frailty versus equal intrinsic treatment effect

**Finding:** Both blind reviewers distinguish mean frailty from relative heterogeneity. With actual-enrollment variances 10 and 0, common baseline exposure 0.02 during the skip, perfect post-skip inversion gives unanchored KCOR 5/6 and anchored 1 under a fixed absolute-frailty treatment null. Shifted mean frailty also changes reference baseline without changing theta. Raw cumulative hazard itself is not invariant to depletion.

**Proposed scope (for discussion):** Clarify the post-skip reference person and equal-reference-baseline versus proportional-reference benchmark. Preserve the intended anchored and unanchored estimands, not redesign them.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m141 (wording only); m140 preserved. Independent closure not yet evaluated.

Exact derivation: a mean-one gamma frailty Z with initial variance theta, after common skip cumulative baseline B, has survivor mean mu=1/(1+theta B). Its gamma relative variance after rescaling by mu is still theta. For a subsequent common absolute baseline increment U, observed cumulative hazard increment is log(1+theta mu U)/theta, and exact gamma inversion gives mu U. Thus with theta_A=10, theta_B=0 and B=.02, mu_A=5/6, mu_B=1; unanchored ratio=5/6, and division by its reference ratio gives anchored 1. [Numerical check](exact-null-example.json). This confirms the reference-frailty benchmark, not a defect in inversion. Section 1.2 now distinguishes mean and relative frailty heterogeneity; Box 2 identifies the post-skip within-cohort mean reference; Section 2.1 attributes correction to inversion rather than raw cumulative accumulation; Section 3.3 qualifies the negative-control benchmark. Both estimands, code and all numerical results unchanged. [Patch](paper.md.patch).

**Round 9 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/9/resolution.md).

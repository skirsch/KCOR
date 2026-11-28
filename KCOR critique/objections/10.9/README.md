# 10.9 - Diagnostics cannot guarantee failure in unsuitable designs

**Severity:** Low to moderate scientific wording

**Origin:** New specific counterexample

**Related issues / root causes:** 1.4, 8.9, 9.7.

**Finding:** Section 2.14 says required diagnostics will fail in unsuitable designs. A homogeneous time-varying hazard can reproduce exactly the gamma-Gompertz marginal survival; observed checks cannot distinguish mechanisms.

**Proposed scope (for discussion):** Qualify guarantee: checks may expose misspecification but observational equivalence can pass.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/10/blind-a.md) | [Blind B](../../iterative%20review/evaluations/10/blind-b.md) | [Resolution](../../iterative%20review/evaluations/10/resolution.md)

**Status:** implemented in m153; m152 preserved. Independent closure not yet evaluated.

The applicability statement no longer promises diagnostics always detect misspecification; interpretation remains conditional on the requirements. [Batch patches and regression](../../iterative%20review/evaluations/10/post-review-batch-1/).


**Round 11 independent disposition:** resolved in original scope on m156. [Targeted review](../../iterative%20review/evaluations/11/resolution.md).

# 8.5 - Overgeneralized necessity of an external effect scale for alpha

**Severity:** Moderate scientific claim, optional module

**Origin:** New specific counterexample to older alpha assumption

**Related issues / root causes:** 4.7, 5.10, 6.13, 7.10: alpha identification and precision

**Finding:** Blind A constructs a homogeneous reference and a gamma cohort with known quiet parameters; two aggregate weekly survival observations under a constant unknown multiplier yield a full-rank Jacobian for multiplier and alpha (determinant -5.5082e-5). Local identifiability under that special model is not a claim of Czech identification or useful precision.

**Proposed scope (for discussion):** Verify the derivation; distinguish the existing localization procedure requiring a specified effect scale from structural identifiability under additional longitudinal assumptions. No new alpha estimator or Czech claim.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m140 (wording only), preserving m139-review. Independent closure not yet evaluated.

Independent reproduction during 8.4 validation: using the two exact weekly-boundary survival observations from Blind A, numerical integration gives H=(0.04049789,0.08143710), Jacobian rows (0.02587292,0.00883113) and (0.06371760,0.01961961), determinant -5.50820574e-5, and root solver recovers constant multiplier 0.7 and alpha 1.18. This verifies the local-identification counterexample, not identification in actual Czech data. No edits adopted.

The abstract, contribution summary, Sections 2.7.1, 2.7.2 and 5.4, and SI S2/S2.1 now consistently describe a requirement of the proposed localization procedure rather than a universal non-identifiability theorem. Section 2.7.2 adds one short qualification for repeated survival observations with known baseline/frailty, common wave exposure and a constant unknown multiplier. The joint longitudinal procedure is not evaluated or implemented; actual Czech non-identification conclusion unchanged. [Counterexample script](verify_counterexample.py) | [Numerical result](counterexample.json) | [Paper patch](paper.md.patch) | [Supplement patch](supplement.md.patch).

**Round 9 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/9/resolution.md).

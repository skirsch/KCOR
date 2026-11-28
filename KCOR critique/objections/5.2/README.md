# 5.2 - Round 5 finding

**Severity:** Moderate

Zero early mortality prevents pooled optimization despite later positive quiet data. Separate an initialization from a required substantive anchor.

**Related issues:** 4.5, sparse information

**Review evidence:** A B R - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** initialization correction implemented in m104; m103 frozen.

The joint fitter seeds from the mean of eligible quiet weekly hazards divided by the unit-scale integrated Gompertz weekly hazard, bounded to the existing optimizer range, only when the early seed is nonpositive or unavailable. Positive early seeds and the legacy fixed-k seed path are unchanged. No zero hazards are removed from the objective. No reporting anchor, objective, death threshold or diagnostic changed.

Verification: both module-off and specified alpha=1.18 fits now reach the optimizer with 310 later quiet deaths. Both retain the existing insufficient_signal flag because the relative-RMSE denominator at observed zero is 1e-12; this is not evidence of low frailty or inability to optimize. Keep this as a separate diagnostic question linked to 5.8. Positive early seed yields exactly the same result; all-zero mortality still fails, and 11 quiet deaths trigger the existing low-death policy. See [test](verify.py), [old](m103-regression.json), [new](m104-regression.json), [patch](KCOR.py.patch).

Table 8: all point estimates and seven selected matched resamples per enrollment per mode exactly match the previous saved values. Full 999-draw intervals are from m100, not newly rerun. [Primary check](identified_only-table8-check.json), [sensitivity check](forced_alpha-table8-check.json).

The paper already describes the joint seed and pooled optimization; no text change was necessary. No change to the actual draft PR.

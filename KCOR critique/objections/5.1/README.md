# 5.1 - Round 5 finding

**Severity:** High for optional wave fit

The final mask excludes wave rows, but their deaths and quiet-bin counts still contribute to the seed-based minimum-death/weak-information flags. All three independently reproduced this.

**Related issues:** 4.1, eligibility consistency

**Review evidence:** A B R - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** corrected in m103, m102 frozen.

The wave fitter now filters the quiet mask before seed-gate calculation, updates the diagnostic counts and final objective mask, rejects fewer than three eligible quiet weeks, and recalculates the existing weak-identification rule on final fit rows and the fitted exposure span (B+W, reducing to the proportional span at alpha=1). Anchor-only rows may participate in the objective but do not qualify as quiet deaths. The configured minimum 30, existing low-death fallback, optimization equations, and wave attribution are unchanged. The paper already specifies the eligible residual rows; no wording change needed.

Regression: 26 eligible deaths now yield insufficient_deaths with correct count for alpha=1 and 1.18; 30 and 46 clear the death gate. Both zero-row and anchor-only cases fail, and five eligible fit rows trigger weak identification. Previous alpha continuity regression preserved. See [test](verify.py), [old results](m102-regression.json), [new results](m103-regression.json), and [patch](kcor_survivor.patch).

Table 8: configured quiet/wave overlap zero in both enrollments. Six original points and seven matched resamples per enrollment per mode match saved m100 values exactly with no failed/flagged draws in the check. The existing 999-draw intervals remain from m100; a new full 999-draw replay was not performed. [Check](check_table8.py), [primary](identified_only-table8-check.json), [sensitivity](forced_alpha-table8-check.json). No change to the actual draft PR.

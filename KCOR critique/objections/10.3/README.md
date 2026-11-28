# 10.3 - Hard-coded seed ceiling falsely flags a baseline gap

**Severity:** High

**Origin:** New specific counterexample

**Related issues / root causes:** 8.4, 9.7, 9.11.

**Finding:** Seed fit capped at 100 while final fit ceiling is 500. True variance 200, exact zero-excess gap: final theta 200 but first-pass offsets -0.011288 and -0.0534085 in two independent designs; sampled 49-draw intervals entirely negative despite correct model.

**Proposed scope (for discussion):** Use configured ceiling and feasible start in seed, flag bound-truncated seed as diagnostic unassessed, regress exact/sampled cases and refresh affected diagnostics.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/10/blind-a.md) | [Blind B](../../iterative%20review/evaluations/10/blind-b.md) | [Resolution](../../iterative%20review/evaluations/10/resolution.md)

**Status:** implemented in m154; m153 preserved. Independent closure not yet evaluated.


Known-truth regression: true variance 200 recovered in seed and pooled fit; zero-gap 95% percentile interval [-0.002792, 0.002726] (49 draws). True variance 550 gives a bounded seed (500) and unassessed diagnostic. All six primary Table 8 point estimates and intervals unchanged at printed precision after 999 matched draws. Of 54 gap intervals, 30 assessed with no deficit detected and 24 unassessed, not passes. Forced-alpha 49-draw smoke completed; full forced-alpha intervals and other archived analyses are not newly verified.

S6 fresh 999-draw replay: all fourteen frailty estimates and intervals and seven KCOR endpoints and intervals agree at published precision. Evidence in s6/.


**Round 11 independent disposition:** resolved in original scope on m156. [Targeted review](../../iterative%20review/evaluations/11/resolution.md).

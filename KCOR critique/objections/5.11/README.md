# 5.11 - Round 5 finding

**Severity:** Minor to moderate, edge cases

Death-fraction clipping, horizon-wide inversion cap, flagged fallback and secondary analytic interval limitations. The cap is already disclosed but not flagged; extending follow-up may change earlier inverted values.

**Related issues:** 4.9, 4.6

**Review evidence:** A B - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** the two approved numerical corrections are implemented in m116; m115 is preserved. Finite weekly death fractions below one use the exact log survival transformation. A fraction of one or outside the valid range raises a warning. The gamma inverse uses the fitted variance independently at each time and evaluates large positive exponents in log space when needed. Unrepresentable results are unavailable rather than clipped. Normalization preserves the valid prefix, including nonlinear correction on that prefix, and logs the first unavailable week per cohort.

**Verification:** [deterministic regression](verify.py) recovers hazard 9.21034037 at a death fraction of 0.9999 and preserves 0.0162178123 at cumulative observed hazard 0.01 with theta=90, even when later values are added. An exponent of 711 still yields a representable inverse (6.74736e306); an exponent of 810 yields an unavailable result. A workbook smoke test exhausts the exposed cohort at the last week, with both its birth-year and all-ages normalized hazards and anchored/unanchored ratios unavailable and both groups flagged in the run log. This is an edge-case check, not a full bootstrap replay. Secondary analytical interval limitations and other fit fallback policies were not changed.

[Implementation patch](KCOR.py.patch) · [Manuscript patch](paper.patch).

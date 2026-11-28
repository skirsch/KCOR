# 5.9 - Round 5 finding

**Severity:** Moderate scientific claim

Common Gompertz slope misspecification need not cancel to first order. Both provide counterexamples; confirm actual sensitivity evidence or narrow wording.

**Related issues:** 1.1, slope/model conditionality

**Review evidence:** A B - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** addressed in candidate m114. Section 2.4.3 explains why similar percentage changes in the two normalized cumulative hazards have little effect on KCOR and quantifies the effect in the specified 40-week example. This folder holds the full derivation.

The [mathematical verification](mathematical-review.md) gives the exact cancellation condition and relates it to the all-window weekly joint fit. With the specified hazard histories, changing the annual slope from 0.085 by ±0.020 moved anchored KCOR by about 0.4% in either direction. These are deterministic sensitivity values. The example gives no estimate of practical sensitivity for Table 8.

To reproduce the example, run `python verify_slope_example.py PATH_TO_KCOR_CHECKOUT` from this folder with the candidate's Python dependencies installed. The [verification script](verify_slope_example.py) compares both fitted variances and the anchored ratio with the closed-form solution at each slope.

The [main-text patch](paper.patch) applies to m111 from the root of that checkout. The supplement matches m111. The [clean manuscript](KCOR-clean.pdf) contains the revised paragraph. The [round-5 redline](KCOR-redline.pdf) compares m114 with the reviewed m102 checkpoint. Its numbered superscripts link each manuscript edit to the objection at the back. The [m112 checkpoint](checkpoints/m112/) preserves the supplement derivation and the [m113 checkpoint](checkpoints/m113/) preserves the one-sentence version.

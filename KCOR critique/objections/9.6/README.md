# 9.6 - Finite contrast after cohort has no people at risk

**Severity:** High

**Origin:** New specific defect

**Related issues / root causes:** 9.5, 8.1, 7.1.

**Finding:** After complete nondeath exit, numerator has zero opening risk from April 4 onward yet July 4 anchored KCOR is 0.732363, status OK.

**Proposed scope (for discussion):** Mark unsupported intervals unavailable; preserve last supported estimate and distinguish subgroup exhaustion from whole-cohort exhaustion.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m147; m146 preserved. Independent closure not yet evaluated.

At zero opening risk, the ordinary fitting wrapper restricts input to the supported prefix and the normalization flags all subsequent weeks risk_set_exhausted. Explicit all-zero suffixes and omitted suffixes after verified complete exhaustion are handled identically; no missing positive-risk observation is filled. Direct all-ages cohorts with remaining people are not invalidated by an exhausted component, but a positive-weight pooled contributor is not silently dropped. The declared exit-mode 56-week reviewer fixture has a finite supported prefix and unavailable point/interval from April 4 onward in both proportional and forced-alpha modes. Numerator fitted variance and corrected hazard prefix agree with truncating that numerator at the support boundary. The written five-draw workbook produces prior intervals and zero valid draws after exhaustion (smoke test, not inference). Actual death-only exhaustion with both explicit/omitted zero suffix passes. Prior source-history and 24 solver cases pass. Original Table 8 minimum opening risks are 32,735 and 22,455; all 12 points reproduced within 2.23e-16; archived intervals retained, no full refit. Section 2.6.1 clarifies the support rule. No estimator change within supported intervals. [Workbook verification](workbook-verification.json).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).

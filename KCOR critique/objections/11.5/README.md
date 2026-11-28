# 11.5 - Alpha equals one is not identical to disabling the module

**Severity:** Moderate, method description

**Origin:** New specific wording discrepancy

**Related issues / root causes:** 1.2, 4.1, 4.2, 10.5.

**Finding:** Same exact proportional history, enabled marked wave alpha=1 versus module off: fitting masks 50 versus 60 rows, theta 2 versus 5.61408. Inversion identity with same parameters still holds.

**Proposed scope (for discussion):** Clarify that the inversion reduces to proportional inversion at fixed parameters; marked-week exclusion may change the fit.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** documented in m158; m157 preserved. Independent closure not yet evaluated.


Main Section 2.7.1 and SI S2.1 now distinguish the fixed-parameter proportional inversion identity from the fitting-mask change when marked waves overlap baseline fitting weeks. No method or result change.


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

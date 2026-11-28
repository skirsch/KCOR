# 12.5 - Historical synthetic negative control misattributed to current gamma estimator

**Severity:** High for validation claim

**Origin:** New specific evidence/provenance discrepancy

**Related issues / root causes:** 8.10, 9.3, 11.17.

**Finding:** Figure source fits separate four-parameter log-hazard relaxation curves on entire follow-up of discrete mixture, not gamma quiet-window fitting or inversion. Replay reproduces endpoint 0.997533. Current SI S9 validation is separate.

**Proposed scope (for discussion):** Label as historical relaxation estimator and remove attribution to revised method; regenerate through current fitter only if desired and explicitly approved.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** documented in m165; m164 preserved. Independent closure not yet evaluated.


Historical figure source and five-class mixture identified explicitly in main Section 3.1.1, settings table and SI S4.2.1/2; same figure retained but no longer attributed to the current estimator. Current S9 validation distinguished. No new experiment, numerical data or estimator change.


**Round 13 independent disposition:** resolved in original scope on m167. [Targeted review](../../iterative%20review/evaluations/13/resolution.md).

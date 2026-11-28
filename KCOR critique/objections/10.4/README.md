# 10.4 - Anchor and quiet union can create a nonexistent gap

**Severity:** Moderate

**Origin:** New specific defect

**Related issues / root causes:** 8.11, 2.3, 3.1, 9.7.

**Finding:** Identical 55 fitted rows declared all quiet vs quiet starting immediately after four anchors yield theta 4.184287 vs 4.181767 and a phantom offset 0.0000614. Re-entry built from quiet-only segments rather than union.

**Proposed scope (for discussion):** Keep seed and quiet-death eligibility unchanged; locate re-entries only after actual gaps in the modeled baseline mask.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/10/blind-a.md) | [Blind B](../../iterative%20review/evaluations/10/blind-b.md) | [Resolution](../../iterative%20review/evaluations/10/resolution.md)

**Status:** implemented in m155; m154 preserved. Independent closure not yet evaluated.


Regression: 55 sampled fitted weeks, quiet starts at 0, 1, 2, 3, 4 with four anchors; all fits agree within numerical tolerance (theta approximately 3.952477), no offset or signed gap diagnostic. Exact true theta 4, injected gap exposure 0.02: recovered 4 and 0.02 both for first gap after anchors and an internal gap. 10.3 regression also passed. Actual Czech configured re-entry dates for 2021_24 and 2022_06 unchanged (2022-22, 2023-19, 2024-11), so the correction does not alter the primary Table 8 or S6 estimators for those full histories. No new bootstrap rerun claimed; prior forced-alpha and other archived replay limitations remain.


**Round 11 independent disposition:** resolved in original scope on m156. [Targeted review](../../iterative%20review/evaluations/11/resolution.md).

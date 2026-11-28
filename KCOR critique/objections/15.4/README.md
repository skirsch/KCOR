# 15.4 - Worked S6 example omits the declared primary unanchored contrast

**Severity:** Moderate; material interpretation/reporting, not a wrong anchored estimate

**Origin:** New reporting concern

**Related issues / root causes:** 6.9, 13.6, 14.3.

**Finding:** S6 labels its anchored values correctly but does not present the defined primary unanchored contrast. Six displayed age-specific unanchored point estimates are below unity while six anchored points are above; all-age point is above in both. These are different estimands, not contradictory estimates. No unanchored intervals were computed in this review.

**Proposed scope (for discussion):** Discuss intended S6 presentation with author. If the primary ratio should be shown, add matched unanchored pointwise intervals from the actual per-draw hazards/fits; do not merely rescale anchored bounds. Retain anchored results.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/15/blind-a.md) | [Blind B](../../iterative%20review/evaluations/15/blind-b.md) | [Resolution](../../iterative%20review/evaluations/15/resolution.md)

**Status:** Dismissed as a requested change of estimand; original intent confirmed by user. Purpose sentence implemented in m177.


**User-confirmed original intent:** S6 is explicitly an illustration of anchored KCOR. No unanchored table or additional analysis is required. To prevent this design-choice objection returning, the approved sentence added to the S6.1 opening is: "The purpose of this worked example is to illustrate anchored KCOR." No numerical or algorithm change. [Patch and PDF verification](../../iterative%20review/evaluations/15/final-s6-wording/README.md).


**Round 16 independent disposition:** Closed by user-confirmed original intent; approved purpose statement verified. Blind A repeated the additional-table recommendation, but it is not a defect in the intended anchored illustration. [Resolution](../../iterative%20review/evaluations/16/resolution.md).

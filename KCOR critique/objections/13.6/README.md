# 13.6 - Broken reference erases valid earlier unanchored KCOR

**Severity:** Moderate for affected general workbook

**Origin:** New specific support-mask defect

**Related issues / root causes:** 9.1, 12.8, 6.2.

**Finding:** In complete fixed cohorts with positive risk until reference-week exhaustion, both fits OK. Earlier adjusted cumulative hazards finite (ratio about 1.069), but unanchored row unavailable because reference-wide unsupported mask applied to both estimands.

**Proposed scope (for discussion):** Use date-specific unanchored support, preserve all-date anchored reference requirement; regress point and matched intervals.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/13/blind-a.md) | [Blind B](../../iterative%20review/evaluations/13/blind-b.md) | [Resolution](../../iterative%20review/evaluations/13/resolution.md)

**Status:** implemented in m168; m167 preserved. Independent closure not yet evaluated.


Separate unanchored date support from reference-wide anchored mask; complete exhausted-reference fixture retains earlier unanchored point 1.191314 [1.028140,1.500353] (three matched draws, smoke only). Anchored unavailable throughout. [Batch patches and verification](../../iterative%20review/evaluations/13/post-review-batch-1/).


**Round 14 independent disposition:** resolved in original scope on m169. [Targeted review](../../iterative%20review/evaluations/14/resolution.md).

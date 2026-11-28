# 7.5 - Archived age-shift control and current restricted Czech CMR are different populations

**Severity:** High for empirical provenance

**Origin:** Returning 6.14, new proven source mismatch

**Related issues / root causes:** 6.14, 5.6, 6.8.

**Finding:** The historical input blob f2dbb3d7 is in September 2025 commit bb2d6824. Its historical CMR reproduces every one of the 2,169 grouped weekly count rows in each of two sheets. The full-history product filter appears in November 2025 commit f58d8a8f. The current S6 input is restricted; the older control is not. Current ?3.1.2 places the restriction disclosure beside the historical control, incorrectly implying the same eligibility.

**Proposed scope:** Distinguish the two input versions and populations in text, captions and Table 8. Retain the original historical control unless the lead author deliberately chooses a replacement. The different-input point replay is not an isolated product-rule effect and has no replacement CIs.

**Review evidence:** [Resolution](../../iterative%20review/evaluations/7/resolution.md)

**Status:** addressed in m131 on top of completed m130; preview superseded.


**Approved correction:** identify the two archived age-shift figures and Table 8 as September 2025 pre-restriction input; scope the existing product restriction and audit to the separate current Czech S6 application. The existing product definition, mortality data, results, figures and uncertainty intervals are preserved. This is a reporting/provenance change, not a change to population eligibility or general KCOR.

The resolution reviewer independently identified the source CMR Git blob 38e9b13ef413bed60356fe5c286575c6be03478f in bb2d6824 and reproduced both age-shift input sheets (2169 grouped rows per sheet), maximum count/death discrepancy zero. The product filter first appears in f58d8a8f. The m129-based preview checks layout and wording while the separate full 7.3 computations run; it must not replace the final m130-derived scientific edition.

[Paper patch](paper.patch) | [Supplement patch](supplement.patch) | [Composable manuscript edit](apply_7_5.py) | [Cumulative redline update](redline_7_5.py).

**Final integration:** implemented in m131 on top of completed m130, with prior preview superseded. No estimator or data changes.

**Round 8 independent disposition:** resolved within original scope; see [resolution review](../../iterative%20review/evaluations/8/resolution.md).

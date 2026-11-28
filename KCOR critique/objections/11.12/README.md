# 11.12 - Table 8 and S6 references identify superseded archives

**Severity:** Low to moderate for reproducibility

**Origin:** New provenance discrepancy after m154

**Related issues / root causes:** 10.3, 10.8, 9.10.

**Finding:** Table 8 caption still cites older primary archive and calls forced-alpha current; SI S6 and analysis README also refer to older primary paths. Printed values unchanged; full forced-alpha intervals remain prior-seed.

**Proposed scope (for discussion):** Correct source paths, archive labels and reproduction output path without claiming new forced-alpha refits.

**Review evidence:** A, R. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Main and SI sources updated; primary 999-draw refresh matches all old points and draws to 2.22e-16; prior-seed forced-alpha full intervals explicitly archived, fresh 3-draw smoke separate. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

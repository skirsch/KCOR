# 11.8 - Hyphenated enrollment sheet name disables bootstrap

**Severity:** Moderate

**Origin:** New specific defect

**Related issues / root causes:** 9.1, 10.6.

**Finding:** Point parser accepts 2023-19 and 2023_19, but bootstrap split on underscore only makes hyphenated enrollment intervals unavailable.

**Proposed scope (for discussion):** Share enrollment-date parser and test identical outputs under both names.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Point and bootstrap use same enrollment parser; hyphen and underscore produce identical person-history cells. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

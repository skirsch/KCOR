# 7.10 - Historical alpha diagnostics named as current

**Severity:** Moderate for optional alpha reporting

**Origin:** Returning 6.13, remaining wording

**Related issues / root causes:** 6.13, 5.10, 4.7.

**Finding:** The current ?3.4.2 correctly reports the revised objective's lower-boundary optima. The conclusion still refers to a preferred pooled alpha region and a caption labels historical diagnostics primary, which can imply current localization.

**Proposed scope:** Date the historical diagnostics and align the conclusion with the current non-identification result. Full weekly alpha estimation remains deferred; no supplied-alpha inversion change.

**Review evidence:** [Resolution](../../iterative%20review/evaluations/7/resolution.md)

**Status:** addressed in m129, preserving m128 (approved batch of straightforward corrections).


The conclusion now explicitly distinguishes the historical preferred region from the revised boundary optimum and non-identification. The existing archived figure is labeled as an archive, and its caption describes the actual plotted lower-boundary leave-one-out and segmented estimates and boundary-piled bootstrap rather than attributing unplotted variation to it. No changes to alpha identification, numerical inversion, or sensitivity values.

[Combined batch code patch](../../iterative%20review/evaluations/7/post-review-batch-1/workbook.patch) | [Paper patch](../../iterative%20review/evaluations/7/post-review-batch-1/paper.patch) | [Tests](../../iterative%20review/evaluations/7/post-review-batch-1/tests.patch)

**Round 8 independent disposition:** resolved within original scope; see [resolution review](../../iterative%20review/evaluations/8/resolution.md).

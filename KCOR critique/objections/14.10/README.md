# 14.10 - Historical contamination default rerun labels and bars differ from published figure

**Severity:** Minor reproducibility

**Origin:** New related reporting discrepancy beyond 13.8

**Related issues / root causes:** 13.8, 12.3, 2.7.

**Finding:** Published archived grid plots reproduce pixels exactly. Default simulator emits SD bars rather than prescribed mean CI and console calls endpoint asymptote; last eligible week may precede final scheduled week.

**Proposed scope (for discussion):** Clarify last eligible endpoint, console labels, direct archive plotting entry and correct or explicitly distinguish default uncertainty bars.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m170; targeted regression or artifact verification completed. Independent round review pending.


**Approved correction and verification:** Default contamination sweep calculates pointwise Student-t 95% Monte Carlo mean intervals, not SD bars, and calls the statistic endpoint. The paper states the last eligible week of 130. An explicit historical-grid plotting CLI reproduces both published PNGs byte-for-byte; legacy file and CSV names retained.

[Round 14 post-review batch](../../iterative%20review/evaluations/14/post-review-revisions.md).


**Round 15 independent disposition:** Resolved in original scope by the independent round-15 review. [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

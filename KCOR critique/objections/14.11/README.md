# 14.11 - Distributed TeX remains stale relative to current PDF and Markdown

**Severity:** Minor artifact provenance

**Origin:** New specific reproducibility discrepancy

**Related issues / root causes:** 11.12, 13.7, 13.8.

**Finding:** Canonical generated paper.tex is unchanged from m167; published review PDF and Markdown correct. TeX still omits benefit audit caveat and calls endpoint asymptote.

**Proposed scope (for discussion):** Regenerate distributed TeX from current manuscript sources or explicitly label historical artifact.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m170; targeted regression or artifact verification completed. Independent round review pending.


**Approved correction and verification:** Canonical paper.tex regenerated from the current combined Markdown sources using the manuscript toolchain, including the benefit audit caveat and finite-endpoint text. Clean and redline PDFs regenerated.

[Round 14 post-review batch](../../iterative%20review/evaluations/14/post-review-revisions.md).


**Round 15 independent disposition:** Resolved in original scope by the independent round-15 review. [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

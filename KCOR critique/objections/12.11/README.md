# 12.11 - Historical S7 mixture mass and initial risk disagree

**Severity:** Minor historical provenance; numerical risk accounting if reused

**Origin:** New specific defect in historical generator

**Related issues / root causes:** 11.17, 6.8, 9.1.

**Finding:** Truncated quantile weights sum to .998 but initial opening risk uses nominal 2 million: 4,000 unexplained people in first-week accounting. Two skipped weeks exclude this denominator from fitted hazards.

**Proposed scope (for discussion):** Document archive limitation and use consistent mixture mass/initial risk for any new run; do not rewrite historical outputs.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** implemented in m162; m161 preserved. Independent closure not yet evaluated.


Historical S7 first-week nominal population mismatch and two-week skip explicitly documented; original code and archive unchanged; any future rerun must use consistent initial risk and mixture mass. [Batch patches and PDFs](../../iterative%20review/evaluations/12/post-review-batch-1/).


**Round 13 independent disposition:** resolved in original scope on m167. [Targeted review](../../iterative%20review/evaluations/13/resolution.md).

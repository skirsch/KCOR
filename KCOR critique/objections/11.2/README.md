# 11.2 - Dedicated Table 8 replay invents finite ratios after risk-set exhaustion

**Severity:** High for affected replay histories

**Origin:** Related defect beyond 10.1; not a failure of that reader correction

**Related issues / root causes:** 10.1, 9.1, 8.1.

**Finding:** Extinguishing a required dose-2 subgroup at week 29 yielded dedicated pooled ratios 73.342353 and 72.173637; the ordinary workbook correctly reported unavailable. Published histories unaffected.

**Proposed scope (for discussion):** Use the ordinary endpoint/support rules and check the actual pooled workbook row.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/11/blind-a.md) | [Blind B](../../iterative%20review/evaluations/11/blind-b.md) | [Resolution](../../iterative%20review/evaluations/11/resolution.md)

**Status:** implemented in m157; m156 preserved. Independent closure not yet evaluated.


Dedicated replay no longer clips 100% mortality into a finite increment; exhausted 2/0 and 2/1 unavailable, unaffected comparison still 1.016193. [Batch patches, verification and PDFs](../../iterative%20review/evaluations/11/post-review-batch-1/).


**Round 12 independent disposition:** resolved in original scope on m161. [Targeted review](../../iterative%20review/evaluations/12/resolution.md).

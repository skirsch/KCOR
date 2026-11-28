# 15.1 - Historical manufacturer workbooks still distribute superseded outputs

**Severity:** High for ancillary distributed sheets; no indication of primary paper impact

**Origin:** Direct delivery carryover of 14.1 and 14.2

**Related issues / root causes:** 14.1, 14.2.

**Finding:** Corrected producer and converter pass fresh tests, but released KCOR.xlsx manufacturer sheets still carry pre-enrollment unadjusted ratios under KCOR labels, and historical input manufacturer partition fails reconciliation.

**Proposed scope (for discussion):** Preserve original archived inputs/results, clearly mark obsolete manufacturer tabs where encountered or replace active tabs with unavailable status until full verified regeneration. No change to main paper.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/15/blind-a.md) | [Blind B](../../iterative%20review/evaluations/15/blind-b.md) | [Resolution](../../iterative%20review/evaluations/15/resolution.md)

**Status:** Implemented in m174; targeted artifact verification complete.


**Approved correction:** The six obsolete manufacturer input/output tabs are now explicit unavailable notices with no current values. The full original data and result workbooks remain exactly preserved in m173. All other workbook package parts are byte-identical; the corrected pipeline rejects the obsolete notices as input. No paper or method changes, and no full manufacturer regeneration. [Evidence and archive hashes](../../iterative%20review/evaluations/15/manufacturer-archive/README.md).


**Round 16 independent disposition:** Resolved in the original scope by independent targeted round-16 review. [Resolution](../../iterative%20review/evaluations/16/resolution.md).

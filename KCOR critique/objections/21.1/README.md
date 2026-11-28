# 21.1 - Ordinary standalone summary omits residual and gap diagnostics

**Severity:** Moderate for interpretation of standalone workbook and console

**Origin:** Newly verified specific finding in round 21; not a demonstrated regression from round 20.

**Related issues / root causes:** 20.1, 19.1, 11.4, 15.5.

**Finding:** Written ordinary workbook has finite KCOR with both theta fits OK, residual_screen_flag=True and gap_assumption_status=not_assessed. Standalone KCOR_summary.xlsx and console present it without either qualification; summary also loses the bootstrap residual counts. Main dose_pairs retains all.

**Proposed scope (for discussion):** Propagate point residual flags/unavailable, draw flag/unavailable counts and gap status into standalone rows and console; keep numerical fit status separate and preserve points and intervals.

**Review evidence:** [blind-b.md](../../iterative%20review/evaluations/21/blind-b.md)

**Status:** Open; correction not adopted.

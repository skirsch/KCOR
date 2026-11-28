# 16.5 - Window-omission ancillary CSV contains superseded full-fit intervals

**Severity:** Minor artifact provenance

**Origin:** New specific archive inconsistency

**Related issues / root causes:** 14.11, 11.12.

**Finding:** S6 window-omission CSV reads older summary for full-fit bounds; age 40-49 old [0.823855,2.711896] differs from current reported [0.960387,2.821651]. Omission ranges are point sensitivity and the paper/main interval archive are correct.

**Proposed scope (for discussion):** Use the current archived interval source for the full-fit columns or remove/clearly label stale ancillary bounds; keep point-sensitivity values unchanged.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/16/blind-a.md) | [Blind B](../../iterative%20review/evaluations/16/blind-b.md) | [Resolution](../../iterative%20review/evaluations/16/resolution.md)

**Status:** Implemented in m178; targeted verification complete.


**Approved correction:** All seven stale interval pairs updated from canonical current S6 archive, generator path corrected; all other values preserved. [Verification](../../iterative%20review/evaluations/16/straightforward-batch/README.md).

**Round 17 independent disposition:** Resolved in original scope. All seven ancillary interval pairs match current archive. [Verification](../../iterative%20review/evaluations/17/resolution.md).

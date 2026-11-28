# 21.3 - Ordinary output can retain a contradictory ISO week label

**Severity:** Low calendar reporting defect

**Origin:** Newly verified specific finding in round 21; not a demonstrated regression from round 20.

**Related issues / root causes:** 20.2, 14.5.

**Finding:** Both reviewers identify ordinary date-derived fitting with first-value supplied ISO label retained in output. A shift of eight weeks changes both displayed ISO fields but leaves the numeric result and dates unchanged.

**Proposed scope (for discussion):** Use canonical date-derived ISO labels in ordinary outputs and disclose contradictory source metadata. Preserve correct-date fit and ordinary bootstrap behavior.

**Review evidence:** [blind-a.md](../../iterative%20review/evaluations/21/blind-a.md) | [blind-b.md](../../iterative%20review/evaluations/21/blind-b.md)

**Status:** Open; correction not adopted.

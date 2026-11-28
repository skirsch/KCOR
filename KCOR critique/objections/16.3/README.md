# 16.3 - Nested-total validator rejects valid fractional counts from floating-point roundoff

**Severity:** Low numerical edge case; real point availability regression

**Origin:** New regression from 15.3

**Related issues / root causes:** 15.3, 14.4.

**Finding:** Dividing a valid nested integer fixture by ten gives fractional expected counts accepted by the source-history validator. Complement residual is 2.27e-13 person; exact loss!=0 now withholds both valid points. Actual Czech integer counts unaffected.

**Proposed scope (for discussion):** Keep exact integer checks and use scale-aware floating-point tolerance in fractional count complement checks, consistent with the existing point-history policy. Integer bootstrap unchanged.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/16/blind-a.md) | [Blind B](../../iterative%20review/evaluations/16/blind-b.md) | [Resolution](../../iterative%20review/evaluations/16/resolution.md)

**Status:** Implemented in m178; targeted verification complete.


**Approved correction:** Fractional expected-count nested complements use original-operand-scale machine roundoff tolerance; exact integer checks retained. Reviewer fixture restored and material discrepancy rejected. [Verification](../../iterative%20review/evaluations/16/straightforward-batch/README.md).

**Round 17 independent disposition:** Resolved in original scope. Nested complement fractional reconciliation verified; separate exhaustive sum issue recorded as 17.2. [Verification](../../iterative%20review/evaluations/17/resolution.md).

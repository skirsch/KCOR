# 17.1 - A supplied Monte Carlo total suppresses another replicate?s direct all-ages output

**Severity:** Moderate for mixed batched Monte Carlo inputs

**Origin:** Newly identified, pre-existing defect; not a direct return of the previous objection.

**Related issues / root causes:** 16.2, 15.2, 14.4.

**Finding:** A simulated replicate produces 20 direct all-ages rows alone and with another age-only replicate. Adding exact supplied totals solely to the other replicate removes all 20 rows from the first replicate in both m177 and m180, including the written worksheet. The age-standardized output remains unchanged. This can exclude valid realizations from the all-ages MC summary.

**Proposed scope (for discussion):** Make total-source selection and reconstruction independently by replicate and dose coverage; verify standalone/batch/order equivalence for direct all-ages rows as well as standardized output.

**Review evidence:** [Round 17 report](../../iterative%20review/evaluations/17/resolution.md)

**Status:** Implemented and verified in persistent m181; independent review pending.

**Approved correction:** Implemented and verified. [Evidence](../../iterative%20review/evaluations/17/mc-totals/README.md).

**Round 18 independent disposition:** Original cross-replicate omission resolved; new one-dose reconstruction regression remains (18.1). [Verification](../../iterative%20review/evaluations/18/resolution.md).

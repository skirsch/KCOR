# 16.2 - Monte Carlo pooling weights depend on other replicates in the same workbook

**Severity:** Moderate for the batched Monte Carlo branch

**Origin:** New implementation defect

**Related issues / root causes:** 10.8, 2.9.

**Finding:** A fixed simulated replicate has a different pooled endpoint when another replicate is appended to its workbook. Sheet-wide weights are computed across mc_id before per-replicate pooling. Ordinary production person bootstrap is distinct; primary Table 8 and S6 archives not implicated.

**Proposed scope (for discussion):** Compute weights independently by replicate, propagate them through pooling/status summaries, and test order and split/batch invariance; preserve intentionally fixed external weights if specified.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/16/blind-a.md) | [Blind B](../../iterative%20review/evaluations/16/blind-b.md) | [Resolution](../../iterative%20review/evaluations/16/resolution.md)

**Status:** Implemented in m180; targeted MC workbook and ordinary-parity verification completed.


**Approved correction:** Calculate and cache existing pooling weights by independent MC replicate, use the matching replicate calendar and weight lookup for both pooling/status paths. Exact standalone/batch/order invariance verified, including written workbook. Existing paper and primary results unchanged. [Verification](../../iterative%20review/evaluations/16/mc-pooling/README.md).

**Round 17 independent disposition:** Resolved in original scope. Replicate-specific pooled weights and calendar verified; separate mixed-total availability issue recorded as 17.1. [Verification](../../iterative%20review/evaluations/17/resolution.md).

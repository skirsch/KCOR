# 18.4 - Partial Monte Carlo aggregates report incorrect full-age labels and no coverage

**Severity:** Moderate for interpretation of partial MC outputs

**Origin:** Newly verified current defect; regression status not established.

**Related issues / root causes:** 16.1, 17.1, 18.1.

**Finding:** In a partial mixed MC workbook, reconstructed numerator has only 1940, denominator has 1940 and 1950. It is labeled All Ages, indistinguishable from the other replicate with a supplied total. Per-dose coverage flags are blank.

**Proposed scope (for discussion):** Reuse ordinary available-age labels and population coverage reporting per replicate and dose. Do not require equal age composition, but accurately identify source support.

**Review evidence:** [blind-b.md](../../iterative%20review/evaluations/18/blind-b.md)

**Status:** Implemented and verified in persistent m182; independent review pending.

**Approved correction:** Addressed together with 18.1-18.4 using coherent per-replicate full-source preparation, validation, raw comparator and reporting. [Verification](../../iterative%20review/evaluations/18/mc-source-consistency/README.md).

**Round 19 independent disposition:** Original scope resolved according to targeted reviewer progress before usage-limit interruption. [Recovered progress](../../iterative%20review/evaluations/19/resolution-progress.md).

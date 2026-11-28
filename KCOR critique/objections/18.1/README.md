# 18.1 - Monte Carlo display filtering mixes population coverage in direct all-ages comparisons

**Severity:** High for affected MC workbook results; no demonstrated main-result impact

**Origin:** Regression from expanded 17.1 one-dose reconstruction.

**Related issues / root causes:** 17.1, 16.1, 16.2.

**Finding:** With identical dose histories for three birth years, exact supplied totals for both doses or neither give 1. Supplying only the denominator total gives 0.489556; supplying only the numerator total gives 2.042668, both marked ok. The display filter removed a source age from the reconstructed side but not from the supplied side. The same-range packaged generator avoids this case. Wider external inputs or rereading with a narrower range can trigger it. These are deterministic fixture points, not interval estimates.

**Proposed scope (for discussion):** Establish the aggregate population from complete raw sources before the display filter, or consistently restrict and label both sides; withhold incompatible populations. Test all four supplied-total layouts, including an age outside the display range.

**Review evidence:** [resolution.md](../../iterative%20review/evaluations/18/resolution.md) | [blind-a.md](../../iterative%20review/evaluations/18/blind-a.md)

**Status:** Implemented and verified in persistent m182; independent review pending.

**Approved correction:** Addressed together with 18.1-18.4 using coherent per-replicate full-source preparation, validation, raw comparator and reporting. [Verification](../../iterative%20review/evaluations/18/mc-source-consistency/README.md).

**Round 19 independent disposition:** Original scope resolved according to targeted reviewer progress before usage-limit interruption. [Recovered progress](../../iterative%20review/evaluations/19/resolution-progress.md).

# 14.1 - Manufacturer workbook reports pre-enrollment unadjusted ratio as KCOR

**Severity:** High for ancillary manufacturer sheets

**Origin:** New specific ancillary implementation defect

**Related issues / root causes:** 9.1, 12.8, 13.6.

**Finding:** Manufacturer path includes 67 pre-enrollment weeks, no frailty inversion, undefined anchor fallback. Released raw labeled result 1.222234 vs corrected dates/anchor-only raw 0.920011; neither establishes corrected KCOR. Main Table 8/S6 separate.

**Proposed scope (for discussion):** Verify original manufacturer output intent, use validated fixed-cohort KCOR path if intended as KCOR; otherwise label raw comparator with correct dates and no undefined-anchor fallback.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/14/blind-a.md) | [Blind B](../../iterative%20review/evaluations/14/blind-b.md) | [Resolution](../../iterative%20review/evaluations/14/resolution.md)

**Status:** Implemented in m171; targeted regression and converter verification completed. Historical full manufacturer inputs require regeneration; independent review pending.


**Approved correction:** Original intent verified in Git commits 4fe94be8 and 7be6f1b6 (October 21, 2025): explicitly labeled M/P KCOR by dose and decade. Replaced duplicate raw-hazard implementations with a shared ordinary KCOR calculation, using the actual enrollment date, usual stabilization skip, all-window frailty fit, strict anchoring, and the ordinary person bootstrap. Workbook and log use the same results. The old incompatible source is reported unavailable rather than producing a misleading number.

[Manufacturer correction and verification](../../iterative%20review/evaluations/14/manufacturer-batch/README.md).


**Round 15 independent disposition:** Partial at delivery: corrected code independently verified, but distributed historical manufacturer artifacts remain superseded. See [15.1](../15.1/README.md). [Resolution review](../../iterative%20review/evaluations/15/resolution.md).

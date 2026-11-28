# 9.2 - Sensitivity selection changes calendar horizon

**Severity:** High

**Origin:** Direct return; direct predecessor 8.3

**Related issues / root causes:** 7.2, 8.2.

**Finding:** With the supplied total missing the last week, ordinary output is missing_week_history while sensitivity reports 0.999525 OK. All 1,323 complete-data cells verified.

**Proposed scope (for discussion):** Preserve source calendar and validation outcomes before selecting population.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m145; m144 preserved. Independent closure not yet evaluated.

The sweep passes the complete sheet to ordinary preprocessing, retaining the identical calendar and preaggregation source-history checks. After those checks, only when an explicit inclusive source is available, it avoids fitting unrelated ages; reconstructed all ages retain their source components. The same ordinary estimator and reanchoring remain. The missing-final-total fixture is unavailable in both modes at all three tested reference lengths; all complete-data parity tests passed. The complete Czech grid was freshly replayed: all 1,323 cells, fit-status fields and figure identical to m144 (maximum difference zero; PNG byte identical). No change to the paper or existing PDF is necessary. [Verification](verification.json), [fresh cells](KCOR_SA_cells.csv).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).

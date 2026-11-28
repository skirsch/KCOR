# 12.7 - Undefined anchor differs between dedicated replay and ordinary pooling

**Severity:** Moderate for affected replay histories

**Origin:** Related pre-existing discrepancy, distinct from 11.2 and 11.6

**Related issues / root causes:** 11.2, 11.6, 9.8.

**Finding:** Moving first six weeks of 1940 dose-2 deaths to survivors yields ordinary available-strata points 1.209732 and 1.219862 with retained weight 0.704731, but dedicated NaNs. Previous exhaustion and sparse-weight fixtures remain fixed.

**Proposed scope (for discussion):** Share ordinary contributor, anchor and failure rules; preserve point composition through matched draws.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** implemented in m166; m165 preserved. Independent closure not yet evaluated.


Paper-specific replay only: available-strata pooling when an anchor is undefined, unavailable result on required risk-set/normalization failure, and per-draw contributor masks with no interval if any required draw changes the point composition. Undefined-anchor test reproduces ordinary 2/0=1.209732 and 2/1=1.219862 with 1950 and 1960 contributing. Exhaustion still produces unavailable affected contrasts. One anchor death test: 3 of 8 draws change composition for both affected pairs, so intervals are unavailable; unaffected pair has all 8 valid draws. Primary 999-draw replays retain all subgroups and match previous points, draws, parameters, flags and gap CSV exactly. Ordinary code and manuscript byte-identical; paper PDFs remain valid.

Fresh forced-alpha three-draw smoke also passed; all contributor masks stable. Full forced-alpha intervals remain archived, not newly refit.


**Round 13 independent disposition:** resolved in original scope on m167. [Targeted review](../../iterative%20review/evaluations/13/resolution.md).

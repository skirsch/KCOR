# 9.8 - Pool silently drops positively weighted ages across draws

**Severity:** Moderate to high

**Origin:** New specific defect

**Related issues / root causes:** 2.9, 4.6, 5.3, 8.2.

**Finding:** Both blind reviews reproduce sparse age anchor undefined while pooled point 1 and normalization OK, using only the remaining age. A bootstrap can silently change contributors.

**Proposed scope (for discussion):** Record contributor identities and retained weights; decide fixed or available-strata estimand from original intent; flag incomplete fixed-pool draws.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m149; m148 preserved. Independent closure not yet evaluated.

The original available-strata weighted-geometric point and recalculation of weights are retained. Each pool now records eligible and actual positive-weight ages, retained weight fraction and complete/partial/unavailable status. A bootstrap draw with a different contributor set from the point is counted explicitly, flagged and does not supply a valid fixed-composition replicate; the interval is unavailable if any required draw changes membership. Missing rows remain separately counted. Existing gap-assumption flags now use the actual contributors. In the partial-point fixture, 1940 alone contributes, the point is 1, retained weight 0.499934. In the one-anchor-death fixture, 7/19 draws change composition, 12 remain valid and the interval is withheld. The supported fixture retains all 19 draws and interval 1.000 [0.929, 1.161] (smoke test only). All 2,000 Table 8 point/draw histories have positive anchor-period deaths in all three subgroups (minimum 281 and 335 by enrollment); the dedicated replay averages all three with no silent omission. All 12 Table 8 ordinary points reproduced to 2.23e-16, existing intervals retained. Prior gap diagnostic regression still passes. Section 2.8 updated. [Regression](regression.json) | [Workbook verification](workbook-verification.json).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).

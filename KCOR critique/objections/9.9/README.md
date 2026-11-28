# 9.9 - Legacy analytical variance uses wrong anchored contrast

**Severity:** Moderate

**Origin:** New specific defect, related older uncertainty work

**Related issues / root causes:** 1.3, 4.6, 4.8.

**Finding:** Both derive variance terms Var(J)/(A+J)^2 and [1/(A+J)-1/A]^2 Var(A), while old code divides by J^2. Bootstrap availability is unnecessarily gated by legacy bounds.

**Proposed scope (for discussion):** Retire legacy inferential intervals or correct actual contrast; independently gate refitted bootstrap by its own valid point and draws.

**Review evidence:** A, B. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m150; m149 preserved. Independent closure not yet evaluated.

The legacy count-based formula is removed from the four reporting branches and old analytic output columns; the bootstrap availability gate no longer refers to it. Point-only and disabled-bootstrap output leaves interval placeholders blank. The unanchored interval has its own method and valid-draw count. In a written 19-draw workbook all four supported age/direct/pooled reference points give exactly 1 [1,1]. With an undefined anchor the anchored point and interval are unavailable, while the unanchored value is 0.893 [0.779,0.951]; these are software smoke tests, not inferential claims. Current Table 8 points reproduced to 2.23e-16 and existing refitted percentile intervals retained. Historical Table 8 values and originally reported bounds are preserved for archival comparison only, not labeled as valid current confidence intervals. Section 2.9.1 and Table 8 header/caption updated. No estimator or resampling change. [Regression](regression.json) | [Table 8 verification](workbook-verification.json).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).

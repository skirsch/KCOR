# 12.8 - Tiny positive references accepted by pool but rejected by direct row

**Severity:** Moderate numerical edge case

**Origin:** Newly introduced inconsistency from 11.3

**Related issues / root causes:** 11.3, 9.8.

**Finding:** Valid one-stratum stress input with 10^13 people, one/two early deaths each week, pooled point 0.509146 but sole contributor and all-ages direct reference unavailable due 1e-12 threshold.

**Proposed scope (for discussion):** Unify direct and pooled reference validity and numerical range checks; test both small and large ratios.

**Review evidence:** R. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** implemented in m167; m166 preserved. Independent closure not yet evaluated.


Shared finite-ratio helper eliminates arbitrary near-zero threshold and returns unavailable for underflow or overflow of positive ratios, while preserving a genuine zero numerator. Direct, pooled and all-ages ordinary small-reference control all match at 0.5666666667; large-ratio control all match at 21082653773.60724. These are deterministic numerical stress fixtures. Earlier Table 8 missing-anchor composition and round-11 batch tests pass. Primary Table 8 full 999-draw replay: maximum point/draw difference 4.44e-16; all displayed values and intervals unchanged, parameters, diagnostic arrays, contributor masks and gap CSV identical. Full forced-alpha intervals remain prior-seed archive.

Ordinary workbook verification reproduces all six current primary point estimates exactly, and all six earlier forced-alpha smoke points within 2.22e-16. This is point verification, not a full forced-alpha interval replay.


**Round 13 independent disposition:** resolved in original scope on m167. [Targeted review](../../iterative%20review/evaluations/13/resolution.md).

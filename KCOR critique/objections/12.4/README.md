# 12.4 - Protective first gap falsely localizes deficit to later ordinary gap

**Severity:** High for per-gap assumption diagnostic

**Origin:** New specific defect related to gap diagnostics

**Related issues / root causes:** 10.3, 11.18, 9.7, 3.1.

**Finding:** Exact signed offsets -0.046924 and 0 returned -0.101682 and -0.113628; second gap genuinely follows baseline. 50 matched person draws yield spurious second-gap interval [-0.12384,-0.10697]. Entry-rate seed clamps negative log multiplier before diagnostic baseline calculation.

**Proposed scope (for discussion):** Use signed entry-rate identity for diagnostic baseline only, retaining constrained fitting objective; mark out-of-bounds diagnostic seed unassessed, regress exact two-gap history and refresh relevant gap screens.

**Review evidence:** A. [Blind A](../../iterative%20review/evaluations/12/blind-a.md) | [Blind B](../../iterative%20review/evaluations/12/blind-b.md) | [Resolution](../../iterative%20review/evaluations/12/resolution.md)

**Status:** implemented in m164; m163 preserved. Independent closure not yet evaluated.


Exact protective-gap regression: first offset -0.046923664 [ -0.047966314, -0.045819821 ], second offset approximately zero [ -0.000691085, 0.000655101 ] (50 matched person draws). Second gap no longer falsely flagged; first remains violation. Seed-ceiling regressions pass. Primary Table 8 999-draw replay: point/draw arrays, fitted parameters and ten diagnostic flag arrays exactly identical to m163; all 54 gap status classifications unchanged (30 no deficit, 24 unassessed). Three assessed limits change for 2022-06 birth group 1960 dose 0; fresh signed gap CSV archived. No change to constrained fitting objective or primary Table 8/S6 intervals. Forced-alpha full intervals remain prior-seed archive.


**Round 13 independent disposition:** resolved in original scope on m167. [Targeted review](../../iterative%20review/evaluations/13/resolution.md).

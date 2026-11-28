# 9.10 - Table 8 generic flag counts omit residual flags

**Severity:** Moderate

**Origin:** New specific reporting defect

**Related issues / root causes:** 8.9, 4.6, 7.6.

**Finding:** Existing identical inclusive-group fits imply at least 987/999, 950/999 and 973/999 contributing residual-flagged draws for 2021 module-off pairs. Generic flagged_draws counts only insufficient signal.

**Proposed scope (for discussion):** Label actual count precisely and report available contributor-specific diagnostic counts; do not discard draws or change percentile arithmetic.

**Review evidence:** B. [Blind A](../../iterative%20review/evaluations/9/blind-a.md) | [Blind B](../../iterative%20review/evaluations/9/blind-b.md) | [Resolution](../../iterative%20review/evaluations/9/resolution.md)

**Status:** implemented in m151; m150 preserved. Independent closure not yet evaluated.

The current matched Table 8 runner passes opening risk counts into the production residual screen, records ten distinct diagnostic categories at the subgroup and comparison levels, and reports the original point flag and number of bootstrap draws per category. The generic legacy field is replaced in new outputs by explicit insufficient-signal and unsuccessful-fit counts. The primary full refit (2,000 point/draw histories) agrees exactly with the m148 current points and CI endpoints; the forced-alpha diagnostic is evaluated at the exact archived parameters and death histories with the same production survivor-profile and sampling screen, verified against 300 fresh refits (every parameter, point, subgroup and comparison flag identical). No draws discarded. The primary residual screen flags every comparison in 999/999 draws; forced-alpha 2021 pairs 999/999, 2022 pairs 997/999, 986/999, 986/999. Raw gap-clamp counts are separate. All remaining eight categories are zero for every comparison and point. The previously reviewed inclusive-only 2021 primary counts 987,950,973 are independently reproduced. These residual flags indicate model discrepancy, not numerical failure or proof of bias direction. Main Section 3.1.2 and SI S10 report results, and the exact category matrix is available below. The new checkpoint schema passed the resume regression. No estimator, point or percentile changes. [All counts](diagnostics/comparison-diagnostics.csv) | [Verification](diagnostics/verification.json).

**Round 10 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/10/resolution.md).

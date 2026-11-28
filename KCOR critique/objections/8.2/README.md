# 8.2 - Weighted pooled workbook series exports only its final date

**Severity:** High for ordinary output

**Origin:** New specific defect

**Related issues / root causes:** 2.9, 4.6, 7.4: production reporting and pooling

**Finding:** Blind A and the resolution reviewer independently reproduced 174 dates per stratum but only one weighted pooled date in the historical Table 8 input. The pooled row is emitted outside the date loop; the archived Table 8 point was reconstructed separately rather than taken from the workbook. The resolution reviewer also verified that the intended 2022-12-26 pooled output is absent and that the summary code permits nearest-date substitution.

**Proposed scope (for discussion):** Emit each date within the loop and verify the actual exported Table 8 reporting-date row against the archive; add full trajectory regression, no change to pooling definition.

**Review evidence:** A, resolution. [Blind A](../../iterative%20review/evaluations/8/blind-a.md) | [Blind B](../../iterative%20review/evaluations/8/blind-b.md) | [Resolution](../../iterative%20review/evaluations/8/resolution.md)

**Status:** implemented in m136 (approved straightforward-correction batch; m135 preserved). Independent closure not yet evaluated.

Corrected per-date pooled export and removed nearest-date substitution. Undefined date rows remain explicitly unavailable. Actual exported Table 8 rows for both enrollments, both modes and all 3 pairs reproduce the archived points within 2.22e-16. 174 and 140 dates per pair are now exported, respectively. Archived intervals unchanged; no full resampling rerun needed because fit objective and dedicated Table 8 calculation are unchanged. The missing-source-component policy is tracked separately as 8.1. [Verification](workbook-verification.json).

[Batch manuscript and code patches](../../iterative%20review/evaluations/8/post-review-batch-1/).

**Round 9 independent disposition:** resolved within stated scope. [Resolution](../../iterative%20review/evaluations/9/resolution.md).

# 5.5 - Round 5 finding

**Severity:** Moderate/high for inference

Ordinary bootstrap holds birth-year/dose sizes fixed, while paper describes pooled resampling with variable sizes; Table 8 actually uses pooled component sampling. Both designs may be defensible; establish intended target. Nondeath-exit term missing in prose.

**Related issues:** 1.3, 4.6, uncertainty specification

**Review evidence:** A B - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** corrected in m107; m106 preserved.

The ordinary resampling conditions on the observed enrollment counts of each birth-year/dose group. Sex and DCCI composition can vary within that group. This is a conditional inference design; it does not incorporate random cohort-size variation. Table 8 retains the existing pooled sampling of disjoint components with a fixed total and varying component sizes. The manuscript and supplement now describe the actual methods separately. The risk-set equation subtracts both deaths and nondeath exits, with death-only histories as a special case.

The ordinary sampler and KCOR estimator are unchanged byte-for-byte. [Verification](verification.json) checks ten ordinary draws preserve all age/dose sizes, ten exit-bearing draws obey risk-set identity, and ten Table 8 draws preserve the total while the disjoint component sizes vary. Existing point and interval outputs are unchanged; no new full bootstrap run needed for a documentation-only correction.

[Paper patch](paper.patch), [supplement patch](supplement.patch), [verification script](verify.py). The actual draft PR is unchanged.

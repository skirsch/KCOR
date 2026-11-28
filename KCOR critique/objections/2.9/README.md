# 2.9 Subgroup fitting and pooling

Evaluation finding N9 from the [November 21, 2026 review](<../../iterative review/evaluations/2/README.md>).

Corrected in iteration 2. Section 2.9.1 distinguishes the primary fitted comparison groups from strata retained for resampling or standardization, and describes the subgroup-by-pseudo-dose fits and pooling used for Table 8. A fresh LLM evaluation is pending.

§2.9.1, Table 8 and SI §S6; [follow-up](<../../iterative review/evaluations/2/repeat-1/C-follow.md>)

[Objection register](<../../iterative review/open-objections.md>) · [All objections](../../README.md)

## Original scope

The paragraph was added in [the January 28 bootstrap revision](https://github.com/skirsch/KCOR/commit/2d98b3f2a1440fb9428b837eec1e6e82e310decf). Step 1 preserved strata when resampling, and Step 2 refitted the parameters for each comparison cohort. This context supports preserving the distinction between groups used to organize resampling and groups whose mortality curves are fitted. The historical wording extended this distinction to every analysis. [Original procedure](<../../iterative review/baseline/documentation/preprint/paper.md#291-stratified-bootstrap-procedure>)

The Table 8 replay preserves overlapping subgroup memberships when sampling people, fits one curve for each subgroup and pseudo-dose, and pools the resulting ratios. SI §S6 also reports separate fits by age band and vaccination status and a separate all-ages fit. [Table 8 replay](../1.3/table8_bootstrap.py#L59-L98) · [Supplement](<../../iterative review/revised/documentation/preprint/supplement.md>)

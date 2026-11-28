# Round 9 convergence - m142

Two independent blind GPT-6.2 Astra High reviews and one independent targeted Astra High review of m135 against m142. Original question verbatim: "Please examine KCOR closely. What is wrong, if anything, and how would you fix it?" All three found the proportional gamma inversion sound. No full simulation or large bootstrap rerun; no correction adopted.

## Direct carryovers from round 8 (four)

- **[9.1](../../../objections/9.1/README.md) - Missing sex suffix in overlapping control** (High; R). Deleting a male suffix with 552,591 opening risk is accepted by ordinary validation; inclusive 2/0 1.027059 to 1.019856, weighted 1.050521 to 1.046511, OK. Bootstrap rejects.
- **[9.2](../../../objections/9.2/README.md) - Sensitivity selection changes calendar horizon** (High; R). With the supplied total missing the last week, ordinary output is missing_week_history while sensitivity reports 0.999525 OK. All 1,323 complete-data cells verified.
- **[9.3](../../../objections/9.3/README.md) - Table 8 resume ignores anchor override** (Moderate; R). Changing SA_ANCHOR_WEEKS from 4 to 8 reuses 1.045599; fresh calculation gives 0.966288. Default fresh run behaves correctly.
- **[9.4](../../../objections/9.4/README.md) - Near-zero fitted variance still described as weak identification** (Low to moderate; R). SI S6.1.2 note is corrected but diagnostic bullet still equates a near-zero estimate with low curvature or weak information. H double-prime(0)=k(gamma-k theta).

## New specific findings (eleven)

- **[9.5](../../../objections/9.5/README.md) - Unexplained nondeath risk-set losses accepted** (High; A). Ordinary 56-week fixture: removing 12,000 people from each of two sex risk sets without recorded deaths changes true-null endpoint 1 to 1.45793, OK.
- **[9.6](../../../objections/9.6/README.md) - Finite contrast after cohort has no people at risk** (High; B). After complete nondeath exit, numerator has zero opening risk from April 4 onward yet July 4 anchored KCOR is 0.732363, status OK.
- **[9.7](../../../objections/9.7/README.md) - Protective proportional gap is clipped to zero excess** (High; B). Exact gamma theta 4, 60-week gap multiplier 0.3: fit theta 2.25257; true anchored endpoint 0.661664 vs fitted 0.588477, status OK. Multipliers 1 and 1.7 recover truth.
- **[9.8](../../../objections/9.8/README.md) - Pool silently drops positively weighted ages across draws** (Moderate to high; A, B). Both blind reviews reproduce sparse age anchor undefined while pooled point 1 and normalization OK, using only the remaining age. A bootstrap can silently change contributors.
- **[9.9](../../../objections/9.9/README.md) - Legacy analytical variance uses wrong anchored contrast** (Moderate; A, B). Both derive variance terms Var(J)/(A+J)^2 and [1/(A+J)-1/A]^2 Var(A), while old code divides by J^2. Bootstrap availability is unnecessarily gated by legacy bounds.
- **[9.10](../../../objections/9.10/README.md) - Table 8 generic flag counts omit residual flags** (Moderate; B). Existing identical inclusive-group fits imply at least 987/999, 950/999 and 973/999 contributing residual-flagged draws for 2021 module-off pairs. Generic flagged_draws counts only insufficient signal.
- **[9.11](../../../objections/9.11/README.md) - Valid small frailty ceiling breaks supplementary fit start** (Low; B). Configured cap 0.25, truth theta 0.1: hardcoded second start 0.5 causes proportional error and marked-wave ValueError. Default 500 unaffected.
- **[9.12](../../../objections/9.12/README.md) - Forced-alpha origin inside ongoing nonlinear wave** (Moderate for sensitivity only; A). 2022-06 origin 2022-08 lies inside configured 2021-41 to 2022-12 wave. Fresh origin gamma assumption does not follow from nonlinear pre-origin selection; example skewness 3.762 vs moment-matched gamma 3.886.
- **[9.13](../../../objections/9.13/README.md) - Age-related depletion explanation confuses absolute and relative variance** (Moderate; R). SI S6.1.2 attributes smaller theta at older ages to ongoing depletion; a proportional gamma tilt changes rate, not shape or normalized variance.
- **[9.14](../../../objections/9.14/README.md) - Current Table 8 verifier reads previous numerical ceiling** (Low; R). Documented checker fails tolerance 1e-8 after cap500; max old/new point difference 2.425e-7; printed numbers and current archived percentiles match.
- **[9.15](../../../objections/9.15/README.md) - Abstract overstates alpha localization** (Low to moderate; A). Abstract says module localizes alpha, while full survivor-law weekly identification remains unvalidated and Czech fit is sensitive only.

## Previously disclosed application limitation (one)

- **[9.16](../../../objections/9.16/README.md) - Retrospective Czech eligibility interpretation** (Application-specific; A, B). Both reviewers raise known future-product restriction, including 363,033 excluded solely at or after enrollment. The intended retrospective population and selection caveat were already disclosed; direction/magnitude unknown. Main general claim may merit application-specific qualification.

## Round 8 objections independently resolved (nine)

[8.2](../../../objections/8.2/README.md), [8.4](../../../objections/8.4/README.md), [8.5](../../../objections/8.5/README.md), [8.6](../../../objections/8.6/README.md), [8.7](../../../objections/8.7/README.md), [8.8](../../../objections/8.8/README.md), [8.9](../../../objections/8.9/README.md), [8.11](../../../objections/8.11/README.md), [8.12](../../../objections/8.12/README.md).

The targeted review verified all 1,884 pooled dates/pairs/modes including blanks, all 12 current Table 8 points within 2.23e-16 of the ceiling-500 archive, both percentile archives, 1,323 sensitivity grid cells and byte-identical figure, and all six S9 summary rows plus first dataset in each law against ordinary workbook output. The S9 replicate-level intervals were recomputed into coverage/mean summaries, not inner bootstrap percentiles independently regenerated.

Review sources: [Blind A](blind-a.md), [Blind B](blind-b.md), [Resolution](resolution.md). No reviewer recommendation is adopted automatically. The repeated retrospective Czech population concern is a known limitation, not a new change to the general KCOR method.

Frozen input verification: 1285 files; unchanged=True. [Manifest](inputs.json), [Verification](input-verification.json).

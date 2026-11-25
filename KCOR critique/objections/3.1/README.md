# 3.1 · A quiet gap can reset a positive frailty estimate

Series 3 isolates a consequence of the previously reported insufficient-signal fallback. In a fully specified model with positive frailty and zero true excess during a gap, the first quiet window recovers the true frailty. A nonpositive reconstructed gap offset then triggers a return of zero applied frailty with a successful fit flag. This changes the contrast even with noiseless data. The test concerns what the code does with the offset after calculating it. The boundary-index correction in [2.3](../2.3/README.md) stands.

The branch was already present in the original edition. Series 3 added five reproducible cases, a one-window comparison and an explicit effect on the ratio. In iteration 3 the revised fitter applies zero additional cumulative hazard for a nonpositive gap estimate and proceeds to the pooled frailty fit. Section 2.5 documents the rule. The [pre-change edition](<../../iterative review/checkpoints/iteration-2/>) is selectable separately.

[Table 8 estimates, intervals, widths and subgroup diagnostics](verification.md) · [Matched person-bootstrap replay](replay.py) · [Noiseless and ordinary-workbook checks](check.py)

[Matched fits and source history](<../../iterative review/evaluations/3/gap-check.md>) · [Reproduction script](<../../iterative review/evaluations/3/gap_check.py>)

The [root-cause record](<../../iterative review/root-causes.md>) links 3.1 to the fitter comparison, the single-window low-variance experiment and the 2022 Table 8 subgroup flags. It also records an isolated trial of continuing the pooled fit after a nonpositive gap estimate.

## Gap deficit flag

The author approved a diagnostic flag for deficits that are large relative to their sampling uncertainty. The separate [Table 8 diagnostic replay](gap_deficit_diagnostic.py) uses the existing joint fitter and the archived 999 person-bootstrap draws per enrollment. It records the signed offset at each quiet-window re-entry using the first-window seed parameters, before the fit sets any negative offset to zero.

The flag is raised when the upper end of the pointwise 95% percentile interval is below zero. It identifies a discrepancy with the fitted baseline at quiet-window re-entry. The first offset uses all modeled weeks up to the first re-entry. Later offsets compare successive re-entries. The flag leaves the applied offsets, frailty estimates and KCOR calculation as returned by the Table 8 replay. A missing seed fit is recorded as incomplete.

The replay returned 45 complete intervals across both enrollments. Two first-pass offsets in the original data were negative, both in the 2022 enrollment. Both intervals included zero, so the diagnostic flagged no deficits. The reconstructed first-pass offsets matched the one-iteration fitter on the original data. Every replayed KCOR value matched the archived joint-fit draws to numerical tolerance.

Run `py -3.11 -B "KCOR critique/objections/3.1/gap_deficit_diagnostic.py"` from the repository root to reproduce the [gap estimates, intervals and flags](gap-deficit-summary.csv).

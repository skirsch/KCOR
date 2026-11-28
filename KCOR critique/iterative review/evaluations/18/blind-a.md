# Independent review of frozen KCOR candidate m181

## Substantive defect in Monte Carlo workbook results

The workbook can report a full-population dose comparison whose numerator and denominator cover different ages. This happens with valid fixed-cohort histories and an exact supplied total. It can change even the direction of the reported contrast.

I ran the ordinary Monte Carlo workbook entry point using the 20-week, two-dose fixture from `test/test_round7_workbook.py` and the MC settings in `test/test_round16_mc_pooling.py`. I added a third birth-year group, 1970, by making ten copies of the 1940 group and tripling its weekly deaths from the eighth week in both doses. I rebuilt opening risk sets from those deaths. All three groups are present for each dose in the source. The configured MC display range is 1940–1950. For each variant I computed any supplied totals as exact weekly sums over all three ages.

| Exact redundant totals included in the same source population | Final anchored “All Ages” KCOR | Reported normalization status |
|---|---:|---|
| Both doses | 1.008827 | ok |
| Denominator only | 0.460821 | ok |
| Neither | 1.002719 | ok |

The row with a denominator total compares two ages in the numerator against three ages in the denominator. Without totals, both sides omit the 1970 group but the output still says “All Ages.” The two age-coverage fields were blank in all three cases. The mechanism is visible at `code/KCOR.py:6360`, which filters away raw ages before reconstructing missing totals, and `code/KCOR.py:4301`, which prefers an available supplied total separately for each replicate and dose in exhaustive scope. At `code/KCOR.py:6460`, the available-age labeling and coverage path explicitly excludes Monte Carlo runs.

The same path skips the raw component and total validation at `code/KCOR.py:6424`. In a separate check with only ages 1940 and 1950, I supplied internally consistent death-only total histories but tripled subsequent dose-1 deaths in the supplied total alone. The unchanged constituents gave an all-age endpoint of 1.002719. The inconsistent supplied total gave 2.501490 with status ok. The weighted age-stratum result stayed at 1.002762. This second example concerns a malformed input. The first example needs no input error.

**Repair.** Resolve the intended all-age population, reconstruct missing totals and reconcile supplied aggregates on the complete raw source for each MC replicate and dose, before narrowing the age display. If a genuinely partial extract supplies only the displayed ages, identify that population and its dose-specific coverage consistently in the workbook. Verify the three valid variants reproduce the same full-population point for complete source data, and separately verify an incompatible supplied total is withheld without suppressing unrelated valid age-stratum estimates. Take care that applying the ordinary reconciliation only after the MC range filter would incorrectly reject a legitimate full-population total.

The defect concerns the batch Monte Carlo workbook. Table 8 uses a separate overlapping layout. I found no basis to infer a change to its published numbers from this result.

## Method and reported empirical evidence

The proportional gamma inversion and the fixed-Gompertz identification proposition are internally consistent under their stated survivor and baseline assumptions. Quiet-window contamination, prior nonlinear selection, outcome-informed window choice, the lack of a known null in the age-shift exercise, the external scale required for the proposed exponent procedure, and the retrospective product restriction materially limit interpretation. The paper already describes these. The primary age-shift residual diagnostics warrant the paper's audit-only interpretation. These limitations do not by themselves establish an arithmetic error or warrant replacing the estimator.

I recalculated all six primary Table 8 points and both percentile bounds for each from `analysis/finite_reference/table8-primary/identified_only-draws.npz`, and the six sensitivity points and intervals from `analysis/ceiling500/table8/forced_alpha-draws.npz`, excluding the original observation from the percentiles. All agree with the printed table at its displayed precision. I did not regenerate the 999-draw fits.

## Minor correction

Sections 2.3, 2.6 and 2.7.1 still describe the optional NPH procedure as acting only after gamma inversion. Sections 2.5 and 2.7.3, and the implementation, correctly use the nonlinear survivor law for both the fit and the inversion when a nonlinear exponent is supplied. Amend the former passages to distinguish leaving observed weekly hazards intact from changing the fitting model. This is a wording correction. The checked Table 8 numbers do not require revision for it.

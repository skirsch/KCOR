# Objection 2, post-wave increments

The selectable revision replaces the constant post-wave offset in section 2.7.3. The in-wave calculation supplies the terminal baseline and wave exposures. The solver weights the starting gamma distribution by survival to wave exit and uses observed survival relative to that boundary. The corrected cumulative curve starts from the previously reported exit level, including any monotonicity repair. The repair itself is never counted as wave exposure.

## Production comparison

Ran the original and revised KCOR workbook entry points on the same rows from the Czech CMR workbook, enrollment sheet 2021_13, selecting input birth years 1950, 1955 and 1960. The normal grouping and pooling produced nine output birth-year/dose series. Both runs loaded the same Czech YAML, including the forced sensitivity exponent 1.18. Outputs were written to separate temporary directories.

- All 468 cohort-week rows through the final in-window date (2022-03-21) matched exactly. The pre-exit dose-pair output and the complete quiet-window fit diagnostics also matched exactly.
- Death counts, risk-set sizes and observed cumulative hazards matched throughout follow-up.
- Of 1,197 subsequent cohort-weeks, 663 adjusted weekly increments differed by more than 1e-12. These increments were obtained by differencing the exported adjusted cumulative hazard within each cohort.
- The largest absolute increment difference was 0.0000022770. For birth year 1950, dose 2, on 2022-12-12, the original adjusted increment was 0.0008783430 and the revised increment was 0.0008806200.
- The largest absolute change in a reported KCOR ratio in this replay was 0.001648. This is a sensitivity run on one enrollment and a subset of the input rows.

## Numerical verification

Run `python check_post_wave.py` from this folder. The check independently integrates the starting gamma density in log frailty and verifies recovery of the conditional post-wave increases. It covers exponents below, at and above one, four starting frailty variances, zero exposure, zero frailty, and a quiet week with zero observed increase.

The comparison verifies the code change and its numerical inversion. It does not establish the accuracy of the existing empirical wave attribution.

## Edition-specific regression tests

The [revised regression test](<../../iterative review/revised/test/test_nph_post_window_continuity.py>) exercises the production correction with two weeks after wave exit. The first has zero deaths and the corrected curve stays exactly at its exit level. For the second, the test integrates survival over frailty to generate the observed hazard from a known baseline increase of 0.05. The production correction recovers 0.05 within an absolute tolerance of 0.000000001. The constant-offset rule gives a different increment for the same observations.

The revised test, the [original edition's regression test](../../../test/test_nph_post_window_continuity.py), and the [independent equation check](check_post_wave.py) all passed.

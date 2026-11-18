# Bugs and implementation discrepancies

These findings can be assessed independently of the numbered methodological objections.

- [Monte Carlo sample clock and diagnostics](monte_carlo_sample_isolation/README.md): confirmed software defects, with isolated patch, tests, verification, and PR #4.
- [Baseline-rate fitting mismatch](k_fit_spec_mismatch/README.md): confirmed difference between the paper's joint Step 1 and the fixed-rate code; choosing the intended procedure determines whether code or manuscript needs correction.
- [Weekly reconstruction](weekly_reconstruction/README.md): the approximation can change fit decisions near a zero offset. Includes an isolated exact-update patch and paired verification.
- [Earlier Monte Carlo fix preparation](objection3_production_fix/README.md): retained patch construction, regression, and replay archive. Its status text predates PR #4; use the sample-clock package above for the current submission record.

The [full critique index](../README.md) links each objection. The [path migration notes](../PATH_MIGRATION.md) explain old locations recorded in saved evidence.

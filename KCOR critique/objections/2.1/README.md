# 2.1 Information given to KCOR in the Cox experiment

Evaluation finding N1 from the [November 21, 2026 review](<../../iterative review/evaluations/2/README.md>).

Corrected in §2.11.1. The generator supplied the true frailty variances. The manuscript now describes the experiment as an inversion with known variances. The [series 3 review](<../../iterative review/evaluations/3/convergence.md>) confirmed the disclosure and identified a figure-rebuild follow-up.

[Evaluation finding](<../../iterative review/evaluations/2/repeat-3/A.md>) · [generator](../../../test/sim_grid/code/generate_cox_bias_sim.py#L234) · [Iteration 1 checkpoint](<../../iterative review/checkpoints/iteration-1/revised/documentation/preprint/paper.md>)

[Objection register](<../../iterative review/open-objections.md>) · [All objections](../../README.md)

## KCOR error bars in the Cox experiment

The seven original week-52 estimates were reproduced using the published seeds. For each point, 1,999 person-bootstrap resamples of 200,000 people per cohort were drawn separately within the two cohorts, keeping their respective generating frailty variances fixed. Resampling multiplicities at each event time and at administrative censoring gives the person bootstrap for the Nelson–Aalen endpoint. Each error bar spans the 2.5th and 97.5th percentiles for that point.

[Reproduce the figure](figure_intervals.py) · [Estimates and interval endpoints](figure_intervals.csv) · [Before and after](figure_before_after.png)

From the repository root, run `make -C test/sim_grid revised-cox-kcor-figure` to regenerate this figure for the revised edition. The target checks all seven estimates and fourteen interval endpoints against the archived CSV before replacing the revised PNG. The historical figure uses the existing `cox-bias-figures` target.

The iteration 3 rebuild reproduced all seven points and fourteen endpoints. Every pointwise 95% interval included 1. The regenerated PNG matched the revised figure's SHA-256 in the [source manifest](<../../iterative review/revised/manifest.json>).

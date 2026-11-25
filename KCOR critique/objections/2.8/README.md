# 2.8 Reference anchor

Evaluation finding N8 from the [November 21, 2026 review](<../../iterative review/evaluations/2/README.md>).

Corrected in iteration 2. Section 2.8 now divides KCOR at each time by its value at the prespecified reference week, consistent with §2.1.1, the implementation and Table 8. The [series 3 review](<../../iterative review/evaluations/3/convergence.md>) confirmed this wording and found a one-week difference in the contamination experiment. [Revised §2.8](<../../iterative review/revised/documentation/preprint/paper.md#28-kcor-estimator>) · [Table 8 reference-week calculation](../1.3/table8_bootstrap.py#L95-L97).

§§2.1.1 and 2.8; [follow-up](<../../iterative review/evaluations/2/repeat-1/B-follow.md>)

[Objection register](<../../iterative review/open-objections.md>) · [All objections](../../README.md)

## Iteration 3 paired comparison of reference weeks

The workbook normalizes at the end of the fourth accumulating week (raw index 5 after a two-week skip). The historical contamination script normalizes one week later (raw index 6). Added hazard first applies at rebased week 4, which is raw index 6. The fifth-week denominator therefore includes the first week of added hazard in cohort A. Dividing by that larger early ratio can reduce the reported endpoint. [Workbook convention](<../../iterative review/revised/code/KCOR.py#L815>) · [Experiment anchor](../../../test/quiet_window_contamination/run_contamination_test.py#L210) · [Contamination timing](../../../test/quiet_window_contamination/run_contamination_test.py#L265)

We replayed all 200 original-fitter simulations per comorbidity level from §3.3.2. Each simulated pair of cohorts produced one pair of fitted curves. Both normalizations used those same curves and the same final week. All 1,000 fifth-week endpoints matched the [archived sweep](../2.7/comorbidity_200_runs.csv) within 2.3 × 10⁻¹⁶.

| Added weekly hazard | Fifth-week mean KCOR | Fourth-week mean KCOR | Mean paired change, fourth minus fifth |
|---:|---:|---:|---:|
| 0 | 1.050 [1.034, 1.066] | 1.052 [1.038, 1.066] | +0.002 [−0.007, +0.011] |
| 0.0001 | 1.010 [0.993, 1.027] | 1.049 [1.032, 1.066] | +0.039 [+0.028, +0.050] |
| 0.0003 | 1.060 [1.045, 1.076] | 1.204 [1.185, 1.222] | +0.143 [+0.131, +0.156] |
| 0.001 | 1.229 [1.210, 1.249] | 1.764 [1.733, 1.796] | +0.535 [+0.514, +0.556] |
| 0.003 | 1.457 [1.435, 1.479] | 3.405 [3.343, 3.466] | +1.948 [+1.898, +1.998] |

| Added weekly hazard | Fifth-week difference from zero added hazard | Fourth-week difference from zero added hazard | Change in that difference |
|---:|---:|---:|---:|
| 0.0001 | −0.040 [−0.063, −0.017] | −0.003 [−0.025, +0.019] | +0.037 [+0.022, +0.051] |
| 0.0003 | +0.010 [−0.012, +0.032] | +0.152 [+0.129, +0.175] | +0.142 [+0.126, +0.157] |
| 0.001 | +0.179 [+0.154, +0.204] | +0.712 [+0.678, +0.747] | +0.533 [+0.510, +0.556] |
| 0.003 | +0.407 [+0.380, +0.434] | +2.353 [+2.290, +2.416] | +1.946 [+1.895, +1.997] |

Brackets give pointwise 95% Monte Carlo confidence intervals. The first table uses t intervals across 200 datasets for each mean or within-dataset change. The second uses Welch intervals because the random streams at different hazard levels were independent. Its last column compares the paired changes at each positive level with those at zero.

With the workbook convention, 0.0003 was the smallest tested addition whose mean increase over zero added hazard had an interval wholly above zero. Both conventions gave a mean above 1 without added hazard. The separate question of why the mean exceeds 1 at zero added hazard remains open under [2.7](../2.7/README.md#why-the-zero-contamination-mean-exceeded-one).

[Reproduction script](compare_anchor_weeks.py) · [Paired endpoints](anchor_week_pairs.csv) · [Full-precision estimates and intervals](anchor_week_summary.csv)

## Revised experiment and manuscript

Section 3.3.2 uses the planned fifth-week denominator. The [original experiment plan](../../../.cursor/plans/quiet-window_contamination_experiment_f3a4b7c2.plan.md#L43) specifies the simulation-grid reference and the first contaminated week. Weeks 0–3 on the rebased clock supply the uncontaminated baseline-rate estimate. The workbook comparison changes the reference only. Both figures and the selected-cell table use the same original fitter, calibration, seeds and 20 replicates per cell across the dynamic, comorbidity and combined scenarios. The separate comparison in the text uses 200 replicates per comorbidity level. The table and the two mean KCOR line plots show pointwise 95% Monte Carlo intervals. The combined heatmaps show cell means and fractions above 1. The frailty-fit error panels show standard deviations.

At the largest dynamic amplitude (0.005), mean KCOR was 0.782 [0.735, 0.829] in the historical grid and 2.116 [1.992, 2.240] with the workbook anchor. The number of runs above 1 changed from 0/20 to 20/20. The planned grid had mean KCOR above 1 in eight of 16 combined cells. With the workbook reference, all 16 means exceeded 1. These results describe the tested simulation, including its timing of added hazard.

The 520 grid runs matched the archived scenario and replicate IDs. The maximum numerical drift in the re-fitted frailty variances was 0.00000222. All 1,000 extended comorbidity endpoints reproduced the preceding fourth-week replay within 1e-8.

From the repository root, rebuild the planned figures and table from the archived grid. Add `--rerun-sensitivity` to the first command to recalculate the separate workbook comparison.

    py -3.11 -B "KCOR critique/objections/2.8/rebuild_contamination.py"
    py -3.11 -B "KCOR critique/iterative review/build_pdfs.py" --initialize-approved-text

[Planned grid with mean intervals](planned-anchor-grid.csv) · [Workbook sensitivity grid](workbook-anchor/grid.csv) · [Individual grid runs](workbook-anchor/runs.csv) · [All 200-run comorbidity means and contrasts](workbook-anchor/comorbidity_200_summary.csv) · [Historical grid](../../../test/quiet_window_contamination/out/quiet_contamination_agg.csv)

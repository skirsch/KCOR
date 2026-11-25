# Objection 3: uncertainty calibration

**Status: on hold pending the lead author's decision on the recommendation to drop resampling blocks of raw weeks.** The reason comes from the application. Even in a quiet period under the gamma-frailty model, the expected death rate changes with baseline risk and with who is still alive. Moving a block to a different point in follow-up mixes death rates from different times and alters the trajectory used to fit frailty. Correct survivor counts within the resample still leave the rates at the wrong times. Person resampling keeps death weeks in place and rebuilds counts and risk sets. The [block placement check](objection3_paper_method/block_bootstrap_operationalization/WHY_BLOCK_PLACEMENT_MATTERS.md) illustrates the problem even when the starting counts match the model exactly.

After the decision, verify that the paper describes the chosen resampling and fitting procedure and that its reported intervals come from that workflow. The coverage comparison below corroborates the recommendation in the tested setting.

**Paper-first position:** [The stated joint Step 1](objection3_paper_method/README.md) recovers the correct normalized value on exact expected counts in the tested one-window model. On the **same 200 model-consistent datasets**, its 95% intervals covered **189/200 (94.5%)** with the software's person-resampling Monte Carlo, versus **38/200 (19%)** with one explicit four-week aggregate-block implementation; [paired assessment](objection3_paper_method/paired_resampling/ASSESSMENT.md). The point fit, datasets, truth, and draw count were identical. Person resampling is the code's actual resampling law, whereas the paper describes blocks without fixing their placement, seam, length, or full diagnostic thresholds. Thus the paired result isolates a large resampling difference but does not assign a coverage rate to the **uniquely intended paper block interval**. The separate [600-dataset point analysis](objection3_paper_method/POINT_RESULTS.md) and [block-rule sensitivity](objection3_paper_method/block_bootstrap_operationalization/ASSESSMENT.md) provide context.

## Supporting production-code comparison

After the separate software repair, the production percentile interval covered the model-defined value in 502/600 datasets (83.7%) in the one-million-person unequal-frailty scenario. All original analyses passed the declared operational gate. Other scenarios performed better; basic intervals covered 97.7% in that same scenario. This is a specific calibration finding for the tested implementation. Because the manuscript specifies a joint initial baseline-rate/frailty fit while this code fixes the baseline rate early, the 83.7% result must not be attributed to the manuscript's stated joint-fit estimator without checking the generating revision and algorithm. It is not a universal claim about all KCOR intervals or historical manuscript results.

## Retained modules and evidence

| Work | Location | Interpretation |
|---|---|---|
| Manuscript method, first component | [paper-method folder](objection3_paper_method/README.md) | Literal joint Step 1 gives 0.9999999663 on exact counts; the manuscript's block bootstrap is not uniquely specified. |
| Explicit aggregate-block version of manuscript outline | [assessment](objection3_paper_method/block_bootstrap_operationalization/ASSESSMENT.md) | Primary four-week version covered 38/200; eight-week sensitivity 10/60. Exact-count and source-time-placement probes explain the scope of attribution. |
| Paired person versus block Monte Carlo | [assessment](objection3_paper_method/paired_resampling/ASSESSMENT.md), [comparison protocol](PAIRED_TEST_PROTOCOL.md) | Same paper estimator and 200 datasets: person 189/200, explicit four-week blocks 38/200. |
| Initial implementation audit | [script](objection3_step1_interval_audit.py), [initial](objection3_step1_results_2026-11-10/), [verified](objection3_step1_results_2026-11-10_verified/) | Maps simulation code and saved results. |
| Interval and paper-asset provenance | [assessment](objection3_step2_interval_provenance_2026-11-10/assessment.txt) | Distinguishes production person bootstrap, simulation-grid weekly Poisson draws, analytic intervals, and manuscript descriptions. |
| Original production validation | [objection3_resolution](objection3_resolution/README.md) | Retains reproduction and results from the affected implementation. Its calibration figures must not be treated as corrected-code performance. |
| Corrected production validation | [assessment](objection3_corrected_validation/ASSESSMENT.md) | Full 4,800-dataset, eight-scenario study with fresh seeds after the software fix. |
| Cause of residual undercoverage | [diagnosis](objection3_corrected_validation/DIAGNOSIS.md), [fit investigation](objection3_fit_investigation/FINDINGS.md) | Finds deterministic fitting bias and bootstrap asymmetry; records joint-fit diagnostic probes and their variance cost. |
| Practical joint-fit candidate | [module](objection3_joint_fit_candidate/README.md), [assessment](objection3_joint_fit_candidate/ASSESSMENT.md) | Separate exact conditional weekly fit. Exact-population gate passes; 60-dataset pilot covers 56/60 with much wider intervals and noisier estimates. Not a validated production replacement. |
| Existing basic interval at larger sizes | [assessment](objection3_basic_interval_check/ASSESSMENT.md) | Fresh 40-dataset screening study at 4 and 16 million people. Basic coverage falls to 17/40 at 16 million, with all analyses accepted. It cannot serve as a general interval-only repair. |

The candidate's 60-dataset interval comparison and the original 600-dataset study use different seeds and bootstrap counts. The candidate report also contains a paired comparison of point estimates on the same 60 datasets. Read the stated limits before drawing a comparative conclusion.

## Independent software dependency

The [Monte Carlo sample-isolation bug package](<../07 - Bugs/monte_carlo_sample_isolation/README.md>) has its own explanation, patch, regression, and PR. Those bugs were discovered while working on this objection but are ordinary software defects. Accepting their repair does not require accepting a changed statistical method. Their repair alone does not settle calibration.

The [baseline-rate fit mismatch](<../07 - Bugs/k_fit_spec_mismatch/README.md>) is a separate specification question revealed by the residual deterministic offset. The paper describes a joint first fit, while the validated code fixes the early rate before iterating frailty. This needs an explicit algorithm choice; the exploratory joint fit is not an approved replacement.

Remaining review questions include calibration, alignment of the paper's description with implementation, and identification of the source workflow behind historical manuscript intervals. No historical figure or table is assigned to a workflow without provenance evidence.


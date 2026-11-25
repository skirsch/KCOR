# Retained evidence archives

The large batches of generated dataset and bootstrap files are archived beside their run summaries. The scripts, configuration, verification notes, and aggregate results are available directly in the numbered folders.

To rerun a script that reads the archived files, change to the directory containing its archive and run `python -m zipfile -e datasets.zip .`. For the historical workbook replay, use `replay_data.zip` in the same command. The archive entries reproduce the original paths relative to that directory. Extracted files are ignored by Git.

| Objection 3 run | Archive | Files |
|---|---|---:|
| corrected validation/final 2026-11-10 | [datasets.zip](<objections/1.3/objection3_corrected_validation/final_2026-11-10/datasets.zip>) | 9,600 |
| joint fit candidate/pilot | [datasets.zip](<objections/1.3/objection3_joint_fit_candidate/pilot/datasets.zip>) | 120 |
| paper method/block bootstrap operationalization/main | [datasets.zip](<objections/1.3/objection3_paper_method/block_bootstrap_operationalization/main/datasets.zip>) | 400 |
| paper method/block bootstrap operationalization/sensitivity | [datasets.zip](<objections/1.3/objection3_paper_method/block_bootstrap_operationalization/sensitivity/datasets.zip>) | 120 |
| paper method/paired resampling/main | [datasets.zip](<objections/1.3/objection3_paper_method/paired_resampling/main/datasets.zip>) | 400 |
| paper method/saved 600 | [datasets.zip](<objections/1.3/objection3_paper_method/saved_600/datasets.zip>) | 600 |
| resolution/development/reproduction 1000 iterations | [replay_data.zip](<objections/1.3/objection3_resolution/development/reproduction_1000_iterations/replay_data.zip>) | 2 |
| resolution/final 2026-11-10 | [datasets.zip](<objections/1.3/objection3_resolution/final_2026-11-10/datasets.zip>) | 9,600 |

# Objection 3 resolution module

This directory contains an isolated validation of the current production Monte Carlo uncertainty procedure. Production code, data, manuscript files, and earlier experiments are not edited. This module is part of the untracked critique folder; no commit is made.

Start with [the assessment](ASSESSMENT.md), [the design](DESIGN.md), and [the production findings](IMPLEMENTATION_FINDINGS.md). The module is complete: fixed numerical validation, production comparisons for the declared age-specific scope, interval reconstruction, and the completion audit have passed. Objection 3 remains as a specific implementation/documentation/calibration issue; completion does not require favorable coverage or changes to production code.

## Files

- `production_adapter.py`: imports original KCOR functions; reproduces production grouping, fitting, fallback, inversion, anchoring, and quantiles. Configuration overrides are in-memory.
- `simulation.py`: exact gamma survival probabilities, independent person samples, and pooled person-bootstrap samples represented by lifetime-category counts.
- `verify_production.py`: compares complete trajectories with the full production workbook calculation; keeps input/output workbooks and logs.
- `verify_person_aggregation.py`: executes the original CMR aggregation function on explicit records and actual pandas resamples.
- `check_model.py`: checks the analytic reference using exact gamma inversion and verifies that independent outer samples differ.
- `design.json`, `DESIGN.md`: scenarios, sample sizes, gates, tolerances, and stopping rules fixed before final outcomes.
- `run_study.py`: deterministic development/final runs, per-job results, retained draws, source hashes, and guarded resume.
- `summarize.py`: reconstructs all intervals, checks dataset uniqueness, replays selected seeds, and reports calibration and numerical precision.
- `verify_captured_production.py`: compares the 100,000 completed production age-specific points, theta columns, and production interval summary captured at the actual final batch size.
- `target_sensitivity.py`: post-run diagnostic using the population outputs recorded during development; it does not change the primary reference, procedure, or gate.
- `audit_completed_module.py`: checks the final evidence scope, source hashes, production comparisons, scenario denominators, and consistency of reported coverage.
- `development/`: retained development attempts, successful reproduction tests, and pilot. Failed early verification attempts are preserved; they are not final evidence.
- `final_2026-11-10/`: 4,800 final datasets and 4,795,200 bootstrap estimates, individual result records, the fixed-run manifest, and the numerical assessment/audit.
- `source_snapshot/`: byte-for-byte evidence copies of the tested production scripts, configuration, build target, and manuscript sources. Its manifest records hashes and Git context. Runtime code still imports the repository's production functions; these copies are retained for provenance, not maintained as another implementation.
- `COMPLETION_AUDIT.md`: requirement-by-requirement evidence and completion status.

## Reproduce

Run these from the repository root using the existing Python environment with NumPy, pandas, SciPy, PyYAML, openpyxl, and the production KCOR dependencies. Exact versions used for the final run are recorded in its manifest. Output directories must be new; scripts refuse to overwrite previous work.

```powershell
python 'KCOR critique/03 - Objection 3/objection3_resolution/verify_production.py' --draws 9 --design-case --out 'KCOR critique/03 - Objection 3/objection3_resolution/development/reproduction_new'
python 'KCOR critique/03 - Objection 3/objection3_resolution/verify_production.py' --draws 999 --design-case --out 'KCOR critique/03 - Objection 3/objection3_resolution/development/reproduction_1000_new'
python 'KCOR critique/03 - Objection 3/objection3_resolution/verify_person_aggregation.py' 'KCOR critique/03 - Objection 3/objection3_resolution/development/person_aggregation_new.json'
python 'KCOR critique/03 - Objection 3/objection3_resolution/check_model.py' 'KCOR critique/03 - Objection 3/objection3_resolution/development/model_checks_new.json'
python 'KCOR critique/03 - Objection 3/objection3_resolution/run_study.py' --phase development --out 'KCOR critique/03 - Objection 3/objection3_resolution/development/pilot_new' --workers 6
python 'KCOR critique/03 - Objection 3/objection3_resolution/run_study.py' --phase final --out 'KCOR critique/03 - Objection 3/objection3_resolution/final_reproduction' --workers 6
python 'KCOR critique/03 - Objection 3/objection3_resolution/summarize.py' 'KCOR critique/03 - Objection 3/objection3_resolution/final_reproduction'
```

The retained study's closure check is run with a new output filename:

```powershell
python 'KCOR critique/03 - Objection 3/objection3_resolution/audit_completed_module.py' --out 'KCOR critique/03 - Objection 3/objection3_resolution/completion_evidence_new.json'
```

The retained final audit is `completion_evidence_final.json`. Additional analysis scripts also refuse to overwrite their earlier output files.

Reusing the final seed namespace reproduces the same final experiment; it does not constitute fresh validation after changing an estimator. If a run is interrupted, inspect the process first. After it has actually stopped, `--resume` can finish missing jobs only when the recorded source/configuration hashes still match. An incomplete `.npz`/`.json` pair requires diagnosis; the runner refuses to overwrite it.

The 1,000-sample full-workbook command is substantially slower than the array adapter because it also executes production's aggregate-output loops. In the retained run, all 1,000 completed age trajectories were captured read-only from production before an unused aggregate loop finished. All 100,000 points matched exactly; theta columns and production quantiles also matched. The unused remainder was then stopped. The full-size workbook export was not completed, and no age-pooling validation is claimed. Smaller full-workbook comparisons completed normally. The capture script, source trace, CSV, checksums, and `run_disposition.json` preserve exactly what was done.

To recheck that retained full-size evidence without rerunning the expensive workbook assembly, run the following against a copy of the evidence directory that omits the verifier's two output files (`verification_age_scope.json` and `captured_production_interval_summary.log`); the verifier refuses to overwrite them:

```powershell
python 'KCOR critique/03 - Objection 3/objection3_resolution/verify_captured_production.py' 'KCOR critique/03 - Objection 3/objection3_resolution/development/reproduction_1000_copy'
```

`development/capture_completed_age_rows.py` records the read-only capture used in the live Python 3.14 process. Observation did not change numerical inputs, fitted parameters, or source code. The direct 1,000-sample workbook command above remains available to regenerate the full export if desired.

Early development directories are retained intentionally. Attempt 01 assumed the wrong output-sheet name. Attempt 02 exposed the difference between intended sample-specific time and the actual combined-sample time index. Attempt 03 reproduced the curves but then called the summary helper by the wrong name. These were resolved before the successful reproduction fixtures and before the final design was frozen; no final study settings were tuned from validation outcomes.

## Interpretation limits

The final study concerns the current age-specific production path, with optional wave adjustment disabled. It does not certify age pooling, the optional correction, historical published intervals, or the intended estimator after a future repair. Conditional coverage is undefined if no analyses pass the declared gate; it must not be reported as zero or perfect coverage. Assumption-violation stresses are reported separately from within-model results.

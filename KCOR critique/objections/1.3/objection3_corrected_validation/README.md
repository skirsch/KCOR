# Objection 3: calibration after the Monte Carlo repair

This self-contained extension reruns the earlier fixed coverage design against the corrected source from [PR #4](https://github.com/skirsch/KCOR/pull/4). The earlier module and production checkout remain unchanged.

The full run and audit are complete. Read [ASSESSMENT.md](ASSESSMENT.md) for the result and its limits. The primary percentile interval still undercovers in the model-consistent large-population unequal-frailty scenario, although coverage improves sharply after the repair in most scenarios.

The post-run [DIAGNOSIS.md](DIAGNOSIS.md) investigates the remaining 83.7% coverage, separating numerical checks, deterministic population fitting, and bootstrap asymmetry without changing the prespecified result.

See `DESIGN.md` and `design.json` for the fixed rules. `provenance.json` records reused source hashes and the corrected production snapshots. `simulation.py`, `summarize.py`, and `check_model.py` reuse the earlier implementations; the adapter changes only its source import and time indexing, and the runner records the new source hashes. The production-comparison script records the corrected source and actual reporting point.

Run from the repository root:

```powershell
python 'KCOR critique/03 - Objection 3/objection3_corrected_validation/check_model.py' 'KCOR critique/03 - Objection 3/objection3_corrected_validation/development/model_checks_reproduction.json'
python 'KCOR critique/03 - Objection 3/objection3_corrected_validation/verify_production.py' --draws 9 --design-case --out 'KCOR critique/03 - Objection 3/objection3_corrected_validation/development/production_10_reproduction'
python 'KCOR critique/03 - Objection 3/objection3_corrected_validation/run_study.py' --phase development --out 'KCOR critique/03 - Objection 3/objection3_corrected_validation/development/pilot_reproduction' --workers 6
python 'KCOR critique/03 - Objection 3/objection3_corrected_validation/run_study.py' --phase final --out 'KCOR critique/03 - Objection 3/objection3_corrected_validation/final_reproduction' --workers 6
python 'KCOR critique/03 - Objection 3/objection3_corrected_validation/summarize.py' 'KCOR critique/03 - Objection 3/objection3_corrected_validation/final_reproduction'
```

Output locations must be new. `--resume` is reserved for a stopped run with matching source hashes and complete per-job file pairs. Check running processes before resuming. This run completed after preserving 1,062 file pairs through a pause and resuming the remaining jobs; `final_execution_resumed.json` records the successful stage exit codes.

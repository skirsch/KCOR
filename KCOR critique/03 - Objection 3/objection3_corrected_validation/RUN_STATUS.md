# Corrected calibration execution — complete

The resumed run finished on 2026-11-11 at 10:54:59 UTC (05:54:59 Toronto time). All 4,800 datasets completed, and the run, summary, and audit each exited with code zero. `final_2026-11-10/assessment/completion_audit.json` reports `checks: passed`. Read `ASSESSMENT.md` for the calibration conclusion. The following paragraphs preserve the pause/restart history for provenance.

On 2026-11-11, the user authorized resuming the run. The previously suspended processes had disappeared, so the old process IDs in `pause_state.json` are historical only. Exactly 1,062 complete JSON/draw file pairs remained, with no `COMPLETE.json`. Frozen execution/configuration hashes matched the manifest. `development/restart_remaining.py` launched the existing runner with `--resume` to compute the 3,738 missing datasets, followed automatically by `summarize.py` and `finish_study.py`. The coordinator process at restart was 31252, with runner PID 29972; these IDs may later become stale. Check live process command lines and `final_resume_run.log` for current status. Stage logs are `final_resume_run.log`, `final_resume_summarize.log`, and `final_resume_audit.log`. Final stage exit codes are written to `final_execution_resumed.json`.

Do not launch another run while this coordinator or runner is active. If the resumed process is lost, recheck all JSON/draw pairs and hashes before any further resume. Preserve all existing artifacts.

On 2026-11-11 at 01:36:26 UTC, the user reported that the run was using too much laptop CPU and wanted to work. The eight processes belonging to this study (six workers, study runner, and coordinator) were suspended with `psutil.Process.suspend()`. `pause_state.json` records their process IDs and confirms that each status was `stopped`. Do not interpret the existence of these processes as active computation. The user has not authorized resuming them yet. The study had completed at least 1,055 of 4,800 datasets just before the pause; check the completed JSON records for the exact count. At the observed pre-pause rate, roughly 6–7 hours of computation remained **after resumption**; there is no calendar ETA while paused.

The post-pause file check found exactly 1,062 completed JSON records and 1,062 matching draw files. Suspending preserves running work in memory and the already saved artifacts. If the user later asks to resume, use `development/resume_paused_run.py` only after checking the recorded PIDs and their command lines. Do not launch a duplicate study process. Do not use the original runner's `--resume` on top of these suspended processes.

The full fixed run was launched on 2026-11-10 at approximately 23:42 UTC. This file records launch and pause context, not a completion claim; inspect the files below for current state.

- Coordinator process at launch: 77592 (`execute_final.py`).
- Study process at launch: 79492 (`run_study.py`), six worker processes.
- Original tool session: 39888, if still available.
- Run directory: `final_2026-11-10/`.
- Progress log: `final_run.log`.
- After computation, the coordinator automatically runs `summarize.py`, then `finish_study.py`.
- Stage logs: `final_summarize.log` and `final_audit.log`.
- Final coordinator exit records: `final_execution.json`.
- Computation completion marker: `final_2026-11-10/COMPLETE.json`.
- Final numerical audit: `final_2026-11-10/assessment/completion_audit.json`.
- Comparison report: `final_2026-11-10/assessment/COMPARISON.md`.
- Detailed comparison, including conditional coverage and its uncertainty: `final_2026-11-10/assessment/comparison.json`.

The development pilot and ten-sample full-workbook equivalence check passed. The pilot is not adequate evidence of calibrated coverage. Initial full-run throughput suggests several hours of computation; no early stopping based on coverage is permitted.

For a follow-up, check whether these processes are still running before doing anything else. Do not launch a duplicate run or change frozen execution files. A completed study is not fully audited until the coordinator records three successful stage exit codes and the completion audit says `passed`. If an individual audit stage fails, preserve its log, diagnose it, and rerun only the necessary stage with new output paths as needed; do not discard the simulation results.

The old module, original production checkout, and PR remain unchanged by this study. The new module imports its own snapshot of the corrected PR source. Original validation sources and exact scenario definitions are checked by the final audit. Final seeds are fresh; comparisons with the old run are scenario-level comparisons.

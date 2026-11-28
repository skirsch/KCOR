# Reproducing solver-scale tests

These files retain the actual matched inputs/results from the m129-to-m130 change, rather than a new simulation design. The Python files here are preserved copies; run the corresponding paths in the candidate root, where their existing imports and data reside.

- `python test/test_solver_scaling.py`: 24 noiseless exact-hazard cases.
- `python analysis/solver_scaling/low_frailty_replay.py <m129-checkout>`: same 200 datasets at each of 50k, 500k and 5M people per cohort, with true variances 0.5 and 4, and zero added hazard. Each generated history is passed to both fitters. `low-frailty-summary.json` gives 95% Student-t intervals of the across-dataset means separately from the empirical 2.5th and 97.5th percentiles of individual estimates. `zero-contamination-differences.csv` gives intervals on paired mean differences. No estimator-level bootstrap coverage is being claimed from these across-dataset mean intervals.
- `python analysis/solver_scaling/s6_replay.py . --draws 999`: S6 all-window refits on the original matched 999 draws. Raw frailty percentiles and applied-variance KCOR percentiles are distinct.
- `python analysis/solver_scaling/table8_replay.py --mode both --draws 999`: both Table 8 modes on the original matched draws. The script checkpoints progress in a mode-specific `-partial.npz`; keep the same draw count when resuming. Final mode-specific JSON and NPZ retain estimates, flags and fitted parameters; the two original points per mode are stored at draw index zero. `python analysis/solver_scaling/check_workbook.py` directly checks all 12 original ratios against the workbook. The check reads the completed mode-specific checkpoints (or final draws). The `input.xlsx` in `analysis/table8` supplies exactly the original workbook.
- `python analysis/age_shift/replay.py --draws 999`: verifies the actual workbook for three draws, then reproduces both age-shift figures' underlying curves using 999 matched draws.

The earlier S6 first-window study remains archived; all current tables use the all-window method. No nonlinear-wave estimator objective or population definition is modified by 7.3.

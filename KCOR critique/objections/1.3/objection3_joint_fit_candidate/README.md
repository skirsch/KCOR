# Conditional joint-fit candidate for KCOR

This module is a bounded trial of a new *fitting* step for the single quiet-window scenario where the corrected production percentile interval undercovered. It keeps the original correction, normalization, model target, and person-bootstrap resampling. It does not modify the original KCOR software or existing objection-resolution modules.

Read `DESIGN.md` before interpreting results. `conditional_fit.py` fits the effective baseline rate and frailty together from the trusted quiet window using the exact weekly conditional gamma model. `check_exact_population.py` runs the fixed deterministic gate. `run_pilot.py` performs the fresh, one-worker pilot only if that gate passes. `summarize_pilot.py` checks saved draws and reports numerical results. Outputs are retained under `exact_population/` and `pilot/`.

This candidate handles one contiguous quiet window. It has not been generalized to separate windows with untrusted gaps, optional wave correction, or empirical paper outputs. The pilot has only 60 outer datasets and cannot establish nominal coverage.

Read `ASSESSMENT.md` for the completed pilot and its interpretation. The exact-population gate passed, but the joint fit produced much noisier point estimates and wider intervals in the finite-sample pilot. The candidate is not proposed as a production replacement.

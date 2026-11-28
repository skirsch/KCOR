# Investigation of objection 3's remaining fit bias

This is an isolated, post-run investigation of the corrected validation's one-million-person unequal-frailty scenario. It reads the corrected production source snapshot and retained study artifacts. It does not edit the production code, the pull request, the completed coverage module, or the paper. Its exploratory refits do not replace the prespecified coverage results.

Start with [FINDINGS.md](FINDINGS.md). The deterministic calculation is in `derive_population.py` and `population_derivation.json`; the 600 saved-original refits are in `analyze_saved_datasets.py`, `saved_dataset_refits.csv`, and `saved_dataset_refits_summary.json`; three exact bootstrap-seed replays are in `replay_bootstrap_fits.py` and `bootstrap_fit_replay.json`. `audit.py` and `audit.json` check the evidence and unchanged source hashes.

The scripts can be run individually from the repository root with the same Python dependencies as the corrected validation. They refuse to overwrite their existing outputs; use a fresh copy of this directory to reproduce them. No new multi-hour coverage run is needed for this investigation.

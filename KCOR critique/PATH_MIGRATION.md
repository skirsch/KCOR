# Path changes for retained critique evidence

The study files were moved into the seven visible topic folders on 2026-11-14. Saved manifests and result data were not rewritten, because their hashes and source paths describe the original runs. When an old path is named in a manifest, apply these prefix changes to locate the same retained file:

| Former path beneath `KCOR critique/` | Current location |
|---|---|
| `step*_results_*` and `step*.py` | `01 - Objection 1/` with the same name |
| `KCOR_Objection_1_Corrected_Framing.md` | `01 - Objection 1/KCOR_Objection_1_Corrected_Framing.md` |
| `objection3_*` study folders and `objection3_step1_interval_audit.py` | `03 - Objection 3/` with the same name |
| `objection3_production_fix/` | `07 - Bugs/objection3_production_fix/` |
| `objection6_weekly_inversion/` | `06 - Objection 6/objection6_weekly_inversion/` |
| `bugs/monte_carlo_sample_isolation/`, `bugs/k_fit_spec_mismatch/` | `07 - Bugs/` with the same final folder name |
| `objections/01_interpretation/README.md` through `objections/06_mathematical_statements/README.md` | The matching numbered objection folder's `README.md` |

The first row applies to the nine objection 1 stages only; the `objection3_step1_*` names follow the objection 3 row. The original reviews and paper PDF remain at the critique root.

Markdown links and active code references were updated where practical. Historical manifests, frozen design files, and captured provenance keep their original bytes. A verification script that checks a source hash against one of those older manifests may need to resolve the old path through this map; a hash mismatch after reorganizing a script is not new evidence about KCOR.

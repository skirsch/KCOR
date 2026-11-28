# Finding files cited under older names

**25 November 2026:** The 19 numbered directories formerly at the critique root now live in `objections/x.y/`. The identifiers are unchanged. A path such as `KCOR critique/1.4/alpha-pooled-replay/` now resolves to `KCOR critique/objections/1.4/alpha-pooled-replay/`. The old names intentionally remain inside frozen evaluation bundles, historical manifest records, earlier source snapshots, and already-rendered manuscripts. Live documentation, replay entry points and the PDF build script use the new location. The PDFs and manifest-pinned method and manuscript source files have not been modified.

Paths in this table are relative to `KCOR critique/`. The [objection index](README.md) lists each topic by name.

| Earlier location | Current location |
|---|---|
| `01 - Objection 1/` through `06 - Objection 6/` | `objections/1.1/` through `objections/1.6/`, respectively |
| `07 - Bugs/` | [software-fixes/](software-fixes/) |
| `iterative review/rounds/round-01-objection-1-diagnostics/` | [1.1/](objections/1.1/) |
| `iterative review/rounds/round-02-objection-2-post-wave/` | [1.2/](objections/1.2/) |
| `iterative review/rounds/round-03-objection-3-person-bootstrap/` | [1.3/](objections/1.3/), except for the [Table 8 replay write-up](objections/1.4/table-8-replay.md) under objection 1.4 |
| `iterative review/rounds/round-06-weekly-inversion/` | [1.6/](objections/1.6/) |
| `iterative review/rounds/iteration-2-n2/` through `iteration-2-n5/` | `objections/2.2/` through `objections/2.5/`, respectively |
| `iterative review/rounds/README.md` | [Revision records](revision-records.md) |
| `iterative review/evaluations/2026-11-21/` | [Evaluation 2 archive](<iterative review/evaluations/2/>) |

## Earlier evidence paths

| Paths in the original study records | Current location |
|---|---|
| Objection 1's nine `step*_results_*` and `step*.py` stages and `KCOR_Objection_1_Corrected_Framing.md` | [1.1/](objections/1.1/), under the same names |
| `objection3_*` studies and `objection3_step1_interval_audit.py` | [1.3/](objections/1.3/), under the same names |
| `objection3_production_fix/` | [Software fix](software-fixes/objection3_production_fix/) |
| `objection6_weekly_inversion/` | [Weekly inversion study](objections/1.6/objection6_weekly_inversion/) |
| `bugs/monte_carlo_sample_isolation/` and `bugs/k_fit_spec_mismatch/` | [software-fixes/](software-fixes/), under the same final folder names |
| `objections/01_interpretation/README.md` through `objections/06_mathematical_statements/README.md` | The README in the corresponding folder from `objections/1.1/` through `1.6/` |
| `KCOR.pdf` and both `KCOR*_Review_EN.md` reports | [Original review](original-review/) |
| Earlier objection index | [PR #5 evidence](PR5-evidence.md) |
| Author's proposed Cox derivation under `iterative review/rounds/` | [Author proposals](<iterative review/author-proposals/>) |

Historical manifests, frozen evaluations and checkpoint files use the paths recorded at the time. Links to a pinned upstream commit use the names in that commit. The [iteration 1 checkpoint](<iterative review/checkpoints/iteration-1/>) retains its own directory layout.


Migration verification: all 19 numbered directories were moved. All live Markdown links checked within the critique folder resolved; the current revised edition's nine manifest hashes are unchanged. The historical pooled-alpha replay initialized successfully from the new path and the post-wave conditional-survival checks passed. Historical evaluation outputs and snapshot references were intentionally not rewritten or rerun.

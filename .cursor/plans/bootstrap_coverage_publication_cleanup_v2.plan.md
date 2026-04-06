---
name: Bootstrap coverage publication cleanup v2
overview: "Single canonical plan: DGP KCOR at target_week truth; replicate CSVs; plot_bootstrap_coverage.py; delete smoke; manuscript pilot-only; QA warnings; preserve core bootstrap loop. Implement via BUILD/Agent. test/sim_grid/out/old only (no code/old)."
isProject: false
todos:
  - id: truth-dgp
    content: Dedicated truth helper; fixed documented truth_seed; true_kcor per scenario in manifest; no 1.2/0.8
    status: pending
  - id: replicate-csvs
    content: Write bootstrap_coverage_replicates.csv + bootstrap_coverage_theta_replicates.csv with required columns
    status: pending
  - id: plot-script
    content: plot_bootstrap_coverage.py → four PNGs (+ optional PDFs for vector); 0.95 line, theta by dose, fixed order
    status: pending
  - id: delete-smoke
    content: Remove bootstrap_coverage_smoke.csv and all code/manuscript refs; grep stale percentages
    status: pending
  - id: manuscript
    content: Pilot/deferred narrative; Table 11 pilot or from large run only; align truth wording
    status: pending
  - id: qa-warnings
    content: "Warnings: coverage [0,1], negative width, collapsed CI rate>n%, n_valid>n_sim, DGP mismatch, n<100"
    status: pending
  - id: preserve-core
    content: Keep per-dataset CI, deepcopy cohorts, theta by dose, Poisson approximate note, coverage_se
    status: pending
---

# Bootstrap coverage — publication cleanup (REQUIRED spec)

**Canonical plan:** use this file only. The duplicate `bootstrap_coverage_v2_spec_13f2f1f6.plan.md` was removed; it was a short pointer created by the plan UI.

## Archive directory note (`test/sim_grid/out/old`)

- There is **no** `test/sim_grid/code/old` in this repo. The only `old` folder found is **`test/sim_grid/out/old/`**.
- Contents are **historical outputs** (CSV/log), not Python source:
  - `bootstrap_coverage_smoke.csv` — **delete** (obsolete smoke artifact).
  - `coverage.log` — log from an **older buggy** run (wrong cross-sim CI; `true_kcor` 1.2/0.8). **Do not** use for truth or manuscript; optional to delete with cleanup.
  - Other CSVs (`rollout_sim_results`, `cohort_size_sensitivity`, `comparison_table`, `sim_grid_diagnostics`, `s7_diagnostics`, `cox_bias_results`) — unrelated to bootstrap coverage; **leave in place** unless the user wants a broader archive purge. **No need to move** into canonical `out/` for this task.

---

## REQUIRED changes before implementing (BUILD)

### 1. Truth definition (critical)

**Mandatory, deterministic procedure (no “or documented constant” shortcuts):**

- Implement a **dedicated helper** (name e.g. `dgp_kcor_at_target_week`) that, for a given scenario, returns the scalar **DGP KCOR at `target_week`** using the **same** scenario construction and KCOR-on-DGP evaluation path the simulations use (same scenario factory / config object as each `scenario_name`, not a parallel ad-hoc definition).
- **Per scenario, how truth is computed:** build that scenario’s DGP parameters exactly as in the coverage driver; simulate **one** reference enrollment / cohort realization using a **single fixed integer `truth_seed`** (documented in the module docstring and as a named constant, e.g. `BOOTSTRAP_COVERAGE_TRUTH_SEED = 0`); from that realization compute KCOR at `target_week` via the project’s standard `compute_kcor_for_scenario` (or equivalent) so the truth matches what “the DGP says” at the evaluation week. **Recompute whenever `target_week` changes** (no caching stale week-80 numbers under a different `--target-week`).
- **Manifest:** when `bootstrap_coverage_run.json` is written (see §8), include **`truth_seed`**, **`target_week`**, and **`true_kcor_by_scenario`** (map `scenario_name` → float). The same values must drive every replicate row’s `true_kcor` and the QA check that compares recorded truth to a recomputation from the helper.
- For **all** scenarios (including effect scenarios), **do NOT** use raw hazard multipliers **1.2 / 0.8** (or any manuscript shortcut) as `true_kcor`.
- **Remove** any language (comments, docs, plans) that suggests keeping 1.2/0.8 “for consistency with the manuscript.” **The manuscript aligns to the estimator / DGP truth, not the reverse.**

**Theta truth (if applicable):** use the same reference draw / helper pattern so `theta_true` per arm is read off the DGP at the same `dose` matching logic as the estimator arms (document alongside KCOR truth in the manifest if useful for audit).

### 2. Replicate-level outputs (required for QA + plots)

Add:

- `test/sim_grid/out/bootstrap_coverage_replicates.csv`
- `test/sim_grid/out/bootstrap_coverage_theta_replicates.csv`

**KCOR replicate row fields (each simulated dataset / scenario):**

- `scenario_name`
- `simulation_id`
- `target_week`
- `true_kcor` (recommended for audit; user list included point/CI — ensure truth is traceable)
- `point_estimate`
- `ci_lower`
- `ci_upper`
- `covered`
- `ci_width`
- `n_boot_valid`
- `n_boot_failed`

**Theta replicate row fields (additionally):**

- `dose`
- `theta_true`
- `theta_hat`

(Plus `scenario_name`, `simulation_id`, `target_week`, CI fields, `covered`, `ci_width`, `n_boot_valid`, `n_boot_failed` as for KCOR where applicable.)

### 3. Plotting script (required)

Create `test/sim_grid/code/plot_bootstrap_coverage.py` reading **canonical** summary CSVs (and replicate CSVs where needed). Generate under `test/sim_grid/out/`:

- `fig_bootstrap_coverage_kcor.png`
- `fig_bootstrap_coverage_kcor_ciwidth.png`
- `fig_bootstrap_coverage_theta.png`
- `fig_bootstrap_coverage_theta_ciwidth.png`

**Optional (recommended if figures may appear in paper or supplement):** emit the same four figures as **vector PDF** with the same basenames (e.g. `fig_bootstrap_coverage_kcor.pdf`, …), e.g. `matplotlib` `savefig(..., format="pdf")` after a vector-friendly setup (avoid unnecessary rasterized artists). A CLI flag like `--pdf` or default-on dual export is fine; not required for CI smoke.

Requirements:

- **95% nominal reference line** on coverage plots (y = 0.95 for KCOR coverage proportion).
- **Theta:** grouped / faceted by **dose** within scenario.
- **Fixed scenario order:** Gamma-frailty null → Injected harm → Injected benefit → Non-gamma frailty → Sparse events.
- Optional QA figure (from prior iteration): `fig_bootstrap_coverage_kcor_ciwidth_boxplot.png` from replicate `ci_width` — include if low cost.

### 4. Remove obsolete artifacts

- Delete `test/sim_grid/out/old/bootstrap_coverage_smoke.csv` (and any other `bootstrap_coverage_smoke.csv` if present).
- Grep repo: remove **code and manuscript** references to smoke CSV or obsolete historical coverage percentages (89.3, 87.6, 94.2, 93.8, 93.5, pilot 100% tables as “final”, etc.).

### 5. Manuscript safety

- **Do not** present current / pilot coverage numbers as **final** validation.
- Replace coverage narrative with: framework **implemented end-to-end**; **stable empirical estimates require larger** `n_simulations` and `n_bootstrap`; **final numerical claims deferred** until that run.
- If Table 11 (`tbl:bootstrap_coverage`) remains: label **pilot / implementation check** or remove empirical column until large run regenerates it.
- Align text with **DGP KCOR at evaluation week** truth (not hazard multipliers).

### 6. QA checks (warnings)

Emit warnings (non-fatal) for:

- `coverage` outside `[0, 1]` (with small float tolerance)
- negative `ci_width`
- `n_valid > n_simulations`
- **`true_kcor` mismatch** vs reference DGP `KCOR(target_week)` (same reference seed as truth helper)
- `n_simulations < 100` or `n_bootstrap < 100` (not publication-grade)
- **Collapsed CIs:** after each scenario’s replicate loop, compute the fraction of simulations where **`ci_lower == ci_upper`** (use a small float tolerance, same as elsewhere). If that fraction **exceeds a threshold** (default **5%**, overridable e.g. `--collapsed-ci-warn-frac`), emit a **warning** for KCOR and separately for theta (per `dose` arm if useful). Rationale: widespread degenerate percentile intervals often indicate too-few valid bootstrap draws, estimator stickiness, or a bug.

### 7. Preserve existing correct logic (do not regress)

- One percentile CI **per simulated dataset**; coverage = **mean** of inclusion indicators across datasets.
- Bootstrap resampling over `scenario_data["cohorts"]` with **`copy.deepcopy`** of scenario / cohort structure as now.
- Theta coverage **per arm**, matched by **`dose`**.
- Poisson resampling **documented as approximate**; `default_rng` usage preserved.
- **`coverage_se`** retained: `sqrt(p(1-p)/n_valid)` for KCOR summary (and analogous for theta arms where valid).

### 8. Optional: run manifest (recommended)

- Write `test/sim_grid/out/bootstrap_coverage_run.json` with `target_week`, `n_simulations`, `n_bootstrap`, `ci_method`, **`truth_seed`**, **`true_kcor_by_scenario`** (and theta truth map if used), `truth_definition` (short prose pointer to helper + seed), `script_version`, `git_commit`, output paths, timestamp.
- Duplicate key manifest fields on each **summary** CSV row for pandas-friendly audit.

---

## Implementation target files

- Edit: `test/sim_grid/code/compute_bootstrap_coverage.py`
- Add: `test/sim_grid/code/plot_bootstrap_coverage.py`
- Edit: `documentation/preprint/paper.md`, `paper.tex`, `main.tex` (and grep `documentation/preprint/` for stragglers)
- Delete: `test/sim_grid/out/old/bootstrap_coverage_smoke.csv` (minimum)
- Regenerate: `bootstrap_coverage.csv`, `bootstrap_coverage_theta.csv` + new replicate CSVs after code change (do not fabricate large-run numbers in prose)

---

## When to run

User preference: **implement only when BUILD / Agent** is triggered; until then this document is the spec of record.

---
name: alpha nph sandbox
overview: Add a sandbox-only comparison harness in `test/alpha/` that runs both anchored and symmetric NPH neutralization paths under a shared contract, emits side-by-side diagnostics, and keeps all new outputs scoped to the alpha sandbox.
todos:
  - id: map-neutralization-axis
    content: Add neutralization-mode config and thread it through the alpha sandbox pipeline as a first-class analysis dimension.
    status: completed
  - id: implement-symmetric-path
    content: Implement the symmetric all-cohort neutralization branch while preserving the current anchored baseline behavior and common comparison contract.
    status: completed
  - id: build-comparison-artifacts
    content: Generate the new comparison CSV, markdown report, JSON artifact, figure, and console summary with explicit recommendation logic.
    status: completed
  - id: sandbox-boundaries
    content: Gate or bypass any manuscript/preprint side effects so the comparison workflow remains sandbox-only within `test/alpha/out/`.
    status: completed
  - id: document-and-verify
    content: Update the alpha sandbox README and verify that both modes run consistently, satisfy the alpha-equals-one invariance check, and produce aligned diagnostics.
    status: completed
isProject: false
---

# Sandbox NPH Neutralization Comparison Plan

## Scope

- Keep all implementation inside [C:/Users/stk/Documents/GitHub/KCOR/test/alpha](C:/Users/stk/Documents/GitHub/KCOR/test/alpha).
- Do not modify [C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py](C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py) or manuscript text files.
- Preserve the current alpha grid, cohort selection, anchor choice, excess definition, theta propagation, and shock window unless explicitly duplicated as unchanged metadata in the comparison artifact.

## Current Touchpoints

- [C:/Users/stk/Documents/GitHub/KCOR/test/alpha/code/estimate_alpha.py](C:/Users/stk/Documents/GitHub/KCOR/test/alpha/code/estimate_alpha.py) is the single driver for config loading, cohort-wave assembly, alpha sweeps, bootstrap/LOO, reports, figures, and console output.
- The current sandbox defines anchored excess in `add_reference_excess()` as `h_excess = hazard_obs - href`, then evaluates objectives in `evaluate_pairwise_objective()` and `evaluate_collapse_objective()`.
- Primary-path helpers such as `build_primary_subset()`, `build_alpha_run_artifact()`, `build_decision_summary()`, `plot_objectives()`, and `generate_manuscript_figures()` currently assume one primary branch and hardcode `dose0`/`exclude_nonpositive`/`gamma_primary` filters.
- [C:/Users/stk/Documents/GitHub/KCOR/test/alpha/params_alpha.yaml](C:/Users/stk/Documents/GitHub/KCOR/test/alpha/params_alpha.yaml) currently has no neutralization-mode dimension, and [C:/Users/stk/Documents/GitHub/KCOR/test/alpha/README.md](C:/Users/stk/Documents/GitHub/KCOR/test/alpha/README.md) documents the existing output contract.

## Planned Changes

- Add a sandbox-only config dimension in [C:/Users/stk/Documents/GitHub/KCOR/test/alpha/params_alpha.yaml](C:/Users/stk/Documents/GitHub/KCOR/test/alpha/params_alpha.yaml): `nph_neutralization_mode: [reference_anchored, symmetric_all_cohorts]`, with both modes included in the default comparison run.
- Refactor [C:/Users/stk/Documents/GitHub/KCOR/test/alpha/code/estimate_alpha.py](C:/Users/stk/Documents/GitHub/KCOR/test/alpha/code/estimate_alpha.py) so neutralization is a first-class analysis axis carried through wave-table preparation, objective sweeps, primary subset selection, bootstrap, leave-one-out, summaries, plotting, JSON artifact generation, and console logging.
- Introduce an explicit sandbox neutralization step before the existing excess/objective code:
  - `reference_anchored`: preserve today’s anchored behavior as the baseline.
  - `symmetric_all_cohorts`: compute cohort-specific frailty factor `F_d(t; alpha)` from the same theta-based depletion model already used by the sandbox, neutralize excess for every cohort, then reconstruct the adjusted hazard stream using the same excess decomposition contract.
- Keep the excess-hazard definition exactly identical across both modes. Do not recompute or redefine excess differently inside the symmetric path.
- Add guardrails that verify both mode runs share identical cohort footprint, cohort-week rows, alpha grid, anchor, excess mode, theta scale, time segmentation, and age-band selection, and fail loudly if any non-mode dimension drifts.
- Explicitly block any mode-specific row dropping or hidden filtering so one mode cannot appear to improve by changing cohort inclusion.
- Define and export downstream coherence diagnostics for each mode. Reuse the collapse objective where helpful, and add at least one explicit comparison metric for post-neutralization cross-cohort dispersion/coherence during the shock window if no standalone metric already exists.
- Add new comparison outputs in `test/alpha/out/`:
  - `alpha_neutralization_mode_comparison.csv`
  - `alpha_neutralization_comparison_report.md`
  - `alpha_neutralization_run_artifact.json`
  - `fig_alpha_neutralization_mode_comparison.png`
- Make boundary-seeking behavior an explicit comparison axis for both modes, including whether the pooled optimum lands on the alpha-grid boundary and the bootstrap boundary fraction.
- Keep the comparison report clearly labeled as sandbox-only and include an explicit recommendation rule: promote `symmetric_all_cohorts` only if identifiability is not worse, downstream coherence improves, and no new instability or boundary-seeking appears across pooled/sensitivity views.
- Update the driver so the end-of-run console summary prints one concise line per neutralization mode plus the final recommendation.
- Prevent this comparison path from writing or depending on manuscript/preprint outputs outside `test/alpha/out/`; any existing figure-generation assumptions that escape the sandbox should be gated or bypassed for this mode-comparison workflow.
- Refresh [C:/Users/stk/Documents/GitHub/KCOR/test/alpha/README.md](C:/Users/stk/Documents/GitHub/KCOR/test/alpha/README.md) so the sandbox contract documents the new neutralization-mode comparison and its output files without implying production adoption.

## Verification

- Run the sandbox driver once with the default comparison config and confirm both neutralization modes execute under the same contract.
- Run an explicit `alpha = 1` invariance check for both neutralization modes and fail verification if the neutralization step is not identity or if downstream objectives/diagnostics diverge beyond numerical tolerance.
- Check that the new comparison CSV, markdown, JSON artifact, figure, and console summary all agree on mode labels, reported alpha metrics, coherence diagnostics, failure reasons, and recommendation.
- Confirm the comparison outputs show identical cohort inclusion footprints and identical excess-definition metadata across both modes.
- Sanity-check that legacy alpha outputs still build for the baseline path and that no production or manuscript files are modified as part of the comparison run.


---
name: update-paper-to-v75
overview: Create a phased manuscript rewrite plan that updates the KCOR paper and supplement to the v7.5 method while enforcing the locked editorial decisions and avoiding partial, internally inconsistent edits.
todos:
  - id: rewrite-main-methods
    content: Rewrite the main paper’s Abstract, conceptual framing, methods core, and algorithm summary to match the v7.5 estimator and locked editorial decisions.
    status: pending
  - id: rewrite-validation-story
    content: Rewrite the paper’s validation, identifiability, and conclusion sections around theta0 recovery, delta correctness, multi-window consistency, and controlled claims.
    status: pending
  - id: rewrite-supplement-methods
    content: Update supplement diagnostics, defaults, simulations, and technical details to support the new estimator, optional NPH module, and SI-only derivations.
    status: pending
  - id: cleanup-empirical-si
    content: Replace the old Czech theta evidence framing with revised theta0-consistent descriptive interpretation and remove it as validation support.
    status: pending
  - id: run-consistency-sweep
    content: Perform a final cross-file harmonization pass for notation, equations, prose, figures, captions, table headers, forbidden legacy phrases, and cross-references, enforcing the no-partial-edits rule.
    status: pending
isProject: false
---

# Update KCOR Paper To v7.5

## Locked Decisions

- Use `theta0` in prose and equations everywhere in the revised manuscript.
- Keep universal KCOR separate from epidemic-wave specifics.
- Present NPH as an optional epidemic-wave extension module, not as a universal KCOR step.
- Put the delta-iteration estimator steps in the main text.
- Put derivations, convergence notes, delta edge cases, and fallback rules in the SI.
- Enforce a no-partial-edits rule: never update equations without the paired prose, interpretation, diagnostics, and limitations text.

## Source Artifacts

- Primary rewrite targets: [documentation/preprint/paper.md](documentation/preprint/paper.md) and [documentation/preprint/supplement.md](documentation/preprint/supplement.md)
- Change-control artifact: [documentation/specs/KCORv7/paper_v75_differences.md](documentation/specs/KCORv7/paper_v75_differences.md)
- Additional planning notes: [documentation/specs/KCORv7/changes_to_v75_differences.md](documentation/specs/KCORv7/changes_to_v75_differences.md)
- Method source of truth: [documentation/specs/KCORv7/KCORv7.5.md](documentation/specs/KCORv7/KCORv7.5.md)

## Phase 1: Main Methods Rewrite

Update the core scientific story in [documentation/preprint/paper.md](documentation/preprint/paper.md) so the manuscript describes the actual v7.5 estimator rather than the superseded constant-baseline quiet-window fit.

Target sections:

- `Abstract`
- `1.2`, `1.5`, `1.6`
- `2.4` to `2.5`
- `2.9` to `2.11`
- `4.4`

Required changes:

- Replace the constant-baseline identification story with the Gompertz-plus-depletion model.
- Rewrite the estimator around main-text steps: seed fit, `H_0^{eff}` reconstruction, `delta_i` computation, pooled `theta0` refit.
- Rename `theta` to `theta0` everywhere.
- Reframe KCOR as universal core machinery, with NPH described as an optional extension for epidemic-wave contexts.
- Update bootstrap and algorithm-summary text so they match the new estimator.

Essential old-to-new pivot to reflect:

```text
Old: identify frailty from one quiet window under \tilde h_0(t)=k.
New: identify enrollment-time theta0 under Gompertz curvature using aligned quiet windows and structured wave offsets.
```

## Phase 2: Validation Rewrite

Rewrite the validation logic in [documentation/preprint/paper.md](documentation/preprint/paper.md) so validation is no longer anchored mainly on synthetic-null flatness.

Target sections:

- `3.1` to `3.3`
- `4.1`
- `6. Conclusion`

Required changes:

- Recast Section 3 around the new validation target: `theta0` recovery, delta correctness, multi-window consistency, and end-to-end KCOR behavior.
- Keep the synthetic null, but explicitly demote it from sole primary validation anchor to one component of a broader validation stack.
- Update identifiability prose in `4.1` to describe the new regime: Gompertz shape, rebased `t=0`, multi-window consistency, delta additivity, and explicit failure modes.
- Tighten claims to avoid overstatement around null behavior, confounding removal, and HVE correction.

## Phase 3: Supplement Rewrite And Empirical SI Cleanup

Update [documentation/preprint/supplement.md](documentation/preprint/supplement.md) to support the main-text estimator and remove old empirical-support logic tied to the superseded `theta` story.

Target sections:

- `S2`
- `S4.0` to `S4.6`
- `S5.1` to `S5.5`
- `S6.1.2`

Required changes:

- Replace SI defaults and diagnostics built around constant-baseline single-window fitting.
- Move derivations, convergence details, delta edge cases, fallback rules, and optional NPH technicalities into the SI.
- Rewrite SI validation sections so they support the new main-text validation architecture.
- Remove the old Czech `theta` evidence framing as validation support; replace it with revised descriptive interpretation consistent with `theta0` and the new estimator.
- Separate universal SI content from epidemic-wave-module content.

## Phase 4: Global Consistency Sweep

Run a manuscript-wide harmonization pass across both files after the section rewrites are complete.

Global checks:

- `theta` to `theta0`
- notation harmonization for `H_obs`, `H_0^{eff}`, `H_gom`, `delta_i`, `Delta(t)`, and rebased time
- figure captions and table headers
- `Box 2`
- cross-references between main text and SI
- optional-NPH wording consistency
- reviewer-defense consistency on Gompertz sensitivity, delta assumptions, overfitting risk, and identifiability gates

Figure and table regeneration checklist:

- regenerate the algorithm workflow figure so it reflects the new estimator and optional-NPH architecture
- review any Box 2 summary visuals or adjacent schematic summaries so they match `theta0`, the new estimator, and optional NPH wording
- regenerate or replace SI diagnostics figures that encode old `theta` semantics
- review all figure captions for `theta` to `theta0` semantics
- review all table headers and footnotes for `theta` to `theta0` semantics
- ensure Czech SI figures are no longer framed as validation evidence for the superseded estimator

Forbidden legacy phrase sweep:

- `constant baseline`
- `single quiet window`
- `theta estimated in the quiet window`
- `KCOR is applied only in quiet intervals`
- `vaccinated theta collapses to zero`
- `proof that confounding is removed`

## No-Partial-Edits Rule

Apply these coupling rules during execution:

- If `2.4` to `2.5` equations change, update `4.1`, `4.4`, `5.x`, and the matching SI diagnostics in the same pass.
- If `theta` tables or parameter interpretations change in the SI, update the surrounding SI interpretation text and any main-text references in the same pass.
- If NPH is repositioned in Methods, update the Abstract, Results framing, and Limitations together.
- If validation claims change in `3.x`, update the Conclusion and SI validation framing together.

## Success Criteria

- [documentation/preprint/paper.md](documentation/preprint/paper.md) and [documentation/preprint/supplement.md](documentation/preprint/supplement.md) describe the same estimator, notation, assumptions, and validation story.
- The revised manuscript uses `theta0` consistently and never silently reverts to old `theta` semantics.
- NPH is presented as an optional extension, not a universal step, while still being fully documented where relevant.
- The main text contains the estimator steps, while the SI contains technical support details.
- No section retains the old single-window constant-baseline identification story by accident.
- No legacy figure, caption, table, or footnote preserves old `theta` semantics by accident.
- The final QA sweep removes forbidden legacy phrases that would silently reintroduce the superseded identification story.


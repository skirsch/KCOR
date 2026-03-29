---
name: refine-paper-per-punchlist
overview: Refine the already-v7.5-aligned KCOR manuscript into a reviewer-ready submission by turning the `punch64.md` punchlist into a focused polish pass across the main paper, SI, and workflow figure/captions.
todos:
  - id: phase-1-lock-semantics
    content: Lock the core manuscript semantics in the Abstract, conceptual framing, notation/setup, Methods math, and workflow reference before touching downstream interpretation or SI.
    status: pending
  - id: phase-2-propagate-meaning
    content: Update downstream interpretation sections so identifiability, limitations, and conclusion language fully reflect the locked Methods semantics and v7.5 framing.
    status: pending
  - id: phase-3-validation-interpretation
    content: Reframe validation and simulation interpretation around theta0 recovery, multi-window consistency, full-trajectory reconstruction, and diagnostic/descriptive claims.
    status: pending
  - id: phase-4-si-alignment
    content: Align `documentation/preprint/supplement.md` to the finalized main-text semantics in the order notation, diagnostics, simulations, then Czech material, with no stronger claims than the paper.
    status: pending
  - id: phase-5-figure-refresh
    content: Regenerate or confirm Figure 1 only after Methods are stable, updating the figure asset, generator, and caption together through the paper’s reproducible toolchain.
    status: pending
  - id: phase-6-final-qa
    content: Run a final manuscript-wide QA sweep for notation drift, forbidden legacy phrases, caption/title overclaiming, symbol definitions, and cross-reference mismatches.
    status: pending
isProject: false
---

# Refine KCOR Paper Per Punchlist

## Scope

Use `[documentation/preprint/punch64.md](documentation/preprint/punch64.md)` as the polish checklist, but treat the current manuscript as a refinement target rather than a blank rewrite. The main paper in `[documentation/preprint/paper.md](documentation/preprint/paper.md)` already contains the v7.5 estimator structure, including the four-step delta-iteration workflow and optional NPH separation, so the work should focus on reviewer-facing clarity, consistency, and credibility. The execution brief should be concrete enough that the remaining known risks cannot be skipped with a superficial pass.

Useful anchor already present in the manuscript:

```text
Step 1: joint seed fit in the nearest quiet window
Step 2: reconstruct the effective cumulative hazard over the full trajectory
Step 3: compute persistent wave offsets
Step 4: pooled refit across all quiet windows
```

## Execution Order

Do not mix phases. Complete each phase fully before moving to the next. The sequencing is dependency-aware: Methods lock semantics, downstream narrative inherits those semantics, the SI aligns only after the main paper is stable, and Figure 1 is refreshed only after the Methods wording is final.

### Phase 1: Lock core semantics

Edit only `[documentation/preprint/paper.md](documentation/preprint/paper.md)` in this order:

1. `Abstract`
2. `1.2`
3. `2.1`
4. `2.4`-`2.7`
5. `2.10`

Goal:

- lock notation for `\theta_{0,d}`, `\hat{\theta}_{0,d}`, `H_{\mathrm{obs},d}`, `\tilde H_{0,d}`, `t_{\mathrm{raw}}`, and `t_{\mathrm{rebased}}`
- lock the estimator description and workflow wording
- add the explicit working-model statement tying validity to the gamma-frailty working model and Gompertz structure
- add one-line intuition after the specific equation clusters: gamma-frailty identity/inversion, Gompertz baseline, seed-fit objective, reconstruction recursion, offset computation, pooled refit, and normalized cumulative hazard
- add the delta-iteration intuition lead-in
- simplify long clause-heavy prose in the Abstract, `1.2`, and `2.5`

Stop condition:

- Phase 1 is complete only when the notation, equations, and estimator description can be read in isolation without ambiguity or dependence on later sections.

### Phase 2: Propagate meaning

Still in `[documentation/preprint/paper.md](documentation/preprint/paper.md)`, update the sections that depend on finalized Methods semantics:

1. `4.1`
2. `4.4` if present
3. `5`
4. `6`

Goal:

- align identifiability language with the locked Methods framing
- remove any leftover single-window or legacy identification logic
- soften overclaims so the paper consistently reads as diagnostic/descriptive rather than causal
- extend tone calibration to figure captions, table titles, and footnotes connected to these sections
- simplify long clause-heavy prose in `4.1` and `5`

### Phase 3: Validation and interpretation

Update the main-paper validation narrative only after Phases 1 and 2 are complete:

1. `3`
2. any adjacent simulation-description sections in `[documentation/preprint/paper.md](documentation/preprint/paper.md)`

Goal:

- align validation with `\theta_0` recovery, multi-window consistency, full-trajectory reconstruction, and end-to-end normalized behavior
- remove legacy validation framing that over-relies on single-window or null-flatness logic
- ensure claims about performance use reviewer-safe language such as “is consistent with,” “supports,” or “behaves as expected under”

### Phase 4: SI alignment

Only after the main paper is stable, update `[documentation/preprint/supplement.md](documentation/preprint/supplement.md)` in this order:

1. notation and terminology
2. diagnostics tables and identifiability wording
3. simulation and control sections
4. Czech empirical material

Goal:

- match main-text notation and terminology exactly
- use the same phrases for `working model`, `non-identifiable`, and `diagnostic failure`
- ensure the SI makes no stronger claims than the main paper

### Phase 5: Figure 1 and caption refresh

Only after Methods wording is finalized, update the Figure 1 asset referenced by `[documentation/preprint/paper.md](documentation/preprint/paper.md)` at `figures/fig_kcor_workflow.png` and any underlying generator if one exists, unless the current asset/generator already reproduces the exact required v7.5 flow without modification.

Goal:

- ensure the figure shows seed fit, full-trajectory reconstruction, persistent offsets, pooled refit, then inversion/KCOR
- make the iteration loop visible
- visually separate the optional NPH module from the core KCOR pipeline
- regenerate with the same reproducible figure toolchain already used for the paper, ideally matplotlib if that is the existing stack
- update the output asset and caption in the same pass so the figure cannot drift from the manuscript wording

### Phase 6: Global QA sweep

Perform one final pass across `[documentation/preprint/paper.md](documentation/preprint/paper.md)` and `[documentation/preprint/supplement.md](documentation/preprint/supplement.md)`.

Check for:

- undefined or drifting symbols
- inconsistent hat vs tilde usage
- forbidden legacy phrases from older single-window wording, explicitly including `constant baseline`, `single quiet window`, `theta estimated in the quiet window`, `KCOR is applied only in quiet intervals`, `vaccinated theta collapses to zero`, and any wording that implies `proof that confounding is removed`
- figure captions, table titles, footnotes, and notes that overstate conclusions
- confirm Figure 1 matches the `2.5` estimator description step-by-step rather than only matching its own caption
- cross-reference, label, and terminology mismatches between main text and SI

## Success Criteria

- The paper reads as a polished v7.5 refinement, not a partially updated transition draft.
- Reviewer-sensitive areas called out in `punch64.md` are addressed directly: notation, equation intuition, tone, identifiability, SI alignment, and workflow visualization.
- Figure 1, its caption, and any generator are either updated or explicitly confirmed to already match the full v7.5 flow with no legacy wording or missing iteration structure.
- Main text and SI present the same estimator architecture, the same limits, and the same level of evidentiary caution.


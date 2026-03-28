---
name: document-v75-differences
overview: Produce a focused differences memo that compares the current KCOR manuscript description against the v7.5 method, so we can update the paper from a verified gap list rather than ad hoc edits.
todos:
  - id: extract-method-deltas
    content: Extract the exact method deltas between the current manuscript and `KCORv7.5.md`.
    status: completed
  - id: classify-delta-layers
    content: Separate each delta into model, estimator, interpretation, notation, validation, and claim-framing layers.
    status: completed
  - id: assign-change-severity
    content: Classify each delta as hard invalidation, soft reframing, or unchanged carryover.
    status: completed
  - id: map-paper-impact
    content: Map each delta to the specific paper and supplement sections that need revision or reframing.
    status: completed
  - id: draft-differences-memo
    content: Draft a structured differences memo in the v7 spec area for review before any manuscript edits.
    status: completed
  - id: add-claim-audit
    content: Add an explicit reviewer-facing claim audit covering identifiability, Gompertz assumptions, NPH, and wave handling.
    status: completed
  - id: prepare-update-input
    content: "Use the finalized memo as the input artifact for the second plan: updating the manuscript to v7.5."
    status: completed
isProject: false
---

# Document KCOR v7.5 Differences

## Goal

Create a concise manuscript-alignment memo that documents exactly how the current paper differs from the v7.5 method before planning any wording updates.

## Primary Sources

- Current manuscript: [documentation/preprint/paper.md](documentation/preprint/paper.md)
- Current supplement: [documentation/preprint/supplement.md](documentation/preprint/supplement.md)
- New method reference: [documentation/specs/KCORv7/KCORv7.5.md](documentation/specs/KCORv7/KCORv7.5.md)

## Core Difference Categories To Capture

- Model change: the manuscript currently defines frailty identification under a constant quiet-window baseline `\tilde h_0(t)=k` with `H^{model}(t)=(1/\theta)\log(1+\theta k t)`; v7.5 replaces this with the full Gompertz-plus-depletion hazard, fixed age slope `\gamma`, and accumulated wave offsets `\Delta(t)`.
- Estimator change: the manuscript currently presents a single quiet-window cumulative-hazard NLS fit of `(k, \theta)`; v7.5 uses a nearest-window joint seed fit, then iterative `\theta` refinement across all quiet windows using delta-offset reconstruction.
- Interpretation change: the manuscript refers to cohort `\theta` without clearly anchoring it in time; v7.5 makes explicit that the target is enrollment-time `\theta_0` at rebased `t=0` after `SKIP_WEEKS`.
- Pipeline change: the manuscript mainly describes skip stabilization plus gamma inversion and KCOR ratio; v7.5 makes the three-layer pipeline explicit: dynamic-HVE skip, wave-only NPH correction, and depletion normalization with anchored KCOR.
- Validation change: the manuscript’s controls and diagnostics are tied to the earlier quiet-window formulation; v7.5 adds simulation framing around `\theta_0` recovery, use of all quiet windows, and the distinction between quiet-window identification versus full-trajectory inversion.
- Claim-framing change: the memo must distinguish between stronger methods and stronger claims, especially around identifiability, NPH correction, asymptotic behavior, and what the method does or does not solve.
- Notation change: the memo must reconcile `\theta` versus `\theta_0`, `t` versus rebased time, and `H_{\mathrm{obs}}` versus `H_0^{\mathrm{eff}}` versus `H_{\mathrm{gom}}`.

## Manuscript Areas To Map

- High priority in [documentation/preprint/paper.md](documentation/preprint/paper.md): `Abstract`, `1.2`, `1.5`, `2.4` through `2.8`, `2.10` through `2.11`, `3.1` through `3.3`, `4.1`, and `5.1` through `5.4`.
- High priority in [documentation/preprint/supplement.md](documentation/preprint/supplement.md): `S2`, `S4.1`, `S5.1` through `S5.5`, and `S6.1`.

## Deliverable Shape

Draft one comparison memo, ideally under [documentation/specs/KCORv7](documentation/specs/KCORv7), with the following sections:

- Executive summary of what changed from the paper to v7.5.
- Change-severity table that classifies each delta as `hard invalidation`, `soft reframing`, or `unchanged carryover`.
- Layered method differences grouped by model, estimator, interpretation, notation, pipeline, validation, and claims.
- Section-impact map listing which manuscript sections are now outdated and why.
- Notation-diff section listing symbols or time axes whose meanings must be updated everywhere.
- Validation-framing section showing how Section 3 and the SI need to be restructured to match v7.5.
- Claims to revise carefully in the abstract and discussion to avoid over- or under-stating the new method.
- Reviewer attack-surface section covering identifiability, dependence on fixed Gompertz `\gamma`, NPH assumptions, and wave-window selection.
- Open questions or wording risks to resolve before editing the manuscript.

## Scope Guardrails

- Do not rewrite the paper yet.
- Treat the memo as a factual gap document, not a persuasive narrative.
- Separate true methodological changes from unchanged pieces that still carry over, such as gamma inversion and the KCOR ratio concept.
- Explicitly separate model replacement from estimator replacement from interpretation changes so subtle downstream inconsistencies are not missed.
- Flag places where v7.5 narrows or sharpens claims, especially around `\theta_0`, quiet-window usage, NPH handling, and the role of non-quiet periods.
- Treat the current `§2.4` to `§2.5` manuscript core as a likely hard-rewrite zone rather than a light edit.
- Treat NPH as a major conceptual repositioning: from a manuscript limitation to a v7.5 pipeline component.

## Success Criteria

- Every major v7.5 method change is mapped to the exact paper/supplement sections it invalidates or weakens.
- Every delta is tagged by layer and severity so the next manuscript-update plan can distinguish rewrites from reframes.
- The memo distinguishes “must change” from “can mostly retain with reframing.”
- The memo explicitly identifies validation-story changes, not just equation changes.
- The memo includes a claim audit strong enough to guide abstract, discussion, and limitations edits without overclaiming.
- The resulting gap list is detailed enough to drive the next plan: updating the abstract, methods, validation framing, notation, and SI consistently.


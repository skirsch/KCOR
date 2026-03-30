---
name: submission-microedits
overview: "Apply a small set of reviewer-hardening manuscript edits: strengthen the Gompertz identifiability rationale, make model dependence explicit, soften one Czech diagnostic sentence, and add three short clarifying sentences to the core methods text."
todos:
  - id: gompertz-rationale
    content: Add the identifiability-focused Gompertz justification sentence in `documentation/preprint/paper.md` §2.4.3.
    status: completed
  - id: model-dependence
    content: Insert the explicit non-model-free / misspecification caveat in the normalization or limitations discussion of `documentation/preprint/paper.md`.
    status: completed
  - id: estimand-clarity
    content: Add the transformed-scale KCOR sentence after the estimand definition and the monotonicity sentence after inversion in `documentation/preprint/paper.md`.
    status: completed
  - id: cox-wording
    content: Tighten the Cox structural-bias sentence in `documentation/preprint/paper.md` §2.11.
    status: completed
  - id: si-tone
    content: Revise the Czech diagnostic-compliance sentence in `documentation/preprint/supplement.md` to emphasize internal consistency rather than validation.
    status: completed
  - id: rebuild-check
    content: Rebuild the manuscript PDF and confirm the new wording is coherent and non-duplicative.
    status: completed
isProject: false
---

# Submission Micro-Edits Plan

## Scope

Make only the high-value wording edits from the review. Do not change figures, results, estimator logic, or the broader narrative. The goal is to reduce reviewer attack surface while preserving the current diagnostic-first positioning.

## Targeted Edits

1. **Strengthen the Gompertz justification in Methods**

- Update the Gompertz rationale in [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md) near `§2.4.3` so it explains not just biological plausibility, but why Gompertz is used as an identifiability-preserving structure instead of a more flexible baseline.
- Current target text is here:

```318:318:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md
This Gompertz form supplies the structured age-like trend against which depletion-induced curvature is identified. Here γ is a fixed Gompertz slope and k_d is a cohort-specific scale parameter. This choice imposes more structure than the earlier flatter-baseline specification, but it aligns the estimator with the biological age-slope assumption used in the v7.5 implementation ...
```

- Insert the new sentence after the existing biological-age rationale so the section explicitly says a more flexible baseline would absorb curvature and weaken identification of `θ_{0,d}`.

1. **Make KCOR model dependence explicit in the normalization section or limitations**

- Add one plain statement that KCOR normalization is not model-free and can add bias under material misspecification of the gamma-frailty or baseline working model.
- Best primary insertion point is the inversion/normalization discussion in [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md), which already says the mapping is not model-free:

```305:305:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md
... This relation is exact only under the assumed gamma-frailty working model; it is not a model-free algebraic identity for arbitrary cohort hazards.
```

- Secondary fallback insertion point is the limitations discussion later in [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md), where model dependence is already discussed in bullets around `§4`.

1. **Clarify KCOR as a constructed estimand in `§2.1.1`**

- Add one sentence immediately after the KCOR definition in [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md):

```143:155:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md
Let H~_{0,d}(t) denote the depletion-neutralized baseline cumulative hazard ...
KCOR(t) = H~_{0,A}(t) / H~_{0,B}(t)
...
Anchoring is used only when explicitly stated ...
```

- The added sentence should make explicit that KCOR is a relative cumulative measure on a transformed scale, not a directly observed quantity in the raw data.

1. **Add the monotonicity intuition after inversion**

- In the normalization discussion of [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md), add a short sentence that the transformation is monotonic in `H_obs,d(t)` for fixed `θ_{0,d}`, preserving ordering while altering curvature.
- Best insertion point is immediately after the normalization paragraph:

```305:305:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md
... it maps the observed cumulative hazard H_obs,d(t) into a depletion-neutralized cumulative hazard scale.
```

1. **Sharpen the Cox mismatch statement**

- Strengthen the Cox critique in [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md) `§2.11` by replacing the current “structural, not finite-sample” phrasing with the more explicit claim that the bias arises from conditioning on survival under latent heterogeneity and is not removable by larger samples or covariate adjustment alone.
- Current sentence is here:

```598:598:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md
... The resulting Cox failure is structural (selection plus non-proportional hazards), not a finite-sample artifact.
```

- Keep the tone strong but confined to the specific frailty-selection setting already described.

1. **Soften the Czech diagnostic-compliance sentence in the SI**

- Update the Czech application wording in [supplement.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md) so satisfying diagnostics is framed as internal consistency under the working model, not estimator validation.
- The exact sentence to revise is:

```368:368:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md
All age strata in the Czech application satisfied the prespecified diagnostic criteria, permitting KCOR computation and reporting.
```

- Align it with the nearby existing disclaimer in the same section:

```331:331:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md
The Czech results do not validate KCOR; they represent an application that satisfies all pre-specified diagnostic criteria.
```

## Editing Principles

- Keep every edit to one sentence unless a sentence split is needed for readability.
- Prefer insertion over rewriting full paragraphs.
- Preserve the existing “diagnostic, descriptive, not causal” framing.
- Do not expand the deeper curvature-versus-baseline caveat beyond these targeted clarifications; the manuscript already says enough there.
- No figure, numbering, or caption changes are needed for this pass.

## Verification

- Rebuild the manuscript PDF after the text edits.
- Check that the new sentences read naturally in context and do not duplicate nearby caveats.
- Verify that the SI wording remains consistent with the main-text non-validation framing.


---
name: submission-audit-followup
overview: Audit the reviewer-requested manuscript micro-edits against the current draft, confirm which are already present, and optionally tighten only the remaining Czech SI sentence if the user wants an exact wording match.
todos:
  - id: verify-main-text
    content: Verify that the five requested main-text micro-edits are already present in `documentation/preprint/paper.md` and do not need further revision.
    status: completed
  - id: optional-si-tighten
    content: Optionally revise the Czech SI diagnostic sentence in `documentation/preprint/supplement.md` to the reviewer’s preferred phrasing if an exact wording match is desired.
    status: completed
  - id: conditional-rebuild
    content: Rebuild the manuscript PDF only if the optional SI wording tweak is applied.
    status: completed
isProject: false
---

# Submission Audit Follow-Up Plan

## Finding

The specific manuscript changes requested in the review are already present in the current draft in the exact target files:

- [documentation/preprint/paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md)
- [documentation/preprint/supplement.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md)

Confirmed examples:

```143:150:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md
Let $\tilde H_{0,d}(t)$ denote the **depletion-neutralized baseline cumulative hazard** ...
$$
\mathrm{KCOR}(t) = \frac{\tilde H_{0,A}(t)}{\tilde H_{0,B}(t)}.
$$
KCOR is therefore a relative cumulative measure on a transformed scale, not a direct observable quantity in the original data.
```

```307:320:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md
... This transformation is monotonic in $H_{\mathrm{obs},d}(t)$ for fixed $\theta_{0,d}$, preserving ordering of cumulative risk while altering curvature. KCOR normalization is not model-free: if the gamma-frailty working model or baseline specification is materially misspecified, the inversion may introduce bias rather than remove it ...
...
The Gompertz specification is used as a minimal parametric structure that captures the dominant exponential age-related hazard trend while preserving identifiability of depletion curvature; more flexible baselines would absorb curvature and weaken identification of $\theta_{0,d}$.
```

```600:600:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md
... The resulting Cox bias is structural: it arises from conditioning on survival in the presence of latent heterogeneity and cannot be removed by increasing sample size or covariate adjustment alone.
```

```368:368:C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md
All age strata in the Czech application satisfied the prespecified diagnostic criteria, indicating internal consistency of the working model in this dataset rather than validation of the estimator and permitting KCOR computation and reporting.
```

## Recommended Scope

Treat this as a verification / wording-alignment pass, not a substantive manuscript revision.

1. **Confirm no-op status for the five main-text edits**

- Reconfirm that the Gompertz identifiability sentence, non-model-free disclaimer, KCOR transformed-scale sentence, monotonicity sentence, and strengthened Cox sentence in [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md) match the intended reviewer-hardening language closely enough.
- Unless a stricter wording match is desired, do not change these passages.

1. **Optionally tighten the SI Czech sentence to match the review more literally**

- The only plausible remaining polish item is the sentence in [supplement.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md).
- If desired, revise it from the current wording to the reviewer’s preferred form:
  - `In this application, all examined age strata satisfied the prespecified diagnostic criteria; this indicates internal consistency of the working model in this dataset but does not constitute validation of the estimator.`
- This is a tone refinement only; the current sentence already conveys the same point.

1. **Rebuild and verify wording only if any polish is applied**

- If the SI sentence is changed, rebuild the manuscript PDF and confirm the revised sentence remains consistent with the nearby disclaimer in [supplement.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md).
- No figure, numbering, or methodological changes are needed for this pass.

## Decision Point

The only real choice is whether to leave the manuscript as-is or make the SI sentence match the reviewer’s preferred phrasing more exactly. Everything else in the review is already implemented in the current draft.

---
name: upgrade-nph-v76
overview: Upgrade the manuscript and SI from the v7.5-style optional wave rescaling to the v7.6 optional NPH exponent model, while preserving the KCOR core, non-causal scope, and diagnostic-first framing.
todos:
  - id: map-paper-nph-updates
    content: Map and rewrite all main-text NPH references, centered on replacing §2.7.1 and adding §2.7.2.
    status: pending
  - id: add-si-nph-estimator
    content: Extend the supplement with NPH exponent assumptions, diagnostics, and estimator details aligned to v7.6.
    status: pending
  - id: run-terminology-qa
    content: Perform a consistency pass so KCOR core text stays unchanged while NPH wording is upgraded throughout.
    status: pending
isProject: false
---

# KCOR v7.6 NPH Upgrade Plan

## Goal

Update the manuscript to replace the current optional epidemic-wave rescaling language with the v7.6 optional NPH exponent model: a cross-cohort frailty-dependent amplification module centered on an estimable parameter `alpha`, identified from relative cohort structure rather than absolute wave levels.

## Main Text Revisions

- Rewrite the NPH framing in [c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md](c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md) Abstract and Introduction so NPH is described as an optional `alpha` exponent module, not a prespecified scalar rescaling.
- Update contribution/scope language in the same file where the paper currently says KCOR has an optional epidemic-wave extension, so the distinction becomes:
  - universal KCOR core
  - optional assumption-bearing NPH exponent model
- Replace the current §2.7.1 rescaling formulation with a new NPH exponent section. The current section is explicitly the old target:

```464:474:documentation/preprint/paper.md
#### 2.7.1 Optional epidemic-wave NPH extension

In some settings, especially COVID-era mortality analyses, external epidemic waves can amplify hazards non-proportionally across cohorts. KCOR treats this as an **optional extension module**, not as part of the universal definition of the estimator. When this module is used, a prespecified wave-specific adjustment is applied to the stabilized hazard before cumulative-hazard accumulation:

$$
h_d^{\mathrm{adj}}(t)=c_d(t)\,h_d^{\mathrm{eff}}(t),
$$
```

- Expand Methods with:
  - a reframed §2.7.1 defining excess hazard and the `E[z^alpha | t]` working model, anchored by one explicit formal model equation such as `h_d(t) = h_{0,d}(t) + A(t) E[z^alpha | t]` or the equivalent excess-hazard form `h_excess,d(t) ∝ E[z^alpha | t]`
  - an explicit statement that `alpha` is global across cohorts within the analyzed period, not cohort-specific, and that any stray `alpha_d` notation should be removed
  - one explicit interpretation sentence that when `alpha = 1`, amplification is proportional to baseline risk and the NPH module is effectively inactive
  - a new §2.7.2 giving the pairwise and collapse estimators, with a sharper identification paragraph explaining that the unknown common wave amplitude `A(t)` cancels in cross-cohort ratios so `alpha` is identified, if at all, from relative cohort structure rather than absolute hazard levels
  - a short practical note that estimator disagreement is diagnostic of weak identification or misspecification
  - one sentence explicitly distinguishing `theta0` from `alpha`: `theta0` governs depletion geometry, while `alpha` governs how external hazards interact with that depletion
  - one explicit signal-requirement sentence stating that identification requires sufficient cross-cohort variation in `theta` and cumulative hazard; when cohorts are too similar, `alpha` becomes weakly identified and objective functions flatten
  - one failure-mode sentence tying the theory to diagnostics, e.g. flat objective functions, boundary-seeking estimates, or disagreement between estimators should be treated as weak signal or misspecification rather than valid `alpha` estimates
- Update all upstream references to §2.7 so they point to the new meaning of NPH preprocessing without changing the core delta-iteration or inversion sections. In particular, adjust mentions in §1.2, §1.5, Box 2, §2.5, §2.6, workflow text/caption, and any “optional extension settings” language to refer to the NPH exponent model rather than ad hoc wave rescaling.
- Add one short Discussion paragraph positioning `theta0` and `alpha` as distinct parameters: depletion geometry versus external-hazard amplification, with wording close to “`theta0` governs how cohorts deplete over time, while `alpha` governs how external hazards interact with that depletion.”
- Rewrite §5.4 so limitations match the v7.6 model: `alpha` is assumption-dependent, weakly identified when cross-cohort theta geometry is too similar, and only captures one class of NPH. The current text to replace is here:

```806:816:documentation/preprint/paper.md
### 5.4 Optional epidemic-wave extension and remaining non-proportional hazard risk

COVID-19 mortality exhibits a pronounced departure from proportional hazards, with epidemic waves disproportionately amplifying risk among individuals with higher underlying frailty or baseline all-cause mortality risk [@levin2020]. This phenomenon represents a distinct class of bias from both static and dynamic healthy-vaccinee effects. Even after frailty-driven depletion is neutralized, wave-period mortality can remain differentially distorted because external infection pressure interacts super-linearly with baseline vulnerability.

For such settings, the revised KCOR architecture includes an **optional epidemic-wave extension** in which prespecified wave-period hazards are rescaled before cumulative-hazard accumulation and inversion (§2.7.1). This extension is context-specific: it is not required for the universal KCOR estimator, it is not identified from the same data used for frailty fitting, and its validity depends on an external epidemiologic argument for the chosen rescaling.
```

- Add a restrained forward reference for later validation/application of `alpha` only where it is already true in the manuscript structure. Avoid overclaiming real-data results if Section 3 is not yet being expanded in this pass.
- Add an explicit reviewer guardrail in either §5.4 or the SI that `alpha` is a model-calibrated summary of frailty-dependent amplification under the working model, not a uniquely identified biological or mechanistic constant.

## Supplement Revisions

- Update [c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md](c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\supplement.md) so the SI assumptions/diagnostics tables no longer frame NPH as externally chosen rescaling only.
- Revise `A8` and the matching diagnostic rows in `tbl:si_assumptions` and `tbl:si_diagnostics` to describe the optional NPH exponent model, its identification conditions, and estimator-stability diagnostics rather than mere “plausibility of chosen rescaling.” The current rows are:

```34:55:documentation/preprint/supplement.md
| A8 Optional NPH extension plausibility | When the epidemic-wave module is used, the chosen wave-period hazard rescaling is externally justified and specified independently of frailty fitting. | Supports wave-period preprocessing before inversion |
...
| Optional NPH sensitivity | Compare results with and without the epidemic-wave module when relevant. | Limited dependence on arbitrary wave correction choices |
```

- Add an SI subsection giving the fuller NPH estimator details that the main text should only summarize. This subsection should cover:
  - excess hazard definition
  - gamma-frailty moment `F_d(t; alpha)`
  - pairwise objective
  - collapse objective
  - the common-amplitude cancellation argument in ratio form
  - identification limits and failure signatures
  - interpretation of `alpha` as a model-calibrated exponent, not a biological constant
- Update SI sensitivity text in `S4.4`, `S5.3`, and related defaults/settings so “optional NPH settings” becomes a richer robustness menu: anchor choice, excess-handling choice, theta propagation scale, estimator agreement, and weak-identification diagnostics where appropriate.
- Update Czech application framing in `S6` only enough to stay consistent with the new module wording. Do not claim the Czech application validates `alpha` unless this pass also adds those results.
- Update any SI summary/default tables that still describe the epidemic-wave module as “off by default” prespecified rescaling, so the language instead reflects an optional estimable module that may be omitted entirely.

## Terminology And Consistency Pass

- Replace manuscript-wide phrases such as “NPH rescaling,” “wave-period hazard rescaling,” and “specified independently of frailty fitting” when they refer to the model itself, while retaining the intended caveat that the NPH module is optional and assumption-bearing.
- Keep the following untouched except for cross-references that must mention the new module:
  - KCOR core equations
  - `theta0` estimation in §2.5
  - inversion in §2.6
  - non-causal estimand definition
- Check that the final wording consistently preserves three guardrails:
  - NPH is not part of universal KCOR core
  - `alpha` is identified, if at all, from cross-cohort relative structure rather than absolute hazard levels
  - failure should surface diagnostically through instability, weak separation, or estimator disagreement

## QA Pass

- Review all mentions of NPH in `paper.md` and `supplement.md` for terminology drift between “extension,” “module,” “exponent model,” “rescaling,” and “preprocessing.”
- Verify that Section numbering and internal references remain coherent after inserting a new §2.7.2.
- Verify that no text accidentally implies `alpha` is universally identified, causal, or part of the core KCOR estimand.
- Verify that no old v7.5-specific scalar-multiplier equation remains in the paper or SI except where explicitly discussed historically.


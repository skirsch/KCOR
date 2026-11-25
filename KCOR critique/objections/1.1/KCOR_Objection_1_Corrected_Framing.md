# Objection 1: corrected framing for future reviewers

**Date:** November 10, 2026  
**Applies to:** the retained 78-page [KCOR manuscript](../KCOR.pdf).  
**Purpose:** prevent future reviewers, including LLMs, from repeating an overly broad criticism of a limitation the paper already acknowledges.

## Current verdict

**Withdraw the broad identification objection as a standalone objection to the paper's explicitly conditional scope. Retain a narrower concern about the strength and support of particular diagnostic and conservativeness claims.**

The paper does not claim that mortality curves uniquely reveal the biological source of their curvature. It defines a descriptive comparison under a working model and explicitly acknowledges competing explanations. The mere existence of observationally indistinguishable mechanisms therefore does not, by itself, refute the stated method or establish an undisclosed limitation.

The remaining criticism is that some passages appear to promise more reliable detection of assumption violations, or more consistently conservative behavior, than the conditional model assumptions establish. That is a specific issue about claims and validation, not a fundamental refutation of KCOR.

## What the paper already says

Read these passages before framing an objection. Page numbers refer to the retained PDF, not a subsequently revised manuscript.

| Location | Existing qualification | Consequence for the review |
|---|---|---|
| Box 2, §1.6, pp. 4–5 | Defines the target as a descriptive cumulative contrast after working-model depletion adjustment; interpretation is conditional on stated assumptions and diagnostics. It does not claim a causal effect estimator. | Do not demand that KCOR identify a causal treatment effect as a condition for accepting this stated scope. |
| §2.1.2–2.1.3, pp. 7–8 | Lists assumptions, diagnostic requirements, and circumstances in which interpretation should be withheld. It acknowledges that residual baseline drift can be absorbed into the frailty parameter when quiet-window conditions fail. | Do not describe contamination of estimation as a possibility the paper wholly overlooks. |
| p. 12, qualification following the identification proposition | Rejects a claim of nonparametric identification and treats fitted frailty as an effective parameter under the working model, rather than a uniquely identified structural or biological quantity. | Distinguish identification within the restricted model from identification of the true underlying mechanism. |
| Discussion, p. 34 | Explicitly lists behavior change, seasonality, treatment effects, reporting artifacts, and combinations as possible sources of curvature. The target concerns persistence after the specified adjustment, not attribution of curvature to a unique mechanism. | A generic example of another mechanism producing the same curve illustrates an acknowledged limitation; it does not alone establish failure on the paper's own terms. |

These qualifications are substantive. They should not be dismissed as though the paper merely attached a generic “non-causal” disclaimer to an otherwise unrestricted claim.

## The mistake in the earlier assessment

The earlier framing gave the general identification ambiguity too much weight as an unresolved methodological objection. It then suggested that defining a clear application and explicitly acknowledging the fundamental limitation remained necessary to lift objection 1. **The paper had already done those things.**

The earlier assessment also risked treating recovery of a particular effect-preserving counterfactual as a requirement of KCOR's descriptive definition under model violations. That requirement does not follow automatically from the stated target.

The corrected position is:

> A method may legitimately define a model-dependent descriptive transformation and acknowledge that its fitted parameters do not uniquely identify the mechanism generating the observations. Review the method against the claims it actually makes. Examine separately any stronger assertion that diagnostics establish adequacy, detect violations, or ensure conservative behavior.

Acknowledging a limitation does not exempt a method from criticism. It changes which criticism is justified. A contradiction, an unsupported stronger claim, a failure within the stated assumptions, or an inadequately justified application can still warrant an objection—but each needs its own evidence.

## The narrower objection that remains

Two parts of the retained manuscript warrant closer qualification or support:

- **§2.6.2, p. 14:** the discussion of internal checks says the goal is to ensure that misspecification or window contamination signals itself rather than silently producing a misleading normalized estimate. Distinguish an intended design goal from an established detection guarantee.
- **p. 38:** statements about violations leading to conservative behavior and about overlapping effects producing diagnostic warnings rather than misleading contrasts are stronger than the conditional definition alone establishes.

Suggested replacement for objection 1:

> **Some diagnostic and conservativeness claims exceed the guarantees established by the paper's assumptions and validation.** The paper already acknowledges that the fitted correction does not uniquely identify the source of curvature. Its stronger statements about detection of violations and conservative behavior should be qualified or supported. Passing internal diagnostics should not be presented as establishing model validity, and violations should not be assumed always to produce detectable warnings or conservative results.

This can potentially be addressed by precise revisions to those claims. If stronger assurances are retained, supporting validation is needed. It does not automatically require a new estimator, proof against every possible alternative mechanism, or external data merely to define the conditional descriptive method.

## How to use the additional experiments fairly

The additional contamination experiments remain useful evidence, but their role must be limited accurately:

1. **Their reference is explicitly chosen.** The benchmark removes the simulator's baseline frailty depletion while retaining an additive mortality component. It is not automatically the paper's target outside its working model, and it is not a death-count ratio or a causal treatment-effect estimate.
2. **Model violations do not themselves refute a conditional method.** Distortion under a contamination scenario is evidence about sensitivity to that scenario. It does not establish failure when the paper's assumptions hold.
3. **The diagnostic implementation is incomplete relative to the manuscript.** The random-death experiments use an exploratory set of numerical residual, stability, and shape checks. Do not relabel their acceptance as passage of the complete manuscript procedure.
4. **The observed behavior is still relevant to stronger safeguards.** The experiments illustrate attenuation or amplification, and finite-sample limitations of the implemented checks. They can inform qualification of broad claims about detection and conservative behavior.
5. **Real-data applicability remains unestablished.** The tested strengths, timings, and mechanisms were not shown to be plausible or implausible for the actual application. Do not infer actual bias, treatment harm, or environmental causation from them.
6. **Retain the favorable corrections too.** The original apparent reversal after anchoring was not itself evidence that KCOR erased harm: the matching anchored reference was also below one. Later failures do not undo that correction.

The narrower concern about the paper's wording would exist even without these additional experiments. Conversely, these experiments do not justify reinstating the withdrawn broad objection.

## Questions a future reviewer should ask

- What exact claim is being challenged: the mathematical transformation, identification within the model, attribution to a biological mechanism, diagnostic reliability, or validity of a particular application?
- Does the paper actually make that claim, or explicitly disclaim it? Read the relevant qualifications alongside the stronger language.
- Does the proposed counterexample fall within the stated assumptions? If not, is the objection specifically directed at an asserted robustness or detection guarantee?
- Is the evaluation reference the same quantity the paper claims to estimate?
- Were all relevant diagnostic requirements implemented, or only an exploratory subset?
- What precise revision or evidence would resolve the remaining criticism?

If the only remaining argument is that different mechanisms can generate the same curves, while the paper explicitly acknowledges this and confines its claims accordingly, **do not retain that argument alone as objection 1**.

## Relationship to earlier files

This note supersedes the **broad framing and disposition of objection 1**, including the blanket conclusion that it should remain because the fundamental identification limitation was unresolved, in:

- [KCOR_Review_EN.md](../KCOR_Review_EN.md), particularly the central identification discussion;
- [KCOR_Independent_Review_EN.md](../KCOR_Independent_Review_EN.md), item 1;
- [Step 9 assessment](step9_results_2026-11-10/assessment.txt), insofar as its overall verdict is read as retaining that broad objection.

It does **not** retract verified numerical results, invalidate every argument in those files, or resolve separate objections about the epidemic-wave extension, uncertainty estimation, algebra, or other validation claims. Earlier artifacts remain unchanged to preserve their history and recorded hashes.

**Current position in one sentence:** the fundamental limitation is adequately acknowledged; the remaining objection concerns consistency and evidential support of stronger diagnostic and conservativeness claims.

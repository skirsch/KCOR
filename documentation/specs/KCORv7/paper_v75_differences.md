# KCOR manuscript vs. v7.5 differences memo

## Goal

This memo documents how the current manuscript and supplement differ from the `v7.5` method specification, so the next step can be a controlled paper update rather than ad hoc editing.

Primary sources:

- Current manuscript: [documentation/preprint/paper.md](documentation/preprint/paper.md)
- Current supplement: [documentation/preprint/supplement.md](documentation/preprint/supplement.md)
- New method reference: [documentation/specs/KCORv7/KCORv7.5.md](documentation/specs/KCORv7/KCORv7.5.md)

## Executive summary

The shift from the current manuscript to `v7.5` is not a cosmetic refinement. It replaces the paper's identification engine for `theta0` with a different model, a different estimator, and a different interpretation target.

The largest manuscript incompatibility is in `paper.md` `2.4` through `2.5`, where the current paper identifies frailty under a constant quiet-window baseline and single-window cumulative-hazard NLS. `v7.5` instead identifies enrollment-time `theta0` using a Gompertz baseline, rebased time, wave-offset reconstruction, and iterative use of all quiet windows.

Two important pieces do carry over: the gamma inversion itself and the KCOR ratio concept remain structurally intact. However, their surrounding interpretation changes because `H_obs`, the fitting target, the meaning of `theta0`, and the role of wave periods all change.

The manuscript also has a major conceptual mismatch on NPH. The paper currently treats epidemic-wave amplification mainly as a limitation and reason to exclude wave periods; the revised paper should instead present NPH as an explicit optional epidemic-wave extension module for COVID-era analyses, rather than as a universal KCOR component.

Finally, the validation narrative must be restructured. The current manuscript emphasizes synthetic null behavior, quiet-window diagnostics, and Cox mismatch under the old estimator. `v7.5` adds a new validation story around `theta0` recovery, delta-iteration correctness, multi-window consistency, and explicit separation between quiet-window identification and full-trajectory inversion.

## Change-severity table

| Delta | Layer | Severity | Current manuscript / SI | v7.5 position | Update implication |
|---|---|---|---|---|---|
| Baseline hazard for frailty identification | Model | Hard invalidation | Constant quiet-window baseline `\\tilde h_0(t)=k` and `H^model=(1/\\theta)\\log(1+\\theta k t)` | Gompertz baseline `k e^{\\gamma t}` with full depletion denominator | Rewrite core methods and any wording that says quiet-window baseline is constant |
| Theta estimator | Estimator | Hard invalidation | Single quiet-window cumulative-hazard NLS jointly fitting `(k, theta)` | Two-stage procedure: nearest-window joint seed fit, then iterative `theta0` refit over all quiet windows with wave offsets | Rewrite estimation algorithm, diagnostics language, and bootstrap description |
| Meaning of `theta0` | Interpretation | Hard invalidation | Cohort `theta` is described generically and often effectively read as a quiet-window quantity | Enrollment-time `theta0` is explicitly anchored at rebased `t=0` after skip weeks | Rewrite notation, prose, and empirical interpretation everywhere `theta` appears |
| Treatment of time origin | Notation, interpretation | Hard invalidation | Time is post-enrollment with skip exclusion, but rebased `t=0` is not central to parameter meaning | Frailty fitting is anchored at `t_rebased=0` after `SKIP_WEEKS` | Update equations, estimator prose, and any figures/text that imply late-window anchoring |
| Quiet-window usage | Estimator, validation | Hard invalidation | One prespecified quiet window is the identification window | All prespecified quiet windows are used in iterative `theta0` refinement | Rewrite methods and SI diagnostics sections that assume single-window fitting |
| Wave effects after quiet periods | Model, estimator | Hard invalidation | Later-wave accumulation is not built into frailty identification; wave overlap is mostly treated as contamination | Wave offsets `delta_i` accumulate into `Delta(t)` and are explicitly modeled in `theta0` estimation | Add delta-iteration logic and remove claims that the method simply avoids active-wave dynamics |
| NPH handling | Pipeline, claim-framing | Hard invalidation | Epidemic-wave non-PH is mainly presented as a limitation in `5.4` | NPH correction becomes an explicit optional extension module for epidemic-wave settings, applied before inversion when used | Reposition NPH from limitation-only framing into an optional Methods extension, while keeping assumptions explicit |
| Gamma inversion | Pipeline | Unchanged carryover | Exact gamma inversion from `H_obs` to `\\tilde H_0` | Same inversion retained | Keep, but reframe surrounding inputs and interpretation |
| KCOR ratio | Estimand | Unchanged carryover | KCOR ratio on normalized cumulative hazards | Same ratio retained, now explicitly mean-anchored over `NORMALIZATION_WEEKS` | Mostly retain, but clarify anchoring mechanics and terminology |
| Dynamic HVE skip | Pipeline | Soft reframing | Skip rule exists as stabilization | Same idea retained, but now explicitly sets the rebased origin for `theta0` | Preserve, but tie it more tightly to `theta0` definition |
| Diagnostics and identifiability | Validation, claims | Soft-to-hard reframing | Diagnostics revolve around fit in one quiet window and post-normalization linearity | Diagnostics now also include geometric identifiability, delta applicability, and multi-window consistency | Rewrite diagnostic framing, especially in Results, SI, and Discussion |
| Empirical `theta` narrative in Czech SI | Interpretation, claims | Hard invalidation | SI interprets near-zero vaccinated `theta` and age patterns under old estimator | `v7.5` expects some vaccinated cohorts to have non-zero `theta0` when signal exists | Rewrite SI tables/text that treat old `theta` patterns as substantive validation |

## Layered method differences

### Model layer

The manuscript's frailty-identification model is currently built around a constant baseline over the quiet window. This is explicit in `paper.md` `2.4.3` and `2.5`, where observed curvature is fit by:

- `\\tilde h_0(t)=k`
- `H^{model}(t;k,\\theta)=(1/\\theta)\\log(1+\\theta k t)`

`v7.5` replaces that with the full Gompertz-plus-depletion hazard:

- baseline `\\tilde h_0(t)=k e^{\\gamma t}`
- observed hazard `h_obs(t)=k e^{\\gamma t}/(1+\\theta H_0^{eff}(t))`
- wave-offset accumulation `H_0^{eff}(t)=H_gom(t)+Delta(t)` after waves

This is a model replacement, not a parameter tweak.

### Estimator layer

The current paper estimates `(k, theta)` once from one quiet window in cumulative-hazard space. `v7.5` uses a different estimator:

- Step 1: jointly fit `(k, theta0^(0))` from the nearest quiet window after enrollment
- Step 2: reconstruct `H_0^{eff}` across all weeks
- Step 3: compute incremental `delta_i`
- Step 4: refit `theta0` on all quiet windows with `k` fixed

This means the manuscript's current fit objective, estimator description, and algorithm figure are all outdated at the estimator level even where the broader gamma-frailty logic still holds.

### Interpretation layer

The current paper talks about cohort `theta` but does not make explicit that the intended quantity is the frailty variance at `t=0` after the skip window. `v7.5` makes that precise: `theta0` is an enrollment-time parameter, not a late quiet-window descriptor of an already depleted population.

This matters because many current interpretations implicitly assume that a late-window fit can stand in for cohort frailty at enrollment. `v7.5` rejects that premise.

### Notation layer

The symbol set is now materially different in meaning:

- `theta` in the paper often reads as a cohort-level frailty curvature parameter estimated in the quiet window; in the revised manuscript it should be replaced by `theta0`
- `t` in the paper is ordinary event time after enrollment with a skip rule; in `v7.5`, frailty fitting is explicitly on rebased time with `t_rebased=0` after skip
- the paper emphasizes `H_obs` and `\\tilde H_0`; `v7.5` additionally requires `H_0^{eff}`, `H_gom`, `delta_i`, and `Delta(t)` as operational objects

If notation is not reconciled globally, the updated manuscript will become internally inconsistent even if individual sections are locally correct.

### Pipeline layer

The current paper's practical pipeline is:

1. Define fixed cohorts
2. Estimate hazards and cumulative hazards
3. Skip early weeks
4. Fit frailty in a quiet window
5. Apply gamma inversion
6. Compute KCOR

For the revised manuscript, the pipeline should be split into:

Core universal KCOR:

1. Define fixed cohorts
2. Estimate hazards and cumulative hazards
3. Apply dynamic-HVE stabilization via `SKIP_WEEKS` and rebase time
4. Estimate `theta0` with the delta-iteration estimator
5. Apply gamma inversion
6. Compute KCOR

Optional epidemic-wave extension:

1. Apply NPH correction during prespecified wave weeks before accumulation and inversion

This keeps KCOR general as a statistical framework while making clear that NPH is context-specific rather than a universal component of every application.

### Validation layer

The paper validates the old estimator using:

- synthetic null behavior under frailty depletion
- age-shift negative controls
- positive controls
- misspecification stress tests
- quiet-window diagnostics

`v7.5` changes what needs validating:

- recovery of `theta0` rather than only qualitative curvature removal
- correctness of the delta-iteration path
- use of all quiet windows rather than one late window
- distinction between quiet-only identification and full-trajectory inversion
- end-to-end negative control behavior after NPH-aware preprocessing

The validation section therefore needs reframing, not just a few extra sentences.

### Claim-framing layer

Because `v7.5` is stronger methodologically, the risk of overclaiming is higher. The updated manuscript must distinguish:

- stronger identification strategy from universal identifiability
- improved null behavior from proof that all confounding is removed
- empirical flatness from causal validation
- an explicit NPH correction assumption from a general solution to all non-PH wave effects

## Identification and assumption changes

The old manuscript and `v7.5` do not merely use different equations. They rely on different identifiability regimes.

### Identification comparison

| Aspect | Old paper | v7.5 | Update implication |
|---|---|---|---|
| What identifies `theta0` | Curvature within one prespecified quiet window under a constant baseline | Gompertz curvature plus consistency across all quiet windows after wave-offset alignment | The identification argument in Methods and Discussion must be rewritten, not patched |
| Time anchor for `theta0` | Implicit and effectively late-window anchored | Explicit enrollment-time `theta0` at rebased `t=0` after `SKIP_WEEKS` | The paper must stop speaking as if quiet-window `theta` and enrollment-time `theta0` are interchangeable |
| Role of waves | Mostly contamination or out-of-scope disturbance | Structured source of persistent cumulative-hazard offsets | Wave effects must be treated as part of the estimator logic, not just a reason to exclude periods |
| Baseline shape used for identification | Constant baseline over the fit window | Gompertz baseline with fixed `gamma` | Identifiability now depends on a stronger biological shape assumption |
| Sufficiency of one quiet window | Central working premise | No longer sufficient by itself | The manuscript must explain why pooled quiet-window alignment is required |
| Non-identifiability signal | Near-linear quiet-window cumulative hazard drives `theta -> 0` | Low curvature, delta inapplicability, or Gompertz weak-identification can all trigger fallback / zeroing | Diagnostic language must be expanded beyond the old `theta -> 0` story |

### Delta assumption that must be named explicitly

`v7.5` introduces a structural assumption that should be made explicit in the memo and later in the paper:

- wave effects are additive in cumulative-hazard space
- the excess cumulative hazard from a wave persists forward after the wave ends
- this persistent excess can be represented by incremental offsets `delta_i` and accumulated as `Delta(t)`

This is not just an implementation detail. It is a modeling assumption. It should be presented later in the manuscript as both:

- an assumption required for the delta-iteration estimator
- a diagnostic target, because implausible or negative `delta` behavior is evidence that the structured-offset model is not appropriate for that cohort

### Flexibility and overfitting risk

One new reviewer concern introduced by `v7.5` is overfitting risk. Relative to the current manuscript, the new estimator has:

- a more structured baseline model
- iterative fitting
- multiple quiet windows
- wave-offset reconstruction

This makes it easier for reviewers to ask whether the method is fitting the hazard trajectory too well. The revised manuscript should therefore address overfitting directly rather than only indirectly through fit diagnostics.

The key safe framing is:

- `v7.5` is not adding free-form flexibility; it is adding structured assumptions
- those assumptions increase model dependence even as they improve internal coherence
- diagnostics, fallback rules, and identifiability gates are therefore more important, not less

### Validation definition change

The validation shift is stronger than a rewording of the results section. It changes what counts as validation.

Old validation anchor:

- `KCOR ≈ 1` under a synthetic null after quiet-window normalization

New validation anchor:

- recovery of enrollment-time `theta0`
- correctness of delta-offset reconstruction
- consistency across multiple quiet windows
- stability of full-trajectory reconstruction, including NPH-aware preprocessing when the optional epidemic-wave module is used

The synthetic null therefore remains useful, but it is no longer sufficient as the primary validation anchor.

### Czech SI implications

The critique of `supplement.md` `S6.1.2` is stronger than a reinterpretation issue. In the current manuscript, those Czech `theta` patterns function as implicit empirical validation of the old estimator.

Under `v7.5`, that use is no longer acceptable:

- those old cohort-specific `theta` patterns are not just reinterpreted
- they can no longer be used as validation evidence for the revised estimator

The next manuscript-update plan should treat the Czech SI `theta` discussion as both:

- a hard interpretation rewrite
- a removal or replacement of an old pillar of empirical support

## Section-impact map

### Main manuscript: must rewrite

| Section in `paper.md` | Severity | Why it is outdated |
|---|---|---|
| `Abstract` | Hard invalidation | Describes quiet-period depletion fitting and analytic inversion under the old pipeline, without Gompertz, `theta0`, delta iteration, or the new optional NPH extension architecture |
| `1.2` | Hard invalidation | Says KCOR is applied only within quiet intervals rather than distinguishing core quiet-window identification from optional epidemic-wave handling |
| `1.5` | Hard invalidation | Contribution list is written around the old identification strategy and old validation narrative |
| `1.6` Box 2 | Hard invalidation | Operational summary says depletion geometry is estimated in a prespecified quiet period, singular, before inversion |
| `2.4.3` | Hard invalidation | Constant baseline assumption is no longer the `v7.5` identification model |
| `2.4.4` and quiet-window protocol | Hard invalidation | Quiet-window validity is framed around one quiet window rather than pooled quiet-window estimation with delta correction |
| `2.5` | Hard invalidation | Estimator, objective, parameter meaning, and identifiability story are all based on the superseded single-window cumulative-hazard fit |
| `2.6` | Soft-to-hard reframing | Gamma inversion survives, but the meaning of `H_obs`, `theta0`, and the full-trajectory input must be updated |
| `2.7` | Soft reframing | Skip rule remains, but now defines the rebased origin of `theta0` |
| `2.8` | Soft reframing | KCOR ratio remains, but anchoring and terminology need alignment with `NORMALIZATION_WEEKS` |
| `2.9` | Hard invalidation | Bootstrap currently says re-estimate `(k, theta)` using the same quiet-window procedure; that must be replaced by the new estimator |
| `2.10` | Hard invalidation | Workflow figure/caption summarizes the old two-stage story and late-curvature fit |
| `2.11` and `2.11.1` | Soft-to-hard reframing | Cox mismatch still matters, but the KCOR comparator side must now reflect Gompertz-plus-delta iteration and the new null-validation framing |
| `3.1` to `3.3` | Hard invalidation | Section 3 is written around old validation claims and must be reframed to match `theta0` recovery, multi-window identification, and optional NPH-aware processing where relevant |
| `4.1` | Hard invalidation | Identifiability paragraph is stated in old `(k, theta)` quiet-window terms and needs re-anchoring around `theta0`, Gompertz curvature, and wave-offset assumptions |
| `4.4` | Hard invalidation | Reporting guidance still names the constant baseline as default and omits the new pipeline structure |
| `5.0` to `5.4` | Hard invalidation | Limitations section assumes NPH is mainly outside-scope contamination; the revised paper must instead separate universal KCOR limitations from assumptions of the optional NPH extension |
| `6. Conclusion` | Hard invalidation | Summative claims still describe the old validation and old normalization story |

### Main manuscript: mostly retain with reframing

| Section in `paper.md` | Severity | Why it can mostly survive |
|---|---|---|
| `2.1` to `2.3` | Soft reframing | Fixed cohorts, discrete hazards, and cumulative hazards remain foundational, but notation and pipeline order need updates |
| `2.4.1` to `2.4.2` | Soft reframing | Gamma frailty working model and inversion identity still hold |
| `4.2` to `4.3` | Soft reframing | Estimand positioning and relationship to negative controls remain broadly valid if rewritten around the new estimator |

### Supplement: must rewrite

| Section in `supplement.md` | Severity | Why it is outdated |
|---|---|---|
| `S2` | Hard invalidation | Diagnostic tables reflect the old single-window fit and omit delta applicability, rebased `theta0`, and assumptions governing the optional NPH extension |
| `S4.0` and `S4.1` | Hard invalidation | Defaults table still lists constant-baseline cumulative-hazard NLS as the reference implementation |
| `S4.2.1` | Soft-to-hard reframing | Synthetic negative control framing survives, but the estimator and fit logic need updating |
| `S4.3` | Soft reframing | Positive control concept survives, but linkage to the new pipeline and validation story must change |
| `S4.4` | Hard invalidation | Sensitivity analysis is centered on old quiet-window perturbation logic and old baseline assumptions |
| `S4.6` | Hard invalidation | Joint frailty/effect simulation uses constant-baseline framing and old quiet-window assumptions |
| `S5.1` to `S5.5` | Hard invalidation | Fit diagnostics, residuals, parameter stability, and quiet-window scans all assume the old estimator and old interpretation of near-zero `theta` |
| `S6.1.2` | Hard invalidation | Empirical interpretation of dose-specific `theta` patterns is tightly coupled to the old estimator and old notion that vaccinated `theta` collapses to zero |

### Supplement: mostly retain with reframing

| Section in `supplement.md` | Severity | Why it can mostly survive |
|---|---|---|
| `S1` | Soft reframing | High-level SI overview can survive once section references are updated |
| `S3.1` | Soft reframing | Injected-effect logic is still usable |
| `S6.1.1` | Soft reframing | Czech registry context remains useful as motivation and application context |

## Notation diff

| Symbol / object | Current manuscript usage | Required `v7.5` usage | Update note |
|---|---|---|---|
| `theta` | Cohort frailty variance, often read operationally as quiet-window-fitted curvature | Replace with enrollment-time frailty variance `theta0` at rebased `t=0` after skip | Use `theta0` in prose and equations everywhere in the revised manuscript |
| `t` | Weeks since enrollment, with skip exclusion | Rebased time `t_rebased = t_raw - SKIP_WEEKS` for frailty fitting | Must define this once and use consistently |
| `k` | Constant quiet-window baseline hazard level | Gompertz scale parameter for `k e^{gamma t}` | Its interpretation changes materially |
| `H_obs` | Observed cumulative hazard from skipped trajectory | Full cumulative hazard accumulated from skip-adjusted hazard, with optional NPH correction applied first when relevant | Must state that wave periods remain in the accumulated trajectory |
| `\\tilde H_0` | Depletion-neutralized baseline cumulative hazard after inversion | Same | Carryover symbol |
| `H_gom` | Not present | Gompertz cumulative hazard during quiet periods | New operational object |
| `H_0^{eff}` | Not present | Reconstructed effective individual-level cumulative hazard across all weeks | New operational object |
| `delta_i`, `Delta(t)` | Not present | Wave-offset increments and accumulated offsets | New operational objects required for estimator description |
| `KCOR(t)` | Ratio of normalized cumulative hazards | Same ratio, but with explicit reference-window anchoring convention in applied plots | Clarify unanchored versus anchored forms |

## Validation-framing changes

The current manuscript's Section 3 is centered on the old question: does gamma-frailty normalization remove quiet-window curvature and prevent spurious drift under a synthetic null?

`v7.5` changes the validation target to a broader and more specific set of questions:

1. Can the method recover enrollment-time `theta0` under the Gompertz-plus-depletion model?
2. Does delta iteration correctly account for permanent wave offsets?
3. Does pooled use of all quiet windows improve identification without contaminating the fit?
4. Does end-to-end KCOR remain flat in negative controls after the full preprocessing pipeline, including the optional NPH module when used?
5. Do positive controls and empirical applications still behave coherently under the revised estimator?

Implication for manuscript structure:

- `Section 3` should be rewritten as a validation story for the new estimator, not merely a replay of the old controls
- the `synthetic null` remains useful, but it is no longer sufficient as the primary validation anchor
- the `theta0` simulator and multi-window identification story likely need promotion from the spec into the main manuscript or at least a visible SI-supported claim
- the current SI quiet-window perturbation narrative should no longer be treated as sufficient validation on its own

## Reviewer attack surface and claim audit

### Claims that should be downgraded or rewritten carefully

- Avoid saying KCOR is applied only in quiet intervals if the updated method distinguishes quiet-window identification from optional epidemic-wave preprocessing.
- Avoid implying that near-zero vaccinated `theta` is a general empirical law; `v7.5` explicitly changes that expectation.
- Avoid framing flat negative-control curves as proof that all confounding has been eliminated.
- Avoid saying the method "solves HVE" without separating dynamic HVE, static HVE, and the optional NPH extension.
- Avoid implying that the new estimator is universally identified; `v7.5` still has explicit low-curvature / low-mortality failure regimes.

### Reviewer questions the revised paper must anticipate

- Why is fixed Gompertz `gamma` justified, and how sensitive are results to that choice?
- Why is `theta0` identified from a nearest-window seed plus later pooled windows rather than from one quiet window?
- Under what conditions is delta iteration applicable, and what happens when `delta <= 0`?
- Why is additive persistent `delta_i` structure in cumulative-hazard space a reasonable assumption?
- Why is the optional NPH correction applied before inversion, and how dependent is it on the assumed `R` value?
- How are quiet windows prespecified, and how much room remains for analyst discretion?
- How is overfitting avoided when the new method uses Gompertz structure, iterative fitting, and multiple windows?
- What aspects of the pipeline are COVID-specific versus general KCOR machinery?

### Safe framing direction

- Present `v7.5` as a more explicit and internally coherent identification strategy, not as proof of complete correction.
- Emphasize that gamma inversion and KCOR remain working-model-based and diagnostic-gated.
- Emphasize that NPH correction is an optional, context-specific, assumption-bearing extension rather than a universal KCOR step.
- Emphasize that failure modes are still surfaced explicitly through identifiability checks and diagnostics.

## Unchanged carryover components

These pieces can be retained conceptually, though their surrounding prose will still need cleanup:

- fixed cohort design
- discrete-time hazard construction from aggregated deaths and risk sets
- gamma frailty as a working model
- exact gamma inversion
- KCOR as a cumulative ratio on the normalized scale
- diagnostic-first and non-causal framing at a high level

## Editorial decisions locked before manuscript editing

The following manuscript-architecture choices are now fixed for the update plan:

- Use `theta0` in prose and equations everywhere in the revised manuscript.
- Treat NPH as an optional epidemic-wave extension module, not as a universal KCOR step.
- Put the delta-iteration estimator steps in the main text.
- Put derivation details, convergence notes, delta edge cases, and fallback rules in the SI.

Remaining wording risks to resolve during manuscript editing:

- Decide whether the main manuscript should present the `theta0` simulation directly or summarize it with SI support.
- Decide how to present anchored KCOR versus unanchored KCOR so the estimand remains clear.
- Decide how aggressively to revise the Czech empirical SI, because the old `theta` interpretation there is deeply coupled to the superseded estimator.

## Update-planning input

This memo should be used as the input artifact for the manuscript-update plan. The next plan should distinguish three work types:

1. Hard rewrites
   `paper.md` `Abstract`, `1.2`, `1.5`, `1.6`, `2.4` to `2.5`, `2.9` to `2.11`, `3`, `4.1`, `4.4`, `5.0` to `5.4`, `6`, plus `supplement.md` `S2`, `S4.0` to `S5.5`, and `S6.1.2`.
2. Soft reframes
   `paper.md` `2.1` to `2.3`, `2.6` to `2.8`, `4.2` to `4.3`, and selected SI overview/context sections.
3. Carryover concepts
   fixed cohorts, gamma inversion, KCOR ratio, and diagnostic-first non-causal positioning.

Additional architecture rules for the next phase:

- Use `theta0` in prose and equations everywhere.
- Keep the seed fit, `H_0^{eff}` reconstruction, `delta_i` computation, and pooled `theta0` refit in the main text.
- Move derivations, convergence details, and fallback-edge-case logic to the SI.
- Present NPH as an optional epidemic-wave extension module in Methods, not as a universal KCOR step.

The key editorial rule for the next phase is: update model, estimator, interpretation, notation, validation, and claims together. Updating equations without updating the narrative will leave the paper internally inconsistent.

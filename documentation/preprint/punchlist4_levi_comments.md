Below are concrete, copy/paste-ready edits that directly address Retsef’s points (literature, identifiability vs confounding, “tail-sampling” selection, need for stronger simulations, and a crisp definition/diagnostic for “dynamic HVE”). I’m using your current section titles and the text I see in `paper.md`. 

---

## A) Abstract edits (keep under 250 words)

### A1. Add one sentence that explicitly limits causal claims + frames identifiability

**Location:** Abstract, after the sentence ending “…via ratios.”

**Replace/insert:**

* Insert this sentence (1 sentence, compact):
  “Because selection and treatment can be confounded in observational data, KCOR is presented as a depletion-normalization and diagnostic framework; causal interpretation requires additional assumptions and is evaluated via prespecified control tests and simulations.”

### A2. Add a “self-check” sentence (internal diagnostic)

**Location:** Abstract, replace the last two sentences starting with “Importantly…” through “…curvature.”

**Replacement text (2 sentences):**
“After frailty neutralization, the depletion-neutralized cumulative hazards are expected to be approximately linear during epidemiologically quiet periods; departures from linearity serve as a built-in diagnostic of assumption violation or quiet-window contamination. KCOR therefore enables interpretable cumulative cohort comparisons in settings where treated and untreated hazards are non-proportional because selection induces different depletion dynamics, while also providing explicit failure-mode signals when curvature cannot be explained by the depletion model.”

(That adds the “self-check” without overclaiming “proof.”)

---

## B) Add a dedicated section: “Built-in diagnostics (‘self-check’) and what failure looks like”

Retsef’s core worry is: “gamma fits don’t guarantee you’ve separated selection from treatment.” Your strongest response is to formalize: (i) what *must* be true for KCOR to claim “depletion removed,” (ii) what the diagnostics look like when it’s not true, and (iii) why this is materially different from standard pipelines.

### B1. New subsection in Methods

**Location:** Methods §2, immediately after **§2.6 Normalization (depletion-neutralized cumulative hazards)** and before **§2.7 Stabilization**.

**Add new subsection title and text:**

**“### 2.6.1 Internal diagnostics and ‘self-check’ behavior”**

**Text to insert:**
“KCOR includes internal diagnostics intended to make model stress visible rather than hidden.

1. **Post-normalization linearity in quiet periods.** During a prespecified quiet window, the working model assumes that curvature in observed cumulative hazard is primarily driven by depletion under heterogeneity. After inversion, the depletion-neutralized cumulative hazard $~H_{0,d}(t)$ should be approximately linear in event time over the same quiet window. Systematic residual curvature (e.g., sustained concavity/convexity) indicates that the quiet-window assumption is violated (external shocks, secular trends) or that the depletion geometry is misspecified for that cohort.

2. **Fit residual structure in cumulative-hazard space.** Define residuals $r_{d}(t)=H_d^{\mathrm{obs}}(t)-H_{d}^{\mathrm{model}}(t;\hat k_d,\hat\theta_d)$ over the fit set $\mathcal{T}_d$. KCOR expects residuals to be small and not systematically time-structured. Strongly patterned residuals indicate that the curvature attributed to depletion is instead being driven by unmodeled time-varying hazards.

3. **Parameter stability to window perturbations.** Under valid quiet-window selection, $(\hat k_d,\hat\theta_d)$ should be stable to small perturbations of the quiet-window boundaries (e.g., ±4 weeks). Large changes in $\hat\theta_d$ under small boundary shifts signal that the fitted curvature is sensitive to transient dynamics rather than stable depletion.

4. **Non-identifiability manifests as $\hat\theta\rightarrow 0$.** When the observed cumulative hazard is near-linear (weak curvature) or events are sparse, $\theta$ is weakly identified. In such cases, KCOR should be interpreted primarily as a diagnostic (limited evidence of detectable depletion curvature) rather than a strong correction.

These diagnostics are reported alongside KCOR curves. Importantly, the goal is not to assert that a single parametric form is always correct, but to ensure that when the form is incorrect or the window is contaminated, the method signals this explicitly rather than silently producing a misleading ‘corrected’ estimate.”

This reframes “self-check” as explicit, reviewer-friendly diagnostics, not “guaranteed truth.”

---

## C) Clarify identifiability: what KCOR can and cannot separate (address the confounding critique head-on)

Retsef: “selection and treatment can be confounded; you need more extensive experiments.” You should explicitly state a narrower, defensible claim: KCOR removes **one specific mechanism** (depletion under heterogeneity) and then *tests* whether residual time-patterns are consistent with (a) an acute selection artifact, (b) a stable treatment effect, or (c) window contamination.

### C1. Add a paragraph under Assumptions / Identifiability

**Location:** §2.1.1 “Assumptions and identifiability conditions” (end of that subsection).

**Add:**
“**Selection vs treatment is not generically identifiable from two cohort hazard curves without additional structure.** KCOR does not claim universal identifiability. Instead, it targets a specific, testable confounding structure: curvature induced by depletion under time-invariant multiplicative heterogeneity. When this structure adequately explains curvature in quiet periods (as indicated by fit diagnostics and post-normalization linearity), KCOR removes that component and compares cohorts in depletion-neutralized space. Residual differences may reflect treatment effects, residual selection not captured by the model, or time-varying external hazards; therefore, KCOR’s inferential workflow requires prespecified control tests and simulation-based operating-characteristic checks (Section 3), rather than assuming that normalization alone proves causality.”

That sentence alone will make a reviewer’s “overclaiming” alarm quiet down.

---

## D) Define “dynamic HVE” precisely and operationalize it

Retsef: “Not sure what you mean by dynamic HVE.” You need a tight definition plus an observable signature and a prespecified test.

### D1. Add a definition where HVE is first discussed

**Location:** Introduction §1.2 or §1.3 (whichever is your first mention of HVE; currently §1.3 has HVE motivation).

**Insert after the paragraph ending “making residual confounding easy to diagnose using control outcomes…”**

**Add:**
“In this paper we distinguish two mechanisms often lumped as the ‘healthy vaccinee effect’ (HVE):

* **Static HVE:** baseline differences in latent frailty distributions at cohort entry (e.g., vaccinated cohorts are healthier on average). In the KCOR framework, this manifests as differing depletion curvature (different $\theta_d$) and is the primary target of frailty normalization.

* **Dynamic HVE:** short-horizon, time-local selection processes around enrollment that create transient hazard suppression immediately after enrollment (e.g., deferral of vaccination during acute illness, administrative timing, or short-term behavioral/health-seeking changes). Dynamic HVE is operationally addressed by prespecifying a skip/stabilization window (§2.7) and can be evaluated empirically by comparing early-period signatures across related cohorts in multi-dose settings.”

### D2. Add a prespecified “dynamic HVE signature test” to Validation

**Location:** Validation §3.1 (negative controls) OR create a new subsection **§3.5 Dynamic HVE diagnostic tests (empirical and simulated)** near the end of Section 3.

**Add subsection text:**
“**### 3.5 Dynamic HVE diagnostic tests**

Dynamic HVE refers to transient hazard suppression immediately after enrollment driven by short-horizon selection around intervention timing (e.g., deferral during illness). It produces a characteristic early-time pattern: an abrupt early reduction in observed hazard that decays over several weeks and is not explained by stable depletion curvature.

**Empirical signature in multi-dose settings (diagnostic, not proof).** When multiple ‘treatment intensities’ exist (e.g., dose-2 and dose-3 cohorts defined at enrollment), dynamic HVE should affect adjacent-dose cohorts similarly at early times because both enrollments are subject to the same short-horizon deferral mechanisms. Therefore, if early post-enrollment curvature is dominated by dynamic HVE, then early-time deviations in KCOR(t) versus the same comparator should show similar transient shapes across adjacent-dose cohorts. Conversely, if early-time behavior differs substantially across adjacent-dose cohorts while post-normalization quiet-window linearity holds, it is less consistent with a single shared dynamic deferral artifact.

**Simulation check.** We include simulations where a transient early hazard suppression is injected around enrollment (multiplying hazard by factor $q<1$ for weeks 0–S), separately from gamma frailty depletion, and confirm that (i) the effect is attenuated/removed by prespecified skip weeks, and (ii) remaining KCOR trajectories in later windows behave as expected under negative and positive controls.”

This is the reviewer-friendly version of “mirror images” without sounding like you’re asserting ground truth.

---

## E) Expand simulations to cover the “two tails” / pathological selection scenario (Retsef’s #3)

He gave a concrete adversarial selection: vaccinated from the middle, unvaccinated from extreme tails. You should implement it in the simulation grid *and* state what you expect KCOR to do (work, partially work, or flag via diagnostics).

### E1. Add a new simulation scenario in §3.4 Simulation grid

**Location:** §3.4, after you list existing stresses (“non-gamma frailty, contamination…, sparse events.”)

**Insert an explicit new bullet and short description:**
“- **Tail-sampling / bimodal selection:** cohorts drawn from different parts of the same underlying frailty distribution (e.g., vaccinated sampled from mid-quantiles; unvaccinated from low+high tails), producing non-gamma mixture geometry at the cohort level.”

### E2. Add a paragraph stating what success/failure looks like

**Location:** §3.4, after the first paragraph describing the grid.

**Add:**
“This scenario is included because it can confound frailty-driven depletion with cohort construction in ways not captured by a single gamma frailty distribution. The goal is not to force KCOR to ‘succeed’ under arbitrary misspecification, but to quantify operating characteristics: when the gamma depletion model is misspecified, KCOR should either (i) remain approximately unbiased in later windows (if the misspecification is mild in cumulative-hazard geometry), or (ii) visibly degrade via its diagnostics (poor $H$-space fit, post-normalization nonlinearity, parameter instability), flagging that depletion-neutralization is unreliable without model generalization.”

### E3. Add an appendix-level spec so it’s reproducible

**Location:** Appendix B (Control-test specifications), add **B.5 Tail-sampling / bimodal selection** after B.4.

**Insert:**
“#### B.5 Tail-sampling / bimodal selection (adversarial selection geometry)

We generate a base frailty population distribution with mean 1. Cohort construction differs by selection rule:

* Mid-sampled cohort: frailty restricted to central quantiles (e.g., 25th–75th percentile) and renormalized to mean 1.
* Tail-sampled cohort: mixture of low and high tails (e.g., 0–15th and 85th–100th percentiles) with mixture weights chosen to yield mean 1.

Both cohorts share the same baseline hazard $h_0(t)$ and no treatment effect (negative-control version). We also generate positive-control versions by applying a known hazard multiplier in a prespecified window. We evaluate (i) KCOR drift, (ii) quiet-window fit RMSE, (iii) post-normalization linearity, and (iv) parameter stability under window perturbation.”

This directly answers the reviewer’s “what if” with “we tested it, and here’s what happens.”

---

## F) Literature: add a short “Related work” subsection + tighten the claim you made to him

Your email line “methods papers cite 1–2 references” is not going to help in peer review. Fix the manuscript instead: add a small, non-exhaustive related work section that signals you know the frailty/depletion literature, negative controls, and selection bias in VE studies, without trying to write a full review.

### F1. Add a subsection “Relationship to prior frailty and depletion literature”

**Location:** Introduction, after §1.2 (or after §1.3 if you prefer). Create a new subsection:

**“### 1.3 Related work: frailty, depletion of susceptibles, and selection-induced non-proportional hazards”**
(Then renumber the current §1.3 to §1.4 etc.)

**Text to insert (short, 2–3 paragraphs):**
“KCOR builds on a long literature on unobserved heterogeneity (‘frailty’) and depletion of susceptibles, in which population-level hazards can decelerate over time even when individual hazards are simple. The gamma frailty model is widely used because its Laplace transform yields a closed-form relationship between baseline and observed survival/cumulative hazard, enabling tractable inference and interpretation [@vaupel1979].

A separate literature emphasizes that observational estimates of vaccine effectiveness can remain confounded despite extensive matching and adjustment, often revealed by negative control outcomes and time-varying non-COVID mortality differences [@obel2024; @chemaitelly2025]. KCOR is complementary: rather than using negative controls only to detect confounding, it targets a specific confounding geometry—selection-induced depletion curvature—and then requires controls and simulations to validate that the intended curvature component has been removed.

We do not claim that KCOR subsumes all approaches to confounding adjustment; rather, it provides a dedicated normalization and diagnostic toolkit for settings where non-proportional hazards arise primarily from selection-induced depletion dynamics.”

This gives reviewers what they want: “you know the field; you’re not reinventing it; here’s the delta.”

(You can add 3–8 more citations later without changing the narrative.)

---

## G) Tone down any “we can fully disambiguate” language and replace with an auditable claim

Retsef is reacting to overconfidence. Make the paper say: “KCOR + controls + diagnostics can *rule out* some explanations and *support* others,” not “prove.”

### G1. Edit “This linearization serves as an internal diagnostic indicating that frailty-driven curvature has been effectively removed.”

**Location:** §3.0 “Frailty normalization behavior under empirical validation”

**Replace that sentence with:**
“This linearization is a diagnostic consistent with successful removal of depletion-driven curvature under the working model; persistent nonlinearity or parameter instability indicates model stress or quiet-window contamination.”

---

## H) Add a short “what KCOR does NOT justify” paragraph in Discussion (preempt the “killed more than saved” leap)

Even though this manuscript is “methods-only,” you should explicitly say it does not, by itself, imply net harm/benefit in any specific intervention without the applied paper + additional assumptions.

### H1. Add to Discussion §4.1 “What KCOR estimates”

**Location:** End of §4.1.

**Add:**
“Because the normalization targets selection-induced depletion curvature, KCOR results alone do not justify claims about net lives saved or lost by a particular intervention. Such claims require (i) clearly specified causal estimands, (ii) validated control outcomes, (iii) sensitivity analyses for remaining time-varying selection mechanisms and external shocks, and (iv) preferably replication across settings and outcomes. Accordingly, this manuscript focuses on method definition, diagnostics, and operating characteristics; applied causal conclusions are deferred to separate intervention-specific analyses.”

This directly neutralizes Retsef’s “I don’t see how you can claim…” concern.

---

## I) Add a “practical comparisons” paragraph without overpromising hidden data

Retsef suggested: Pfizer/Moderna trials, Klalit, etc. You can respond in-paper: RCTs underpowered + no record-level; observational datasets rarely share. But don’t sound like “impossible”; sound like “future work / where feasible.”

### I1. Add to Limitations §5

**Location:** End of §5 (or under a new §5.3 “Data requirements and external validation”).

**Add:**
“**External validation across interventions.** A natural next step is to apply KCOR to other vaccines and interventions where large-scale individual-level event timing data are available. Many RCTs are underpowered for all-cause mortality and typically do not provide record-level timing needed for KCOR-style hazard-space normalization, while large observational studies often publish only aggregated effect estimates. Where sufficiently detailed time-to-event data exist (registries, integrated health systems, or open individual-level datasets), cross-intervention comparisons can help characterize how often selection-induced depletion dominates observed hazard curvature and how frequently post-normalization trajectories remain stable under negative controls.”

---

## J) A small but important wording change: don’t call it “proof”; call it “diagnostic evidence”

Scan and replace a few high-risk phrases (you can do this quickly in Cursor search/replace).

* Replace “prove” → “provide diagnostic evidence” / “support”
* Replace “jaw dropping” / “stunning” (in the paper) → “pronounced” / “large” / “clear”
* Replace “killer verification step” → “strong internal diagnostic”

This aligns with methods-journal tone and removes reviewer triggers.

---

## The minimum set of changes if you want the fastest win

If you do nothing else, do these five:

1. Abstract: add the causal-limits sentence + self-check sentence (A1, A2)
2. Add §2.6.1 “Internal diagnostics and self-check behavior” (B1)
3. Add the identifiability paragraph in §2.1.1 (C1)
4. Add tail-sampling/bimodal selection to simulation grid + appendix spec (E1–E3)
5. Add the “does not justify net lives saved/lost” paragraph in §4.1 (H1)

Those directly answer every one of Retsef’s reviewer-style objections.

If you want, paste me your current Abstract word count after you drop in A1/A2, and I’ll tighten phrasing to stay comfortably under 250 without losing meaning.


# KCOR v6 Methods Paper – Reviewer Comments

## Overall assessment
This is a strong, unusually clear methods paper. The conceptual motivation (selection-induced curvature via frailty depletion) is well articulated, and the gamma-frailty inversion in cumulative-hazard space is elegant and mathematically sound. The paper is already at a level where it could plausibly survive a serious statistical methods review, provided a few clarifications and defensive additions are made.

Below are **actionable, section-by-section comments**, prioritized by impact on peer review risk.

---

## Major comments (high priority)

### 1. Explicitly justify gamma frailty (anticipate reviewer skepticism)
**Issue:** Some reviewers will object to gamma frailty as “convenient” or “assumed.”  
**Recommendation:** Add a short paragraph (Intro or Appendix A) explaining *why* gamma is the natural conjugate choice:
- Closure under aggregation
- Analytical tractability of the Laplace transform
- Longstanding use in survival analysis to model unobserved heterogeneity
- Emphasize that KCOR does **not** claim gamma is “true,” only that it is a *depletion-neutralizing approximation* validated by controls

Suggested sentence:
> “Gamma frailty is used not as a claim of biological truth, but as a mathematically minimal and empirically testable model for selection-induced depletion, whose adequacy is assessed via prespecified negative controls.”

This reframes the assumption as *testable*, not dogmatic.

---

### 2. Clarify identifiability of (k, θ)
**Issue:** A mathematically trained reviewer may ask whether k and θ are identifiable over short quiet windows.
**Recommendation:** Add a short clarification in §2.5:
- Identifiability comes from curvature in cumulative-hazard space
- When θ → 0, the model collapses to linear cumulative hazard
- Quiet windows are chosen to maximize curvature signal from depletion while minimizing external shocks

Optional sentence:
> “In practice, identifiability is assessed via curvature in H_obs(t); flat cumulative hazards naturally drive θ̂ → 0.”

---

### 3. Stronger defense of cumulative-hazard least squares vs MLE
**Issue:** Some reviewers will reflexively expect maximum likelihood.
**Recommendation:** Add a short justification:
- Discrete-time aggregated data
- Numerical stability under sparse events
- Emphasis on shape-fitting rather than likelihood optimality
- Equivalence to MLE under Gaussian error in H-space as bin width shrinks

This preempts a common but shallow objection.

---

### 4. Explicit failure modes section (recommended)
**Issue:** Reviewers trust methods more when authors acknowledge failure.
**Recommendation:** Add a short subsection (either §5 or Appendix):
- Mis-specified quiet window
- External time-varying hazards masquerading as frailty
- Extremely sparse cohorts
- Non-frailty-driven curvature (e.g., administrative censoring artifacts)

You already hint at this—formalizing it will *increase* credibility.

---

## Moderate comments (medium priority)

### 5. Add one intuitive schematic figure
**Suggestion:** A simple 3-panel schematic would help:
1. Individual hazards with frailty
2. Cohort hazard curvature before normalization
3. Flattened cumulative hazards after inversion

This can be synthetic and unlabeled—journals like intuition aids.

---

### 6. Tighten language around “causal” interpretation
**Issue:** You are careful, but some sentences may still trigger causal purists.
**Recommendation:** Replace phrases like:
- “interpretable as differences in baseline hazard”
with
- “interpretable *conditional on the stated selection model* as differences in baseline hazard.”

This reduces referee nitpicking.

---

### 7. Quiet-period selection: emphasize prespecification
**Issue:** Reviewers fear researcher degrees of freedom.
**Recommendation:** Add a sentence emphasizing:
- Quiet window is prespecified
- Same window used across cohorts
- Sensitivity analyses explicitly reported

You mostly do this already—just make it explicit and bold.

---

## Minor / editorial comments

### 8. Terminology consistency
- Consider consistently using **“selection-induced depletion”** rather than alternating with “selection dynamics”
- Define “curvature” once mathematically (second derivative of H(t))

### 9. Notation table placement
The notation table is good; consider moving it earlier in §2 so readers don’t have to infer symbols before seeing them.

### 10. Abstract: slightly shorten first sentence
The first abstract sentence is dense. Consider trimming ~10–15 words for accessibility.

---

## Strategic publication comments (not text edits)

### Target journal fit
This paper is best positioned for:
- *Statistics in Medicine*
- *Biometrical Journal*
- *Lifetime Data Analysis*
- *Epidemiology* (borderline, but possible if framed conservatively)

Avoid vaccine-specific journals for the methods paper.

### Reviewer psychology
This paper will **polarize** reviewers:
- Statisticians will appreciate the math
- Epidemiologists may resist the framing

Your strongest defense is:
- Heavy reliance on negative/positive controls
- Explicit admission of model dependence
- Clear separation of methods vs applied claims (which you already do well)

---

## Bottom line
**Technically strong, novel, and unusually honest.**  
With minor defensive additions (gamma justification, identifiability, failure modes), this is very close to submission-ready for a serious methods journal.

If you want, next steps I can help with:
- Writing a mock “Reviewer #2 hostile critique” and rebuttal
- Tightening the abstract for specific journals
- Generating the suggested schematic figure
- Turning this into a preprint-ready submission package

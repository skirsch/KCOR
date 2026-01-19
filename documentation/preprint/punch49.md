Great — below is a **Cursor-ready, end-to-end punch list** that incorporates **all accepted suggestions**, preserves your core structure, **keeps the Czech data in the main text**, and avoids unnecessary length inflation.

This is written so you can paste it directly into Cursor and execute section by section.

---

# Cursor Punch List — KCOR Revision for *Statistics in Medicine*

## Global constraints (do not violate)

* ❌ Do NOT remove Czech data from main text
* ❌ Do NOT weaken or hedge core KCOR claims
* ❌ Do NOT move core methods to SI
* ❌ Do NOT add new simulations
* ❌ Do NOT add causal language
* ✅ Prefer tightening / reframing / clarifying over cutting

Target net word reduction: **~500–800 words**, achieved via de-duplication only.

---

## PART A — Length reduction via de-duplication (low risk)

### A1. §1.2 Curvature discussion

**Action**

* Identify and remove **one redundant paragraph** that restates:

  * depletion of susceptibles
  * non-PH from frailty
* Keep:

  * the first clear conceptual explanation
  * one citation-anchored paragraph (Obel / Chemaitelly)

**Goal**

* Shorten §1.2 by ~150–200 words without losing conceptual clarity.

---

### A2. §4.2 “What KCOR estimates”

**Action**

* Remove repeated phrasing that duplicates:

  * Box 1 (conceptual contrast)
  * Box 2 (estimand & scope)

**Replace with**

* A single tight paragraph summarizing:

  * KCOR estimates cumulative outcome accumulation
  * It is descriptive, not causal
  * Interpretability depends on diagnostics

**Goal**

* Reduce by ~150 words.

---

### A3. §2.11 Cox comparison

**Action**

* Tighten prose (no content removal):

  * Remove rhetorical repetition
  * Keep figures and core logic
* Explicitly avoid re-explaining frailty from scratch

**Goal**

* Reduce by ~200 words.

---

## PART B — Positioning vs existing methods (accept reviewer request, minimal expansion)

### B1. Add short paragraph: time-varying Cox (in §1.3 or §2.11)

**Insert one paragraph (≤80 words):**

* Explicitly state:

  * time-varying coefficients address non-PH
  * but **do not address selection-induced depletion**
  * estimation remains conditional on survival

**Key sentence to include**

> Time-varying coefficients allow hazards to change over time but do not neutralize frailty-induced depletion because estimation remains conditional on survival.

---

### B2. Add short paragraph: marginal structural models (MSMs)

**Insert one paragraph (≤80 words):**

* Clarify:

  * MSMs target causal effects under exchangeability
  * KCOR targets descriptive cumulative contrasts
  * Different estimands, different assumptions, different failure modes

---

### B3. Add one sentence: flexible parametric models

**Single sentence**

> Flexible parametric survival models improve baseline fit but do not resolve depletion-induced selection bias when frailty heterogeneity is present.

---

## PART C — Quiet-window identifiability (important win)

### C1. Explicit failure rule (§2.1.3 or Box 1 cross-ref)

**Add sentence**

> If no candidate quiet window satisfies diagnostic criteria, KCOR is not interpretable and analysis should terminate without reporting contrasts.

---

### C2. Multiple quiet windows

**Add sentence**

> Multiple disjoint quiet windows may be pooled provided fitted depletion parameters are stable and diagnostics are consistent across windows.

---

### C3. Cross-reference diagnostics

* Add explicit pointer to diagnostics table/figure where quiet-window validity is assessed.

---

## PART D — Czech data framing (keep in main text, tighten language)

### D1. Reframe language (global search)

Replace any causal-leaning phrases such as:

* “selective uptake causes…”
* “vaccine effect”

With:

* “patterns consistent with selection”
* “illustrative of differential depletion”
* “demonstrates estimator behavior under real data”

---

### D2. Add explicit non-causal disclaimer (once, clearly)

**Insert one sentence in Czech section intro**

> This application is included solely to demonstrate estimator behavior and diagnostics under real registry data; no causal interpretation of vaccine effects is implied.

---

### D3. Keep tables but avoid interpretation creep

* Retain Tables S7–S8
* Remove or soften interpretive commentary that goes beyond:

  * selection
  * depletion
  * heterogeneity diagnostics

---

## PART E — Technical clarifications (accept all)

### E1. Eq. 2 hazard transform

**Clarify explicitly**

* State that:

  * the exact transform $h = -\log(1 - \mathrm{MR})$ is used throughout
  * the rare-event approximation is mentioned only for intuition

---

### E2. Bootstrap procedure (§2.9.1)

**Add one sentence**

* Explicitly state resampling unit, e.g.:

  * resample $(d_d(t), N_d(t))$ pairs by cohort-time with replacement
  * preserve within-cohort temporal structure

---

### E3. Anchoring consistency

**Action**

* Ensure consistent language:

  * clearly distinguish KCOR$(t)$ vs KCOR$(t; t_0)$:
    anchoring used only when explicitly stated
* Add one sentence explaining *why* anchoring is used when it appears.

---

## PART F — Figures (mandatory)

### F1. Regenerate Figure 4 and S8

**Action**

* Replace dashboard screenshots with publication-quality plots:

  * vector or high-resolution raster
  * consistent fonts, axes, legends
  * same color scheme as other figures

### F2. Caption tightening

* Ensure captions explicitly state:

  * negative control
  * pseudo-null construction
  * what constitutes “flat” behavior

---

## PART G — Discussion upgrades (small but valuable)

### G1. Emphasize conservative failure mode (§5.1)

**Add one highlighted sentence**

> Violations of identifiability assumptions lead to conservative behavior (e.g., $\hat{\theta} \to 0$) rather than spurious non-null contrasts.

---

### G2. Cause-specific mortality (brief)

**Add 1–2 sentences**

* State how KCOR could, in principle, extend to cause-specific outcomes
* Do NOT implement or simulate

---

### G3. Multiple testing

**Add short paragraph**

* Clarify:

  * KCOR is diagnostic-first
  * multiplicity concerns are mitigated by prespecification and controls
  * readers should treat multiple strata explorations descriptively

---

## PART H — Minor fixes

### H1. Notation table placement

* Add forward reference early (“see Table X for notation”)
  Do NOT move the table.

### H2. Reference date correction

* Fix Czech data reference year (2025 → 2024).

### H3. Code availability

* Ensure Zenodo DOI is final and consistent.

---

## Final Cursor instruction (single block)

> **Cursor instruction:**
> Apply targeted revisions meaningfully addressing reviewer feedback while preserving the core KCOR framework and keeping the Czech data in the main text. Reduce length by ~500–800 words via de-duplication (not removal). Add concise positioning vs time-varying Cox, MSMs, and flexible parametric models. Strengthen quiet-window identifiability rules and explicitly state failure conditions. Reframe Czech analysis as demonstrative only, with no causal claims. Clarify technical points (hazard transform, bootstrap resampling, anchoring). Regenerate low-quality figures. Emphasize conservative failure behavior in the Discussion. Do not add new simulations or move core methods to the SI.

---

If you want, next I can:

* turn this into a **commit-by-commit plan**, or
* generate **drop-in text** for the new comparison paragraphs so Cursor can paste verbatim.

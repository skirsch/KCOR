---
name: punch67 reviewer manuscript
overview: "Apply [documentation/preprint/punch67.md](documentation/preprint/punch67.md) to [documentation/preprint/paper.md](documentation/preprint/paper.md): one global pipeline sentence (§2.1 end or §2.7.1 start), estimand vs RMST in §2.1 / bridge §4.2, informal identification box **only** after §2.4.4, Gompertz minimal-structure phrase + SI γ, bootstrap variance framing, shared-frailty table + survival/marginal contrast, negative control under model-consistent conditions vs §3.3, α reframing with **H-space** domain for NPH correction, confounding §4.1, dedup §2.5–2.7, grep post-inversion NPH + H-space consistency, `make paper`."
todos:
  - id: p67-estimand-rmst
    content: Add KCOR vs RMST/model-free paragraph in §2.1 (+ optional bridge in §4.2)
    status: completed
  - id: p67-glue-pipeline
    content: One global pipeline sentence at end of §2.1 or start of §2.7.1 (exact wording in plan body)
    status: completed
  - id: p67-id-box
    content: Informal identification box + nonparametric disclaimer immediately after §2.4.4 only
    status: completed
  - id: p67-gompertz
    content: "§2.4.3: minimal-structure phrase + Gompertz anchor + SI γ sensitivity"
    status: completed
  - id: p67-bootstrap
    content: "§2.9: variance-propagation framing + future-work sentence on bootstrap theory"
    status: completed
  - id: p67-joint-frailty
    content: "§2.11: @tbl:joint_frailty_comparison + conditional-on-survival / marginal depletion contrast"
    status: completed
  - id: p67-neg-control
    content: "§3.1.2: scope vs §3.3; include model-consistent conditions phrase"
    status: completed
  - id: p67-alpha-reframe
    content: "§3.4 and/or §5.4: optional generalization + non-ID as diagnostic success"
    status: completed
  - id: p67-nph-hspace
    content: "Audit every NPH correction reference: applied to frailty-neutral cumulative hazard (H-space), not raw hazard"
    status: completed
  - id: p67-confounding
    content: "§4.1: paragraph on depletion vs general confounding, pre-causal diagnostic"
    status: completed
  - id: p67-tighten-make
    content: Light dedup §2.5–2.7; grep pre-inversion NPH + H-space wording; make paper
    status: completed
isProject: false
---

# Plan: Reviewer-facing revisions (punch67)

## Scope and non-goals

- **In scope:** Text-only additions and cross-references in the main manuscript; optional one-line supplement pointer if $\gamma$ sensitivity is only named but not cited by section.
- **Out of scope (per punch67):** Formal identification theorems, new datasets, expanding the $\alpha$ empirical section, or causal claims.

## 1. Estimand motivation vs RMST / model-free summaries

**Goal:** Reposition $\mathrm{KCOR}(t)$ as a **geometry-consistent descriptive contrast** under the working model, not a universal estimand.

**Primary placement (choose one for flow):**

- **Option A (Methods):** After the opening paragraphs of [§2.1 Conceptual framework and estimand](documentation/preprint/paper.md), before or after the bullet list on level vs curvature—add a short paragraph contrasting KCOR with **RMST / cumulative incidence** style summaries (no removal of depletion curvature; KCOR targets settings where curvature is attributed to selection-induced depletion).

- **Option B (Discussion):** At the start of [§4.2 What KCOR estimates](documentation/preprint/paper.md), same paragraph if you prefer estimand motivation next to “what it estimates.”

**Recommendation:** Use **§2.1** so Methods readers see the contrast early; add **one bridging sentence** in §4.2 pointing back to §2.1 to avoid duplication.

## 1b. Global pipeline sentence (single mental model)

**Placement (pick one for flow, not both):** End of [§2.1](documentation/preprint/paper.md) **or** start of [§2.7.1](documentation/preprint/paper.md) (before optional NPH detail).

**Exact sentence to add (tune notation only if `paper.md` uses different symbols for θ):**

> KCOR operates by first estimating depletion geometry ($\theta$) from quiet windows, then optionally adjusting wave-period amplification ($\alpha$), and finally comparing cumulative outcomes on the normalized scale.

**Note:** If §2.1 already states this after the 1–3 strategy list, do not duplicate in §2.7.1 unless Methods flow benefits from a one-line recap at the NPH subsection gate. When this sentence mentions optional $\alpha$, align with [§7b](#7b-nph-correction-domain-h-space): correction is on **frailty-neutral cumulative hazard** (H-space), not raw hazard—either fold that into the pipeline sentence or ensure the next NPH paragraph states it unambiguously.

## 2. Informal identification statement (boxed)

**Goal:** Satisfy “identification is informal” without a theorem.

**Placement (fixed):** Immediately after [§2.4.4 Quiet-window validity and identifiability](documentation/preprint/paper.md) (after the refinement block and diagnostic bridge). **Do not** place a second copy under §2.1.2; at most one forward pointer from §2.1 to §2.4.4 if helpful for readers.

**Format:** Match existing callout style (`> **Box N**` or `> **Identification (informal statement).**` blockquote) with bullets (i) common / ~common multiplicative scale, (ii) limited additive/time-varying contamination in quiet windows, (iii) sufficient curvature—aligned with current §2.4.4 language.

**Add one sentence** nearby: formal nonparametric identification is **not** claimed; identification is conditional on the working model and **diagnostic**.

## 3. Gompertz / fixed $\gamma$ justification

**Goal:** One compact paragraph defending fixed $\gamma$ as a **baseline trend anchor** and pointing to **SI sensitivity**.

**Placement:** [§2.4.3 Gompertz baseline and enrollment-time interpretation](documentation/preprint/paper.md)—the file already mentions SI $\gamma$ sensitivity; **extend** with punch67’s logic ($\theta_{0,d}$ from deviations; stability of KCOR / $\hat\theta_0$ over prespecified $\gamma$ range). Include the phrase **minimal structure sufficient to separate baseline growth from depletion curvature** (answers “why not something more flexible?”).

**Optional:** If SI has a concrete grid (e.g. $\pm 20\%$), add **one numeric clause** only if verifiable from [documentation/preprint/supplement.md](documentation/preprint/supplement.md) or repo text; otherwise keep “prespecified range” wording.

## 4. Bootstrap: variance propagation + scope limit

**Goal:** Preempt “nonstandard bootstrap / undercoverage” as fatal.

**Placement:**

- Opening of [§2.9 Uncertainty quantification](documentation/preprint/paper.md): state bootstrap as **variance propagation for the aggregated counting-process pipeline**, not a fully calibrated inferential procedure under arbitrary misspecification.

- After the stratified bootstrap steps or adjacent to existing Table @tbl:bootstrap_coverage discussion: reinforce that **sub-nominal coverage** aligns with **failed diagnostics** (already in §5.x); when diagnostics pass, coverage is **near nominal** (already stated—tighten wording if redundant).

- **One sentence** on future work: formal bootstrap theory for this aggregated construction is **out of scope**.

## 5. Shared-frailty Cox comparison (existing table)

**Goal:** “Not only naive Cox.”

**Placement:** [§2.11 Relationship to Cox proportional hazards](documentation/preprint/paper.md), in or immediately after **“Distinction from shared-frailty Cox models”** (~L649–650).

**Action:** Add an explicit cross-reference to **Table @tbl:joint_frailty_comparison** (already defined in `paper.md` ~L1157): shared-frailty Cox **reduces** but does not eliminate spurious non-null behavior under selection-only regimes; KCOR near-null vs detection when separable—**use table wording only as supported by that table’s caption.** Add that shared-frailty Cox **remains conditional on survival** and **does not remove depletion-induced selection effects from marginal comparisons** (sharpens contrast with KCOR’s cumulative normalized scale).

## 6. Negative control: scope vs robustness

**Placement:** [§3.1.2 Empirical negative control](documentation/preprint/paper.md), after the sentence that defines it as end-to-end behavior (~L758).

**Action:** Add **two sentences**: (1) purpose is **pipeline / normalization** behavior under strong composition shift **under model-consistent conditions**, not exhaustive robustness to all misspecifications; (2) **model stress** and misspecification are addressed in [§3.3 Stress tests](documentation/preprint/paper.md).

## 7. NPH $\alpha$ module: optional generalization + successful non-ID

**Placement:** Prefer [§3.4 Estimation of the NPH exponent](documentation/preprint/paper.md) intro (already notes illustrative Czech use) **and/or** opening of [§5.4 Optional NPH exponent model](documentation/preprint/paper.md).

**Action:** Add punch67 lines: module is a **generalization** validated synthetically; **empirical non-identification** (Czech) is a **successful diagnostic outcome**; when $\alpha$ is not identified, the module is **not used** for correction (consistent with post-inversion NPH narrative already in §2.7).

## 7b. NPH correction domain (H-space)

**Goal:** No reader infers that the optional NPH exponent acts on **raw** (instantaneous) hazard before or instead of the frailty-neutralized cumulative-hazard construction.

**Rule:** Every substantive reference to **NPH correction** / optional $\alpha$ adjustment must state explicitly that it applies to **frailty-neutral cumulative hazard** (match manuscript notation: H-space / post-inversion cumulative hazard / equivalent phrase already in Box 2 or §2.7), **not** to raw hazard $\lambda$ or unnormalized hazard-scale objects.

**Scope of pass:** [§2.7](documentation/preprint/paper.md) (all NPH subsections), [§3.4](documentation/preprint/paper.md), [§5.4](documentation/preprint/paper.md), abstract and key figure captions if they mention NPH correction, and any cross-references from Results that describe “applying $\alpha$.”

**Implementation:** Prefer consistent stock phrase (one sentence) reused where needed rather than long repetition; first mention in §2.7 can be definitive, later mentions can use “on $H$” shorthand only if $H$ is defined as frailty-neutral cumulative hazard in that section.

## 8. Confounding vs depletion (Discussion)

**Placement:** [§4.1 Limits of attribution and non-identifiability](documentation/preprint/paper.md) or new short paragraph at end of §4.1 before §4.2—**after** existing limits text.

**Action:** Explicit **pre-causal diagnostic layer** framing: KCOR targets **selection-induced depletion geometry** distorting marginal comparisons; it **does not** remove general confounding or replace causal methods; persistent contrasts after normalization are **not** automatically causal.

## 9. Minor Methods tightening (~5–10% repetition)

**Action:** Read-only pass on §2.5–§2.7 for **duplicate phrases** (e.g. repeated “optional NPH” clauses) and merge **without** changing technical meaning. No new sections.

## 10. Build

- Run `make paper` from repo root per [documentation/preprint/BUILD_README.md](documentation/preprint/BUILD_README.md).

## Verification

- New box/paragraphs **do not** contradict existing §2.4.4 / §2.7 / Box 2.
- Table @tbl:joint_frailty_comparison narrative matches table caption.
- Grep for over-claiming: e.g. “proves causality,” “formal identification theorem” (should not appear).
- Grep `paper.md` for **residual pre-inversion NPH** ordering (e.g. “prior to inversion,” “before frailty inversion,” legacy preprocessing language); wording should reflect **post-inversion** $\alpha$ where applicable.
- **NPH domain:** Read-through (and targeted grep for “NPH,” “non-proportional,” “$\alpha$,” “exponent”) so that **every** description of correction states **frailty-neutral cumulative hazard / H-space** (not raw hazard), unless the sentence is explicitly contrasting the two.

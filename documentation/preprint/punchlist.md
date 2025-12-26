# KCOR v6 Paper Review Punchlist (Targeted Improvements)

This punchlist reviews the current draft of `paper.md` and identifies **specific, high‑leverage refinements** to improve clarity, reviewer defensibility, and internal consistency. Overall assessment: **the paper is very strong**; the items below are mostly tightening, not reworking.

Source reviewed: `paper.md` fileciteturn1file0

---

## Executive Summary

**Status:**
- Conceptual framing is excellent
- Math is consistent and clean
- The key empirical insight (θ asymmetry + linearization) is already present and correct

**Primary remaining risks:**
- A few statements are slightly *stronger than necessary* for a methods paper
- Some redundancy could be tightened to reduce reviewer fatigue
- A couple of places would benefit from clearer “this is a diagnostic, not an assumption” phrasing

No major structural changes are required.

---

## 1. Abstract (Minor tightening)

**Issue:** The abstract is strong, but slightly long and dense near the end.

**Suggested action:**
- Keep all content, but split the final long sentence into two for readability.

**Current ending (excerpt):**
> After frailty neutralization, cumulative hazards are approximately linear, enabling direct cohort comparison free of selection-induced curvature. KCOR enables interpretable cumulative cohort comparisons in settings where treated and untreated hazards are non-proportional because selection induces different depletion dynamics.

**Suggested edit:**
> After frailty neutralization, cumulative hazards are approximately linear, enabling direct cohort comparison free of selection-induced curvature. KCOR therefore enables interpretable cumulative cohort comparisons in settings where treated and untreated hazards are non‑proportional because selection induces different depletion dynamics.

**Why:**
- Improves flow
- Avoids repetition of “enables” as a rhetorical crutch

---

## 2. Introduction §1.2 (Clarify diagnostic role of linearity)

**Issue:** Linearity is correctly described but could be framed more explicitly as a *diagnostic*.

**Insertion point:** End of §1.2, after the paragraph describing curvature and concavity.

**Insert (1 sentence):**
> Approximate linearity of cumulative hazard after adjustment is therefore not assumed, but serves as an internal diagnostic indicating that selection‑induced depletion has been successfully removed.

**Why:**
- Preempts reviewer confusion about “forcing” linearity

---

## 3. Methods §2.4.1 (Tone softening)

**Issue:** The phrase “depletion‑neutralizing approximation” is good but slightly assertive.

**Suggested action:**
- Add one qualifying clause emphasizing falsifiability.

**Insertion point:** End of the paragraph beginning “Gamma frailty is used not as a claim of biological truth…”

**Add:**
> Its adequacy is evaluated empirically through curvature fit diagnostics and control tests, rather than assumed a priori.

**Why:**
- Reinforces scientific modesty
- Helps with skeptical statistical reviewers

---

## 4. Methods §2.5 (Excellent; no changes required)

**Assessment:**
- This section is particularly strong
- The explanation of θ → 0 behavior is clear, correct, and well‑placed

**Action:** None

---

## 5. Methods §2.6 (Optional emphasis)

**Issue:** The mapping interpretation is correct but important enough to visually highlight.

**Optional action:**
- Consider adding a short emphasized sentence (italic or parenthetical).

**After:**
> This normalization maps each cohort into a depletion-neutralized baseline-hazard space…

**Add (optional):**
> *(Conceptually, this places all cohorts into an equivalent θ‑factored comparison space.)*

**Why:**
- Helps non‑technical readers
- Reinforces the geometric interpretation

---

## 6. Validation §3.0 (Very strong; one small addition)

**Issue:** The asymmetry result is clearly stated; one extra sentence could tie it back to HVE explicitly.

**Insertion point:** End of the first paragraph in §3.0.

**Insert:**
> This asymmetric pattern is a quantitative fingerprint of healthy‑vaccinee selection acting at cohort entry.

**Why:**
- Sharpens interpretation
- Makes the empirical point unmistakable

---

## 7. Discussion §4.1 (No substantive changes)

**Assessment:**
- This section is clear, restrained, and methodologically appropriate
- The mapping vs assumption distinction is well handled

**Action:** None

---

## 8. Limitations §5 (Optional reordering)

**Issue:** The strongest limitation (model dependence) might be better placed first.

**Optional action:**
- Swap the first two bullets so **model dependence** comes before θ‑estimation clarification.

**Why:**
- Matches typical reviewer priority ordering

---

## 9. Global Style Pass (Optional)

**Suggested global edits:**
- Replace a few instances of “demonstrates” → “is consistent with”
- Replace “confirms” → “supports” where used descriptively

**Why:**
- Slightly lowers rhetorical temperature
- Improves acceptance odds for a methods journal

---

## Bottom Line

**Overall judgment:**
- The paper is **submission‑ready** from a technical standpoint
- No major revisions are needed
- The θ‑asymmetry and linearization insights are already well integrated

If you apply only **3 changes**, do these:
1. Add the diagnostic‑linearity sentence in §1.2
2. Add the HVE fingerprint sentence in §3.0
3. Lightly soften wording in §2.4.1

If you want, I can:
- apply these edits directly to `paper.md`, or
- do a final “reviewer‑#2 stress test” pass focused purely on objections.


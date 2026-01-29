Got it — here’s a **Cursor-ready punchlist** focused on **Anand’s feedback**, scoped exactly the way you want it: **only items that are still missing, ambiguous, or could be tightened with low risk**. This assumes the *current* `paper.md` you pasted is the authoritative version.

I’m **not** re-litigating reviewer preferences, and I’m **not** asking Cursor to restructure the paper. This is a “clean up the last 5–10%” list.

---

## Cursor punchlist — Anand feedback follow-through

### ✅ A. Items that *are already properly addressed* (no action)

Cursor should **skip** these unless a regression is detected:

* Clear **non-causal scope** and estimand positioning (Box 2, §1.6, §2.1.2)
* Explicit **diagnostics-first framing** (quiet window, identifiability gates, failure = not reported)
* Explicit separation of **normalization vs comparison**
* Clear explanation of **why Cox fails under frailty** (synthetic null + estimand mismatch)
* Explicit statement that **θ→0 is data-driven**, not assumed
* Bootstrap coverage issue now **explicitly interpreted** (anti-conservative due to variance underestimation, tied to diagnostics)
* Negative controls framed as **estimator validation**, not causal evidence

These were core Anand concerns and they are now solid.

---

### ⚠️ B. Low-risk improvements still worth making

These are the remaining items where Anand’s instincts about “what readers look for” still apply.

#### 1. Add **one explicit “reader expectation” sentence early**

**Location:** end of §1.1 or start of §1.2

**Task:**
Add **one sentence** that explicitly tells the reader *what they will not get*.

**Cursor instruction:**

> Insert a single sentence clarifying that KCOR does not attempt to recover counterfactual survival curves or causal effects, and that readers seeking causal VE estimates should not expect them here.

This reduces reviewer misalignment risk at essentially zero cost.

---

#### 2. Make the **quiet-window identifiability failure rule more explicit**

Right now it’s clear, but slightly implicit.

**Location:** §2.1.2 or §2.4.4

**Task:**
Add a **one-line hard stop rule**.

**Cursor instruction:**

> Add one sentence stating explicitly that if no quiet window passes diagnostics, the analysis terminates without reporting KCOR curves or summary values.

This matches how careful method reviewers think.

---

#### 3. Tighten language where KCOR could be misread as “correction”

There are a few places where a hostile reader could still quote-mine.

**Locations to scan:**

* §2.6
* §3 lead-in
* §6 Conclusion

**Task:**
Replace any remaining instances of language that implies *correction* or *bias removal* with **“normalization” or “mapping”** if present.

**Cursor instruction:**

> Search for language implying “corrected effect” or “bias removal” and replace with depletion-neutralized normalization language where appropriate.

Low effort, high defensive value.

---

#### 4. One sentence linking **bootstrap under-coverage → diagnostics**

You already did 90% of this — just make the link even more mechanical.

**Location:** §5 (bootstrap coverage paragraph)

**Task:**
Add **one sentence** that explicitly states that **those regimes would not be reported in applied analyses**.

**Cursor instruction:**

> Add one sentence explicitly stating that scenarios with sub-nominal coverage correspond to analyses that would fail KCOR diagnostics and therefore not be reported in applied work.

This directly answers a reviewer question before it’s asked.

---

#### 5. Optional (very low risk): micro-signposting in Results §3

Some readers skim.

**Location:** start of §3

**Task:**
Add a **one-line roadmap**.

**Cursor instruction:**

> Add a single sentence at the start of Section 3 stating that negative controls test false positives, positive controls test power, and stress tests probe diagnostic failure modes.

This costs ~15 words and improves readability.

---

## Summary verdict (your actual question)

**Were all of Anand’s *substantive* requested changes made properly?**
→ **Yes.** The core concerns (estimand clarity, diagnostics, non-causality, reader expectations, Cox positioning) are now handled correctly and defensibly.

What remains are **presentation-layer hardening steps**, not methodological gaps.

If you hand this punchlist to Cursor, you’re basically polishing for a skeptical *Statistics in Medicine* or *Biostatistics* reviewer — not changing the paper’s substance.

If you want, next step I can:

* mark these directly against line numbers, or
* rank them by “reviewer payoff per word added”.

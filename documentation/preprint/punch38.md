Great question — and yes, **both points are right**:

1. the empirical negative control should **explicitly state that no gamma-frailty normalization is used**, and
2. the **main paper should point to the SI (S2/S3/S4)** for the control structure.

Below is a **Cursor-ready punch list with exact insert text and locations**, written so Cursor can apply it mechanically.

---

# Cursor Instruction Punch List

**Task:** Make explicit that the empirical negative control uses **no gamma-frailty normalization**, and ensure the **main paper references the SI controls correctly**.

---

## PART A — Supplement edits (make the distinction explicit)

### A1. Add framing sentence at the start of **S4.2 Negative controls**

**Location:**
In `supplement.md`, immediately after the header:

```
### S4.2 Negative controls
```

**Insert the following paragraph verbatim:**

> *Negative controls are implemented in two complementary forms. The empirical negative control uses full-population registry cohorts and does not apply gamma-frailty normalization, as selection-induced depletion is negligible by construction. The synthetic negative control introduces extreme, known frailty heterogeneity and explicitly tests whether gamma-frailty normalization correctly removes curvature under the null.*

---

### A2. Make “no frailty normalization” explicit in **S4.2.2 Empirical negative control**

**Location:**
In `supplement.md`, within:

```
#### S4.2.2 Empirical negative control: age-shift construction
```

Insert **after the paragraph ending with**:

> *“This construction ensures that dose comparisons are within the same underlying vaccination category, preserving a true null while inducing 10–20 year age differences.”*

**Insert the following paragraph verbatim:**

> *No gamma-frailty normalization is applied in this empirical negative-control construction. Because the cohorts represent full-population strata rather than selectively sampled subcohorts, frailty heterogeneity and depletion are minimal, and direct comparison is valid without normalization.*

---

### A3. Add explicit contrast sentence (optional but recommended)

**Location:**
At the **end** of the same subsection (S4.2.2), before the next section header.

**Insert:**

> *This contrasts with the synthetic negative control (Section S4.2.1), where strong, deliberately induced frailty heterogeneity requires gamma-frailty normalization to recover the null.*

This sentence makes the logic reviewer-proof.

---

## PART B — Main paper cross-references (yes, you should do this)

You are correct: **the main paper should explicitly point readers to the SI for controls**, otherwise reviewers will ask “where exactly are these validated?”

### B1. Add a single sentence in the **Methods / Validation overview** of `paper.md`

**Location:**
In `paper.md`, in the Methods section where KCOR validation or diagnostics are summarized (often near where you mention negative/positive controls).

**Insert the following sentence verbatim:**

> *Validation of KCOR is assessed using both negative and positive controls, including empirical registry-based nulls, synthetic frailty-stress tests, and injected-effect positive controls; full specifications and results are provided in Supplementary Information Sections S4.2–S4.3.*

This is sufficient and journal-safe.

---

### B2. (Optional but strong) Clarify that empirical NCs do *not* rely on frailty normalization

If you want the point surfaced in the main text (without overclaiming), add **one clause**.

**Replace the sentence above with:**

> *Validation of KCOR is assessed using both negative and positive controls, including empirical registry-based nulls that do not require frailty normalization, synthetic frailty-stress tests, and injected-effect positive controls; full specifications and results are provided in Supplementary Information Sections S4.2–S4.3.*

This reinforces your key insight without adding technical burden to the main paper.

---

## PART C — Final verification checklist (Cursor must confirm)

Cursor should verify:

* Exactly **one** paragraph at the start of **S4.2 Negative controls** explaining the two types
* **Explicit statement** in S4.2.2 that *no gamma-frailty normalization is applied*
* An explicit **contrast sentence** between empirical and synthetic NCs
* At least **one main-paper reference** pointing to SI Sections S4.2–S4.3
* No equations, figures, or results altered

---

## Why this is the right level of emphasis

* You are **not claiming** that frailty never matters
* You are showing that KCOR:

  * *does nothing when nothing is needed*, and
  * *works when extreme correction is required*

That is exactly the kind of conditional robustness reviewers look for.

If you want next, I can:

* check whether **figure captions** should echo this distinction, or
* help you craft a **one-sentence reviewer-response justification** for this design choice.

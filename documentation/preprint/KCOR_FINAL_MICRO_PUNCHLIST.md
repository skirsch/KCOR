# KCOR – Final Micro‑Punchlist (Text‑Exact Additions Only)

This punchlist contains **only two targeted, low‑risk edits**, with **exact text provided**.
Both changes improve clarity and reviewer robustness without expanding scope or altering results.

---

## 1. Insert a minimal θ_d definition (without touching the abstract)

**Recommendation: DO THIS**

**Location (preferred):**  
Table 2 caption (Definitions / Parameters table)

**Exact text to add (one clause):**
> “θ_d denotes the cohort‑specific depletion (frailty variance) parameter governing curvature in the observed cumulative hazard.”

This ensures:
- θ_d is defined **before first technical use**
- No abstract changes
- No new notation introduced

If Table 2 caption is unavailable, an acceptable fallback is the **first sentence in §2.3 or §2.4** where θ_d first appears.

---

## 2. Add one concise Limitations paragraph on non‑gamma / time‑varying frailty

**Recommendation: DO THIS (exactly one paragraph, no equations)**

**Location:**  
Discussion → Limitations subsection

**Exact paragraph text (copy/paste):**

> *The KCOR framework assumes that selection acts approximately multiplicatively through a time‑invariant frailty distribution, for which the gamma family provides a convenient and empirically testable approximation. In settings where depletion dynamics are driven by more complex mechanisms—such as time‑varying frailty variance, interacting risk factors, or shared frailty correlations within subgroups—the curvature structure exploited by KCOR may be misspecified. In such cases, KCOR diagnostics (e.g., poor curvature fit or unstable θ̂ estimates) serve as indicators of model inadequacy rather than targets for parameter tuning. Extending the framework to accommodate dynamic or correlated frailty structures would require explicit model generalization rather than modification of KCOR normalization steps and is left to future work.*

This paragraph:
- Explicitly acknowledges misspecification risk
- Does **not** introduce new models
- Does **not** promise results
- Frames diagnostics correctly
- Preempts reviewer overreach

---

## Explicitly do NOT do

- Do not add simulations
- Do not add new equations
- Do not expand causal discussion
- Do not rename the method or reframe scope

---

## Status after applying this punchlist

- Symbol clarity: **complete**
- Limitations coverage: **complete**
- Reviewer‑proofing: **maximized**
- Scope integrity: **unchanged**

---

## End of punchlist

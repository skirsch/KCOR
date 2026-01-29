Got it — here’s a **Cursor-ready punchlist**, written exactly as something you can hand to Cursor and let it execute mechanically. This is **scoped to low-risk, forward-moving edits only**, aligned with Anand’s feedback, and avoids conceptual rewrites.

---

## KCOR — Cursor Execution Punchlist (Anand Feedback)

### 1. Abstract & Framing

* Replace any use of **“unbiased”**, **“true hazard”**, or **“recovers the true effect”** with language emphasizing:

  * *neutralization of a specific, diagnosable selection bias*
  * *estimand alignment rather than causal recovery*
* Add a parenthetical definition of **“epidemiologically quiet period”** (e.g., “intervals of stable baseline risk”) on first use in the Abstract.

---

### 2. Terminology Alignment

* Where KCOR’s target quantity is described, replace or supplement with:

  * **“neutralized marginal estimand”**
* Ensure Cox comparisons are framed as:

  * *estimand mismatch under selection on frailty*, not estimator inefficiency.

---

### 3. Quiet-Period / Anchoring Clarifications

* On first mention of SKIP_WEEKS:

  * Add one sentence stating it is **pre-specified** and **diagnostic-driven**, not tuned to outcomes.
* In figures where cumulative hazards start after week >0:

  * Update captions to explicitly state:

    * *“Cumulative hazard indexed from week X following enrollment”*
* Ensure no figure caption can be read as “early risk is hidden”.

---

### 4. Cox Model Failure Explanation

* In the Cox comparison section:

  * Add one sentence explicitly stating that Cox failure is **structural (selection + non-proportional hazards)**, not a finite-sample issue.
* Emphasize that KCOR operates using **Nelson–Aalen–type cumulative hazards without individual-level frailty observables**.

---

### 5. Bootstrap / Uncertainty Language

* Clarify in Results or Methods that:

  * Bootstrap intervals reflect **sampling variability of the aggregated process**, not uncertainty in individual-level causal effects.
* Avoid language implying parameter-estimation uncertainty where none is modeled.

---

### 6. Cross-References to Simulations

* In the main text discussion of frailty robustness:

  * Add an explicit cross-reference to the **pathological/non-gamma frailty mixture simulation** in the Supplement.
* In the Conclusion:

  * Tie each major claim back to a specific validation class:

    * Synthetic null
    * Age-shift negative control
    * Cox failure under true null

---

### 7. Figure Caption Tightening

* Replace any language like:

  * “near-flat under the null”
  * with:
  * “expected to be horizontal under the null, subject to sampling stochasticity”
* Ensure schematic figures are explicitly labeled as **illustrative**, not empirical.

---

### 8. Tone & Claim Softening (Search-and-Replace Pass)

* Globally soften absolute claims:

  * “demonstrates” → “indicates”
  * “shows that X is real” → “suggests X is more likely to reflect real differences”
* Do **not** introduce new limitations sections; keep changes local.

---

### 9. No-Change Guardrails (Do NOT Modify)

* Do not alter:

  * KCOR estimator definition
  * Diagnostics logic
  * Simulation designs
  * Identifiability discussion depth
  * Gamma-frailty modeling assumptions
* Do not add new figures or analyses.

---

### 10. Final Consistency Check

* Verify that:

  * KCOR is described consistently as **descriptive / diagnostic-first**, not causal
  * Failure modes are framed as **detected and excluded**, not ignored
  * Cox is criticized for **estimand mismatch**, not incompetence

---

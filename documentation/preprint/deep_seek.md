Based on a thorough review, **this paper is very close to being ready for submission to *Statistics in Medicine***. It is a well-structured, rigorous, and clearly written methods paper that introduces a novel statistical framework (KCOR) for handling selection-induced non-proportional hazards in retrospective cohort studies.

Here is a detailed assessment, broken down by the criteria typically used by *Statistics in Medicine* and similar top-tier biostatistics journals:

---

### **Strengths (Why it is likely suitable for *Statistics in Medicine*)**

1.  **Clear Novelty and Contribution:** The paper addresses a recognized and important problem (selection-induced depletion confounding in observational survival analysis) with a novel solution. It clearly distinguishes KCOR from existing methods (Cox, frailty models, RMST, MSMs) in both its estimand (cumulative hazard ratio) and its approach (normalization before comparison).
2.  **Rigor and Completeness:**
    *   **Theory:** The gamma-frailty foundation is well-established, and the paper's inversion approach is clearly derived.
    *   **Validation:** Extensive validation via simulation studies (synthetic null, positive controls, sensitivity analyses, failure modes) is a major strength. The use of the ADEMP framework is excellent.
    *   **Diagnostics:** The emphasis on internal diagnostics (fit quality, post-normalization linearity, parameter stability) is a critical and sophisticated feature that aligns with modern statistical practice.
    *   **Reproducibility:** The provided repository, scripts, and data formats are a significant plus.
3.  **Clarity and Structure:** The manuscript is exceptionally well-organized. The flow from introduction/motivation → method definition → validation → discussion/limitations is logical. Tables and figures are informative and support the narrative.
4.  **Appropriate Scope for a Methods Paper:** The authors correctly frame this as a methods paper, using real data only for illustration and validation, not for drawing causal conclusions. They explicitly defer applied claims to companion papers, which is the right approach.

---

### **Areas for Final Polishing Before Submission**

While the paper is strong, these refinements will increase its chances of acceptance:

1.  **Abstract & Key Messages:**
    *   The abstract is dense. Consider slightly streamlining the methods description to make the **core innovation and its advantage** even more prominent in the first few sentences.
    *   Ensure the "Key Messages" box perfectly aligns with the journal's format (if required).

2.  **Clarify "Quiet Window" Selection:**
    *   This is the most critical and potentially subjective part of the method. Section 2.4.4 and the protocol are good, but you could strengthen it by adding a brief **practical example** in the main text (e.g., "For COVID-19 data, a quiet window might be defined as a period between variant waves, verified by stable all-cause mortality rates in the general population").
    *   Emphasize more strongly that **robustness to reasonable perturbations of this window is a core diagnostic** (which you already test in sensitivity analyses).

3.  **Strengthen the Comparison to Existing Methods (in text):**
    *   Table 3 is excellent. In the main text (Section 1.3.1), consider adding a **short, clear summary paragraph** that directly states: "KCOR differs from these methods not in flexibility, but in its *estimand* (cumulative vs. instantaneous contrast) and its *direction of inference* (normalize-then-compare vs. model-and-interpret-coefficient)."
    *   The point about RMST inheriting depletion bias (Section 1.3.2) is crucial and well-made.

4.  **Discussion & Limitations:**
    *   The limitations section is very honest and comprehensive. Consider adding a **short subsection on "Practical Guidelines for Implementation"** based on Section 4.2, which would help applied researchers. *Statistics in Medicine* values methods that can be used.
    *   Reiterate that KCOR is **not a causal identification strategy** but a **confounding normalization framework**, making its assumptions (A1-A5) and diagnostics paramount. This point is made but could be the central thesis of the first discussion paragraph.

5.  **Formatting and Typos:**
    *   Perform a final meticulous proofread. I noticed a few minor issues (e.g., "Chematically" vs. "Chemaitelly" in the reference list, occasional inconsistent equation referencing like `{#eq:hazard-from-mr-improved}`, a missing reference marker `[]` after the Obel quote in Section 1.4.1).
    *   Ensure all figures and tables are called out in the correct order in the text.
    *   Verify that the reference style matches *Statistics in Medicine*'s guidelines exactly.

---

### **Journal-Specific Fit**

*Statistics in Medicine* is a premier journal for **novel statistical methodology with direct medical/epidemiological application**. This paper fits perfectly:
*   **Methodology:** Introduces a new estimand and framework.
*   **Application:** Motivated by a high-impact problem (vaccine safety/effectiveness studies) with clear use in other retrospective cohort settings (elective surgery, drug safety).
*   **Rigor:** Comprehensive simulations and diagnostic procedures.
*   **Implementation:** Provides software, enhancing practical impact.

---

### **Recommendation**

**This paper is of high quality and is absolutely suitable for submission to *Statistics in Medicine*.**

**Action Plan:**
1.  **Address the minor polishing points** above, especially regarding the clarity of the quiet window selection and the central differentiation from other methods.
2.  **Perform a final, detailed proofread** for consistency, formatting, and typographical errors.
3.  **Check the author guidelines** for *Statistics in Medicine* (word limits, abstract structure, reference format, data/code availability statements) and make final adjustments.
4.  **Submit.** The manuscript makes a substantial contribution to the field of survival analysis for observational studies.

**Excellent work.** This is a sophisticated and well-executed methods paper that should be well-received by the statistical community.

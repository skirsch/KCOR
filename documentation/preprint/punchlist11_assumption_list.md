Below is a **Cursor-ready instruction block** you can paste directly into Cursor as a single task.
It is written to be **mechanical, unambiguous, and non-creative**, so Cursor does not “helpfully” drift.

---

## 🔧 Cursor Instructions — KCOR Paper Edits

### **Goal**

Fix the KCOR assumption count at **exactly five (A1–A5)**, make this list authoritative, and move all diagnostics / failure-mode material into a new **Appendix D** with **no new assumptions introduced**.

---

## **Part 1 — Modify Section 2.1.1**

### **Target section**

`§2.1.1 Assumptions and identifiability conditions`

---

### **REMOVE (entirely)**

Delete the existing bullet list and surrounding explanatory text that currently enumerates assumptions, including (but not limited to):

* “Fixed cohorts at enrollment”
* “Quiet-window validity”
* “Temporal separability”
* “Gamma-frailty as geometric approximation”
* “Identifiability via curvature”
* “Baseline hazard regularity”
* “Diagnostics required”

Remove **all assumption-like bullets**, even if they are phrased as “conditions” or “requirements”.

---

### **REPLACE WITH (verbatim content below)**

#### **New §2.1.1 text (authoritative)**

> **2.1.1 KCOR assumptions and identifiability**
>
> KCOR relies on **five explicit assumptions (A1–A5)**. These assumptions are stated exhaustively here; **no additional assumptions are introduced elsewhere in the manuscript**. Each assumption is either enforced by study design or empirically testable via prespecified diagnostics. When an assumption is violated, KCOR does not silently produce misleading estimates; instead, the violation manifests through identifiable diagnostic failures described in Appendix D.
>
> **A1. Fixed cohorts at enrollment.**
> Individuals are assigned to cohorts at enrollment and remain in those cohorts for the duration of follow-up. No post-enrollment switching or censoring is permitted in the primary estimand.
>
> **A2. Shared external hazard environment.**
> All cohorts are exposed to the same calendar-time external mortality environment (e.g., seasonality, epidemic waves, reporting artifacts). External shocks may occur but must affect cohorts symmetrically.
>
> **A3. Selection operates through time-invariant latent frailty.**
> Differences between cohorts at enrollment arise primarily through selection on latent, multiplicative frailty that remains constant over follow-up and induces depletion of susceptibles.
>
> **A4. Gamma frailty adequately approximates depletion geometry.**
> Gamma frailty provides a sufficient approximation to the cumulative-hazard curvature induced by frailty-driven depletion over the estimation window. This assumption is not asserted a priori and is evaluated empirically.
>
> **A5. Existence of a valid quiet window for frailty identification.**
> There exists a prespecified period during which selection-induced depletion dominates other sources of curvature, permitting identification of frailty parameters. This assumption is assessed empirically and is the primary dataset-specific requirement for KCOR applicability.
>
> Of these five assumptions, **A1–A3 are structural and are typically satisfied by cohort construction in national mortality datasets**, while **A4–A5 are empirically evaluated using diagnostics**. Failure of any assumption limits interpretability and is explicitly signaled by KCOR’s diagnostic outputs (Appendix D).

---

## **Part 2 — Create Appendix D**

### **Create a new appendix**

Insert **after the final appendix currently in the manuscript**.

---

### **Appendix title**

```
Appendix D — Diagnostics and Failure Modes for KCOR Assumptions
```

---

### **Appendix D content (verbatim structure; you may paraphrase but do NOT add assumptions)**

#### **D.1 Diagnostics for Assumption A1 (Fixed cohorts)**

* Describe how post-enrollment switching or censoring would manifest as inconsistencies in risk sets or early hazard distortions.
* State that KCOR is not defined for such data and should not be applied.

#### **D.2 Diagnostics for Assumption A2 (Shared external hazard environment)**

* Describe checks based on calendar-time alignment of hazard spikes across cohorts.
* Note that cohort-specific shocks indicate violation.

#### **D.3 Diagnostics for Assumption A3 (Time-invariant frailty)**

* Describe how strong time-varying individual risk unrelated to depletion would produce structured residuals or unstable fits.
* Emphasize that frailty is treated as a geometric construct, not a causal variable.

#### **D.4 Diagnostics for Assumption A4 (Gamma frailty adequacy)**

* Describe residual curvature after normalization.
* Describe poor cumulative-hazard fit or patterned residuals as evidence of misspecification.

#### **D.5 Diagnostics for Assumption A5 (Quiet-window validity)**

* Describe instability of θ under small perturbations of the quiet window.
* Describe failure of post-normalization linearity.
* State explicitly that when this assumption fails, KCOR estimates attenuate or destabilize rather than producing spurious effects.

---

### **Hard constraints for Appendix D**

* ❌ Do NOT introduce new assumptions
* ❌ Do NOT restate assumptions differently
* ❌ Do NOT soften or hedge A1–A5
* ✅ Appendix D exists solely to document **observable diagnostics and failure modes**

---

## **Final Check (must pass)**

* The manuscript contains **exactly five assumptions**, labeled A1–A5
* No other section introduces additional assumptions
* Appendix D references assumptions **by label only**
* Reviewers can locate the full assumption list **only in §2.1.1**

---




Below is a **paper-ready draft of Appendix D**, written to slot directly into the KCOR methods paper.
It **introduces no new assumptions**, references **only A1–A5**, and is explicitly framed as **diagnostics and failure modes**, not conditions.

You can paste this verbatim as **Appendix D**.

---

## Appendix D — Diagnostics and Failure Modes for KCOR Assumptions

This appendix describes the **observable diagnostics and failure modes** associated with each of the five KCOR assumptions (A1–A5). No additional assumptions are introduced here. KCOR is designed to **fail transparently rather than silently**: when an assumption is violated, the resulting lack of identifiability or model stress manifests through explicit diagnostic signals rather than spurious estimates.

---

### D.1 Diagnostics for Assumption A1 (Fixed cohorts at enrollment)

**Assumption A1** requires that cohorts be fixed at enrollment, with no post-enrollment switching or censoring in the primary estimand.

**Diagnostic signals of violation.**

* Inconsistencies in cohort risk sets (e.g., unexplained increases in at-risk counts).
* Early-time hazard suppression or inflation inconsistent with selection or depletion geometry.
* Dependence of results on as-treated reclassification or censoring rules.

**Interpretation.**
KCOR is not defined for datasets with post-enrollment switching or informative censoring in the primary estimand. Such violations are design-level failures rather than modeling failures and indicate that KCOR should not be applied without redefining cohorts.

---

### D.2 Diagnostics for Assumption A2 (Shared external hazard environment)

**Assumption A2** requires that all cohorts experience the same calendar-time external mortality environment.

**Diagnostic signals of violation.**

* Calendar-time hazard spikes or drops that appear in only one cohort.
* Misalignment of major mortality shocks (e.g., epidemic waves) across cohorts.
* Cohort-specific reporting artifacts or administrative discontinuities.

**Interpretation.**
External shocks are permitted under KCOR provided they act symmetrically across cohorts. Cohort-specific shocks violate comparability and are visible directly in calendar-time hazard overlays. When detected, such violations limit interpretation of KCOR contrasts over affected periods.

---

### D.3 Diagnostics for Assumption A3 (Selection via time-invariant latent frailty)

**Assumption A3** posits that selection at enrollment operates primarily through differences in a time-invariant latent frailty distribution that induces depletion of susceptibles.

**Diagnostic signals of violation.**

* Strongly structured residuals in cumulative-hazard space inconsistent with depletion.
* Instability of fitted frailty parameters not attributable to window placement.
* Early-time transients that do not decay and are inconsistent across related cohorts.

**Interpretation.**
Frailty in KCOR is a geometric construct capturing unobserved heterogeneity, not a causal mechanism. If dominant time-varying individual risk unrelated to depletion is present, curvature attributed to frailty becomes unstable. Such cases are revealed by residual structure and parameter instability rather than masked by the model.

---

### D.4 Diagnostics for Assumption A4 (Adequacy of gamma frailty approximation)

**Assumption A4** requires that gamma frailty provides an adequate approximation to the depletion geometry observed in cumulative-hazard space over the estimation window.

**Diagnostic signals of violation.**

* Poor fit of the gamma-frailty cumulative-hazard model during the quiet window.
* Systematic residual curvature after frailty normalization.
* Strong sensitivity of results to minor model or window perturbations.

**Interpretation.**
Gamma frailty is used as a mathematically tractable approximation, not as a claim of biological truth. When depletion geometry deviates substantially from the gamma form, KCOR normalization fails visibly through poor fit and residual curvature. Such behavior indicates model inadequacy rather than supporting alternative interpretation.

---

### D.5 Diagnostics for Assumption A5 (Quiet-window validity)

**Assumption A5** requires the existence of a prespecified quiet window in which selection-induced depletion dominates other sources of curvature, permitting identification of frailty parameters.

**Diagnostic signals of violation.**

* Instability of fitted frailty parameters (θ) under small perturbations of quiet-window boundaries.
* Failure of depletion-neutralized cumulative hazards to become approximately linear during the quiet window.
* Degraded cumulative-hazard fit error concentrated within the nominal quiet period.

**Interpretation.**
Quiet-window validity is the primary dataset-specific requirement for KCOR applicability. When this assumption fails—e.g., due to overlap with strong treatment effects or external shocks—KCOR does not amplify spurious signals. Instead, normalization becomes unstable and KCOR trajectories attenuate toward unity or fail to stabilize, explicitly signaling loss of identifiability.

---

### D.6 Summary: Diagnostic enforcement rather than assumption inflation

KCOR relies on exactly five assumptions (A1–A5), stated exhaustively in §2.1.1. This appendix demonstrates that each assumption has **explicit, observable diagnostics** and **well-defined failure modes**. When assumptions are violated, KCOR signals reduced interpretability through instability, poor fit, or residual structure rather than producing misleading cumulative contrasts. This diagnostic enforcement is a core design feature of the KCOR framework.

---

If you want next, I can:

* tighten this further to reduce word count by ~20–25% for journal limits, or
* cross-check the rest of the paper to ensure **no other section re-introduces assumption-like language** outside A1–A5.

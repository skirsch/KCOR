# KCOR – Final Micro Punchlist: Conservative Bias & Edge-Case Limitations

This punchlist adds **one tightly scoped limitations subsection** clarifying when KCOR is conservative
and which pathological effects may be missed.  
All text below is **exact copy‑paste** and does **not** expand scope or claims.

---

## Add to Discussion → Limitations (single compact subsection)

### Suggested subsection title
> **Conservativeness and Edge-Case Detection Limits**

---

### Paragraph 1: Crossover and conservative bias

**Exact text to add:**
> “Because KCOR compares fixed enrollment cohorts, subsequent uptake of the intervention among initially unexposed individuals (or additional dosing among exposed cohorts) introduces treatment crossover over time. Such crossover attenuates between-cohort contrasts and biases KCOR(t) toward unity, making the estimator conservative with respect to detecting sustained net benefit or harm. Analyses should therefore restrict follow-up to periods before substantial crossover or stratify by dosing state when the data permit.”

---

### Paragraph 2: Acute effects during skipped weeks

**Exact text to add:**
> “KCOR analyses commonly exclude an initial post-enrollment window to avoid dynamic Healthy Vaccinee Effect artifacts. If an intervention induces an acute mortality effect concentrated entirely within this skipped window, that transient signal will not be captured by the primary analysis. This limitation is addressed by reporting sensitivity analyses with reduced or zero skip-weeks and/or by separately evaluating a prespecified acute-risk window.”

---

### Paragraph 3: Degenerate level-only effects

**Exact text to add:**
> “In degenerate scenarios where an intervention induces a purely proportional level-shift in hazard that remains constant over time and does not alter depletion-driven curvature, KCOR’s curvature-based contrast may have limited ability to distinguish such effects from residual baseline level differences under minimal-data constraints. Such cases are pathological in the sense that they produce no detectable depletion signature; in practice, KCOR diagnostics and control tests help identify when curvature-based inference is not informative.”

---

## Explicitly do NOT add

- Do not introduce new estimators
- Do not add equations or simulations
- Do not weaken primary conclusions
- Do not expand causal interpretation

---

## Status after applying this punchlist

- Conservativeness explicitly stated: **yes**
- Acute-risk edge case acknowledged: **yes**
- Pathological level-only effects addressed: **yes**
- Reviewer “gotcha” risk: **minimized**
- Scope integrity: **unchanged**

---

## End of punchlist

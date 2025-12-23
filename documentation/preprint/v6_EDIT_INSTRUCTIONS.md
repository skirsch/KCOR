
# KCOR v6 – Cursor Edit Instructions (Action Checklist)

This file provides **explicit, mechanical instructions** for updating `paper_v6.md`.
Edits are scoped to *clarification, defense, and presentation* only.
No methodological changes are required.

---

## 1. Add figure assets (no inline rendering required)

### Create figures directory
Ensure the following directory exists:

```
figures/
```

Place the following files there (names may be adjusted, but keep references consistent):

```
figures/fig1_kcor_v6_schematic.png
figures/fig2_neg_control_10yr_age_diff.png
figures/fig3_neg_control_20yr_age_diff.png
```

---

## 2. Add figure captions (new subsection)

### Insert after Section 2.4.2 (Gamma-frailty identity)

Add the following caption block **verbatim**:

```markdown
@fig fig1_kcor_v6_schematic:
Three-panel schematic illustrating the KCOR v6 normalization logic.
Left: individual hazards differ only by multiplicative frailty \(z\), with no treatment effect.
Middle: aggregation over heterogeneous frailty induces cohort-level curvature in observed cumulative hazards \(H^{\mathrm{obs}}(t)\) despite identical baseline hazards.
Right: inversion of the gamma-frailty identity recovers aligned baseline cumulative hazards \(H_0(t)\), demonstrating depletion-neutralization.
(Source: `figures/fig1_kcor_v6_schematic.png`)
```

---

## 3. Strengthen negative control interpretation (required sentence)

### Location
Section **3.1.2 Empirical “within-category” negative control**

### Add the following paragraph immediately after the first descriptive paragraph:

```markdown
These age-shift negative controls deliberately induce extreme baseline mortality differences (10–20 year age gaps) while preserving a true null effect by construction, since all vaccination states are compared symmetrically. The near-flat KCOR trajectories demonstrate that the estimator neutralizes selection-induced depletion curvature without introducing spurious time trends or cumulative drift.
```

---

## 4. Add captions for empirical negative control figures

### Insert after the paragraph introducing the two snapshots

```markdown
@fig fig2_neg_control_10yr:
Empirical negative control with approximately 10-year age difference between cohorts.
Despite large baseline mortality differences, KCOR remains near-flat at 1 over follow-up, consistent with a true null effect.
(Source: `figures/fig2_neg_control_10yr_age_diff.png`)

@fig fig3_neg_control_20yr:
Empirical negative control with approximately 20-year age difference between cohorts.
Even under extreme composition differences, KCOR exhibits no systematic drift, demonstrating robustness to selection-induced curvature.
(Source: `figures/fig3_neg_control_20yr_age_diff.png`)
```

---

## 5. Minor wording hardening (low risk, recommended)

### Section 4.1 “What KCOR estimates”
Replace:

> “interpretable as differences in baseline hazard level”

with:

> “interpretable, conditional on the stated selection model and quiet-window validity, as differences in baseline hazard level”

---

## 6. Optional but recommended: explicit failure-mode cross-reference

### Section 5.1 “Failure modes and diagnostics”
At the end of the section, append:

```markdown
In practice, prespecified negative controls—such as the age-shift controls presented in §3.1.2—provide a direct empirical check that KCOR does not generate artifactual cumulative effects under strong selection-induced curvature.
```

---

## 7. Do NOT change

Cursor should **not**:
- Alter equations or notation
- Modify the KCOR estimator definition
- Reinterpret results causally
- Inline-render figures inside Markdown

All changes are presentational and defensive only.

---

## End of instructions

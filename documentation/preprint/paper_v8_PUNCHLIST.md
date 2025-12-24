
# KCOR v6 Paper – Final Punchlist (Preprints / Journal-Ready)

This is a **mechanical punchlist** of remaining fixes and polish items for the KCOR v6 methods paper.
Items are ordered by priority and reviewer impact.

---

## 1. Add explicit **References** section header (required)

### Issue
The reference list currently appears without a clear section header.

### Action
Insert the following heading **immediately before** the numbered reference list:

```markdown
## References
```

This is required for:
- Preprints.org
- Pandoc citation processing
- Most journal templates

---

## 2. Add at least one **canonical gamma-frailty reference** (required)

### Issue
Gamma frailty is well justified conceptually, but reviewers will expect a canonical citation.

### Recommended references (add 1–2)

Choose **at least one** of the following and add to the References section:

**Option A (most canonical, safest):**
- Vaupel JW, Manton KG, Stallard E. *The impact of heterogeneity in individual frailty on the dynamics of mortality.* Demography. 1979;16(3):439–454.

**Option B (survival-analysis textbook standard):**
- Hougaard P. *Analysis of Multivariate Survival Data.* Springer; 2000. Chapter on frailty models.

**Option C (concise modern review):**
- Duchateau L, Janssen P. *The Frailty Model.* Springer; 2008.

### Where to cite
Add an in-text citation in **§2.4.1 Gamma frailty** after the sentence explaining why gamma frailty is widely used.

Example insertion:
```markdown
Gamma frailty is widely used in survival analysis to model unobserved heterogeneity and depletion effects [Vaupel et al., 1979; Hougaard, 2000].
```

---

## 3. Minor math-clarity cleanup in §1.2 (recommended)

### Issue
The curvature explanation references symbols that may not render clearly in some formats.

### Action
Explicitly name the cumulative hazard once.

Suggested replacement sentence:
```markdown
One convenient way to formalize “curvature” is in cumulative-hazard space: if the cumulative hazard H(t) were perfectly linear in time, then its second derivative would be zero, whereas selection-induced depletion generally produces concavity in observed cumulative hazards during otherwise stable periods.
```

---

## 4. Table formatting consistency (low risk, recommended)

### Issue
Tables 1–5 mix caption styles (inline vs block).

### Action
Standardize captions to the form:

```markdown
Table X. <Descriptive title>
```

No content changes needed.

---

## 5. Clarify that figures are **schematic / illustrative** (defensive)

### Issue
Some reviewers may misinterpret synthetic figures as empirical.

### Action
Add one sentence to the caption of **Figure 1** (schematic):

```markdown
This figure is schematic and intended for conceptual illustration; it does not represent empirical data.
```

---

## 6. Optional: add a one-line **scope guard** in the Introduction

### Issue
Preempt “why not causal?” critiques.

### Action
Add the following sentence at the end of §1.4 Contribution:

```markdown
KCOR is proposed as a diagnostic and normalization estimator for selection-induced hazard curvature; causal interpretation requires additional assumptions beyond the scope of this methods paper.
```

---

## 7. Optional: tighten terminology consistency

### Action
Globally prefer:
- “selection-induced depletion curvature”
over:
- “selection dynamics”
- “depletion effects” (when curvature is meant)

This is purely stylistic.

---

## 8. DO NOT change (explicit)

Do **not**:
- Alter equations or estimator definitions
- Change KCOR notation
- Add applied vaccine-effect conclusions
- Add policy or clinical claims

---

## Status summary

- Core methodology: **complete**
- Validation: **strong**
- Negative controls: **excellent**
- Remaining work: **editorial + citation polish only**

After addressing items 1–3, the paper is fully suitable for:
- Preprints.org posting
- Journal submission as a methods paper

---

## End of punchlist

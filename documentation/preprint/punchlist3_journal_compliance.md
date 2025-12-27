# KCOR Methods Paper – Journal Compliance Pass

This document specifies a mechanical, journal-compliance review to be executed
after scientific content is finalized. No methodological changes are requested.

Target journals (overlapping requirements):
- Statistics in Medicine (SIM)
- American Journal of Epidemiology (AJE)
- Biometrics

---

## 1. Front Matter Compliance

### 1.1 Title
- Ensure title is descriptive, neutral, and methods-focused
- Avoid causal or policy claims

**Check**
- No claims of “benefit,” “harm,” or “lives saved”
- Includes “method,” “framework,” or “estimator” language

---

### 1.2 Abstract structure
Confirm abstract includes (explicitly or implicitly):
- Background / motivation
- Methods (what KCOR estimates)
- Validation (negative + positive controls)
- Scope / limitations
- 250 words or less

**Actions**
- No section headings unless journal allows
- Avoid first-person language
- No citations in abstract (SIM/AJE standard)

---

### 1.3 Keywords
- 4–6 keywords
- Methodological rather than topical

**Suggested keywords**
- Survival analysis
- Frailty models
- Selection bias
- Non-proportional hazards
- Observational epidemiology

---

## 2. Methods Section Compliance

### 2.1 Estimand clarity (required)
Confirm the estimand is clearly stated and not causal unless qualified.

**Check**
- KCOR described as:
  - “depletion-neutralized cumulative hazard comparison”
- Explicit statement:
  - “KCOR is not an instantaneous hazard ratio”
  - “No proportional hazards assumption”

---

### 2.2 Assumptions section
Ensure assumptions are:
- Explicit
- Enumerated
- Separated from results

**Required assumptions**
- Fixed cohort membership
- No post-enrollment reassignment
- Correct specification of quiet window
- Identifiability of frailty curvature

---

### 2.3 Competing risks / censoring
Confirm wording clarifies:
- Whether censoring is absent, minimal, or explicit
- How competing risks are treated (or excluded)

No need to add new analysis—just clarity.

---

## 3. Figures & Tables (Journal Style)

### 3.1 Figure standards
- All figures are referenced in text
- Captions are standalone and descriptive
- No color dependence (grayscale safe)

**Check**
- Axis labels defined in captions or figure
- Units specified where applicable
- No unexplained symbols

---

### 3.2 Table standards
- Tables are not screenshots
- Captions appear above tables (Pandoc default is acceptable)
- Notes placed below tables if needed

---

## 4. Simulation & Validation (Critical for Methods Journals)

### 4.1 Negative controls (required by SIM/AJE)
Confirm:
- At least one explicit negative control scenario
- “Near-flat KCOR(t)” criterion stated quantitatively

---

### 4.2 Positive controls
Confirm:
- Directional sensitivity demonstrated
- No overclaiming of effect size accuracy

---

### 4.3 Diagnostic criteria
Ensure diagnostics are:
- Pre-specified
- Not tuned post hoc

Explicitly state:
- RMSE thresholds
- Linearity checks
- Parameter stability windows

---

## 5. Discussion & Limitations

### 5.1 What KCOR estimates
Confirm discussion includes:
- What KCOR *does* estimate
- What KCOR *does not* estimate

**Must include**
- Not a causal estimator by itself
- Not a replacement for randomized evidence
- Sensitive to quiet-window misspecification

---

### 5.2 External validity
Add (or confirm presence of):
- Statement on transportability
- Dataset dependence
- Generalization beyond mortality

---

## 6. Ethics, Data, and Code Statements

### 6.1 Ethics statement
Even if not required:
- State that only de-identified, previously collected data were used
- No human subjects approval required (if applicable)

---

### 6.2 Data availability
Confirm:
- Data access restrictions are stated clearly
- Reproducible code paths are described

---

### 6.3 Code availability
- Repository link included
- Versioned release (tag or DOI if available)

---

## 7. Acknowledgements & Disclosures

### 7.1 Acknowledgements
Confirm:
- Contributions are intellectual only
- No implied endorsement

Already acceptable:
> “The author thanks Clare Craig and Jasmin Cardinal for helpful discussions and methodological feedback during the development of this work.”

---

### 7.2 Conflicts of interest
Explicitly state:
- “The author declares no competing interests.”

---

## 8. Formatting & Style Checks

### 8.1 Language
- Third person throughout
- No rhetorical questions
- No blog-style emphasis

---

### 8.2 Citations
- Consistent style (author–year or numeric)
- No uncited references
- No references cited only in figures

---

## 9. Journal-Specific Toggles (Enable as Needed)

### Statistics in Medicine
- Emphasize estimator properties
- Validation > application
- Avoid policy discussion

### AJE
- Strengthen epidemiologic framing
- Explicit bias discussion

### Biometrics
- Highlight mathematical structure
- Consider appendix for derivations

---

## Completion Criteria

This compliance pass is complete when:
- All assumptions are explicit
- Estimand is unambiguous
- Figures and tables are publication-clean
- No journal policy conflicts remain


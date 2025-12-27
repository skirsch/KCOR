# KCOR v6 Methods Paper – Final Punchlist for Submission

Purpose: finalize the manuscript for submission to a statistics/methods journal
(e.g., Statistics in Medicine) by tightening language, improving reviewer
orientation, and eliminating minor inconsistencies. No methodological changes.

---

## 1. Terminology consistency (global)

### 1.1 Frailty terminology (FINAL STANDARD)

Adopt the following conventions consistently throughout the manuscript:

- **Frailty normalization**  
  → primary process term (used in Methods, headings)

- **Depletion-neutralized**  
  → adjective describing resulting hazards or comparisons

- **Gamma-frailty model / gamma-frailty fit**  
  → mechanism used to estimate frailty parameters (secondary, technical)

Actions:
- Replace any remaining uses of:
  - “frailty neutralization” (as a noun phrase)
  - “gamma-frailty inversion” (as a headline term)
- Ensure:
  - Headings use “frailty normalization”
  - Results use “depletion-neutralized”
  - Gamma frailty appears only when describing estimation mechanics

Acceptance:
- No section mixes these roles
- Terminology aligns with statistical norms and avoids causal overtones

---

## 2. Generalization clarity (event / intervention)

### 2.1 Event terminology
Verify consistency of:
- “event”
- “event date”
- “event time”

Actions:
- Ensure “date of death” appears only when discussing mortality specifically
- Everywhere else, use “event” or “event date (e.g., death)”

Acceptance:
- No accidental reversion to “death” in generic definitions.

### 2.2 Intervention count
Verify:
- “intervention count” is defined once
- “dose” is used only in COVID-specific examples

Acceptance:
- Notation table and Methods intro are aligned.

---

## 3. Abstract tightening (style only)

Goal: reduce cognitive load for methods reviewers.

Actions:
- Break any sentences >35 words
- Reduce stacked clauses introducing:
  - selection
  - curvature
  - frailty
  - cumulative hazards
into two sentences where possible

Do NOT:
- Remove content
- Change claims

Acceptance:
- Abstract reads cleanly on a single pass by a methods reviewer.

---

## 4. Centralize assumptions (reviewer aid)

### 4.1 Add a short subsection in Methods
Add a new subsection after §2.1:

**Title:** “Assumptions and identifiability conditions”

Include bullet points only:
- Fixed cohorts at enrollment
- Quiet-window validity
- Gamma-frailty as geometric approximation
- Identifiability via curvature
- Diagnostics required when assumptions fail

Length:
- 6–8 bullets
- No equations

Acceptance:
- Reviewer can find all assumptions in one place.

---

## 5. Figure proximity and signposting

### 5.1 Simulation grid figures
Issue:
- Simulation grid figures appear distant from first textual reference.

Actions:
- Add explicit forward-reference sentences:
  - “See Figures X–Y below”
- Ensure every figure is referenced in text before appearance.

Acceptance:
- No figure appears “unintroduced.”

---

## 6. Table numbering and references

### 6.1 Tables
Verify:
- All tables use Pandoc auto-numbering syntax
- All tables are referenced in text at least once

Actions:
- Add “Table @tbl:KCOR_algorithm shows …” style references where missing

Acceptance:
- No orphan tables.

---

## 7. Discussion tightening (optics)

### 7.1 Reduce repetition
Actions:
- In Discussion §4:
  - Remove restatement of mechanics already covered in Methods
  - Focus on interpretation, diagnostics, and scope

Acceptance:
- Discussion adds perspective, not re-derivation.

---

## 8. Limitations ordering (reviewer psychology)

Actions:
- Move the strongest/most obvious limitation FIRST:
  - model dependence
- Place “non-gamma frailty” later
- Keep “causal interpretation” explicit but concise

Acceptance:
- Limitations read as honest and controlled, not defensive.

---

## 9. Acknowledgements (final check)

Actions:
- Keep acknowledgements concise
- Ensure wording:
  - “helpful discussions and methodological feedback”
  - no implication of endorsement

Acceptance:
- Standard journal tone.

---

## 10. Final mechanical checks

Actions:
- Verify:
  - No broken figure paths
  - No unresolved citations
  - All equations are referenced or intentionally standalone
  - Section numbering is monotone
- Run a spell/grammar pass ONLY (no rewriting)

Acceptance:
- Paper compiles cleanly to PDF with no warnings.

---

## Stop condition

When all checklist items pass:
- Do NOT continue editing
- Freeze manuscript for submission

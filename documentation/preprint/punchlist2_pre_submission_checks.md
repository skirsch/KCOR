# KCOR Methods Paper – Submission Punch List

This punch list enumerates final pre-submission fixes and consistency checks.
Items are ordered to minimize risk and maximize mechanical executability.
No substantive methodological changes are requested.

---

## 1. Terminology Consistency (High Priority)

### 1.1 Standardize frailty language
Ensure consistent usage across abstract, section headers, and methods:

- Use **“gamma-frailty normalization”** for the *process*
- Use **“depletion-neutralized”** as the *result adjective*
- Avoid introducing new phrases such as:
  - “frailty neutralization”
  - “frailty inversion” (unless explicitly defined as the mathematical step)

**Actions**
- Search for: `frailty neutral`, `neutralization`, `inversion`
- Replace with:
  - Process: “gamma-frailty normalization”
  - Result: “depletion-neutralized cumulative hazard”

Confirm consistency in:
- Abstract
- §2.4–2.6 headings
- Figure captions
- Discussion §4.1

---

## 2. Figures: Placement, Referencing, Completeness

### 2.1 Figure placement relative to first reference
Verify that each figure appears *after* its first textual reference and within the same major section.

Check specifically:
- `fig:kcor_workflow` appears after §2.1 first reference (already correct)
- Simulation grid figures (§3.4) are placed immediately after the subsection header

**Action**
- If any figure floats to the next page in PDF:
  - Move the figure block earlier in the markdown (before large tables or code blocks)

---

### 2.2 Figure numbering and references
Confirm that:
- Every `![...](){#fig:...}` has at least one `Figure @fig:...` reference
- No references exist to undefined figures

**Action**
- Run a search for `@fig:` and verify all targets exist
- Ensure captions are descriptive and end with a period

---

## 3. Tables: Captioning and Referencing

### 3.1 Table numbering sanity check
Pandoc auto-numbers tables from captions. Confirm:

- Every table has a caption of the form:
Table: <caption text> {#tbl:...}

- Every table is referenced at least once in text using `Table @tbl:...`

Pay special attention to:
- `tbl:KCOR_algorithm`
- `tbl:HVE_motivation`
- `tbl:neg_control_summary`
- `tbl:pos_control_summary`

**Action**
- Insert a one-line reference sentence if any table is unreferenced

---

## 4. Figure–Table Ordering Logic

### 4.1 Ensure conceptual order
Within each section:
- Text → Figure → Table (when possible)
- Avoid Table → Figure → Text sequences

This improves reviewer readability.

**Action**
- Reorder markdown blocks if needed; do not change content

---

## 5. Minor Clarity Edits (Safe, Local)

### 5.1 Intervention generalization (already mostly done)
Confirm consistent use of:
- “intervention count” or “discrete exposure index”
- “event date” instead of “date of death” except where mortality is explicit

**Action**
- Search for “dose” outside COVID-specific examples
- Replace with neutral phrasing where not essential

---

### 5.2 Quiet-window phrasing
Ensure consistent phrasing:
- “quiet window” (lowercase, hyphenated)
- Avoid mixing with “quiet period” unless defined as synonymous

---

## 6. Acknowledgements (Already Correct, Lock In)

No changes needed other than final proofreading.

Confirm text reads:

> “The author thanks Clare Craig and Jasmin Cardinal for helpful discussions and methodological feedback during the development of this work.”

No affiliation or endorsement implied.

---

## 7. Submission Hygiene Checks

### 7.1 Cross-references
- No unresolved `@eq:`, `@fig:`, or `@tbl:` references
- No duplicated labels

### 7.2 Math formatting
- Inline math uses `$...$`
- Display math uses standalone `$$` blocks
- No mixed LaTeX environments

### 7.3 Word count
- Word count excludes Supplement unless journal explicitly requires otherwise

---

## 8. Optional (Post-Submission / Low Priority)

- Consider adding a one-line sentence in §4.1 explicitly stating:
“KCOR does not estimate instantaneous hazard ratios.”
- Not required for submission.

---

## Completion Criteria

This punch list is complete when:
- Terminology is consistent
- All figures/tables are referenced and correctly placed
- No unresolved cross-references remain
- PDF renders without layout anomalies



---
name: Clean up formatting artifacts and version references
overview: Remove version artifacts (KCOR v6 → KCOR), fix internal reference formatting artifacts, and remove HTML comments to prepare the manuscript for submission.
todos: []
---

# Plan: Clean Up Formatting Artifacts and Version References

## Clarification

- **HTML comments** (`<!-- ... -->`): Pandoc strips these automatically, so they won't appear in final PDF/DOCX output. However, per user preference, we will remove them entirely for cleanliness.
- **Version text in actual content**: "KCOR v6" appears in the actual manuscript text (not in comments) and WILL appear in the final output. These must be replaced with "KCOR".

## Issues to Fix

### 1. Remove HTML Comments

- **Lines 3-13**: Remove the entire HTML comment block containing version notes and internal instructions
- These won't appear in output but will be removed for cleanliness

### 2. Remove Version Artifacts from Text

Replace all instances of "KCOR v6" with "KCOR" in the actual manuscript text:

- **Line 357**: "KCOR v6 normalization step" → "KCOR normalization step"
- **Line 359**: Figure caption "KCOR v6 normalization logic" → "KCOR normalization logic"
- **Line 408**: "KCOR v6 defaults" → "KCOR defaults"
- **Line 507**: "KCOR v6 pipeline" → "KCOR pipeline"
- **Line 520**: Table caption "KCOR v6 algorithm" → "KCOR algorithm"
- **Line 923**: "KCOR v6 reference implementation" → "KCOR reference implementation"

### 3. Fix Internal Reference Formatting Artifacts

- **Line 129**: Remove `{#tbl:positioning}` - this is a formatting artifact that appears in the actual text. The text on line 129 is explanatory text for Table 1, not a table caption itself, so this identifier should be removed.

### 4. Verify No Other Issues

- Check for any other formatting artifacts or inconsistencies
- Ensure all table/figure references use proper Pandoc syntax
- Verify no broken cross-references

## Files to Modify

- `documentation/preprint/paper.md` - Main manuscript file

## Approach

1. Remove HTML comment block (lines 3-13)
2. Replace all "KCOR v6" with "KCOR" in actual text (6 instances)
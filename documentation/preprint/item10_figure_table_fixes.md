# Item 10: Figure/Table Quality Fixes

This document outlines the manual edits needed for item 10 of the punch list.

## 10.1 Figure 3: Split into two panels

**Current:** Figure @fig:kcor_schematic is a single three-panel schematic.

**Action needed:**
1. Split the existing figure into two separate figures:
   - **Panel A (new Figure 3a):** Conceptual schematic showing hazard curvature + depletion (left and middle panels from current figure)
   - **Panel B (new Figure 3b):** Normalization effect showing before vs after (right panel from current figure)

2. Update the figure reference in paper.md (line 441):
   - Change from single `@fig:kcor_schematic` to two references: `@fig:kcor_schematic_concept` and `@fig:kcor_schematic_normalization`
   - Update caption text accordingly

3. Generate the two new figure files:
   - `figures/fig3a_kcor_schematic_concept.png` (Panels A-B from original)
   - `figures/fig3b_kcor_schematic_normalization.png` (Panel C from original)

**Location in paper:** Section 2.4.3 (around line 441)

## 10.2 Figure 9 (sensitivity heatmaps): Improve readability

**Current:** Figure @fig:sensitivity_overview shows heatmaps with potentially small labels.

**Action needed:**
1. **Increase label font size:**
   - Edit the figure generation script: `code/generate_sensitivity_figure.py` (or wherever this figure is generated)
   - Increase `fontsize` parameters for axis labels, tick labels, and colorbar labels
   - Recommended: axis labels ≥ 12pt, tick labels ≥ 10pt

2. **Add explicit legend:**
   - Add a clear legend/annotation stating:
     - X-axis: "Quiet-window start offset (weeks)"
     - Y-axis: "Baseline weeks"
     - Color scale: "KCOR(t) value" with explicit range (e.g., "0.95-1.05")

3. **Consider 1D slices alternative:**
   - If readability is still poor, create an alternative figure showing 1D slices
   - Plot KCOR(t) vs quiet-window offset for 2-3 representative baseline week values
   - This can be a supplementary figure or replace the heatmap

**Location in paper:** Section 3.3 (around line 847)

**Script to modify:** Find and edit the script that generates `figures/fig_sensitivity_overview.png`

## 10.3 Table @tbl:comparison_estimands: Move earlier and add sentence

**Current:** Table appears around line 881, after RMST discussion.

**Action needed:**
1. **Move table earlier:**
   - Cut the table from its current location (around line 881-887)
   - Paste it immediately after the first mention of RMST comparison (around line 874-875, right after "**Comparison with alternative estimands.**")

2. **Add introductory sentence:**
   - Add this sentence right before the table:
     > "Table @tbl:comparison_estimands summarizes KCOR, Cox, and RMST behavior under the same simulation settings."

**Location in paper:** Section 3.4 (around line 874-881)

## Summary Checklist

- [ ] Split Figure 3 into two panels (3a and 3b)
- [ ] Update figure references in paper.md
- [ ] Generate new figure files
- [ ] Increase font sizes in sensitivity heatmap figure
- [ ] Add legend to sensitivity heatmap
- [ ] (Optional) Create 1D slices alternative figure
- [ ] Move Table @tbl:comparison_estimands earlier
- [ ] Add introductory sentence before table

## Notes

- Figure edits require image editing software (e.g., Python matplotlib, ImageMagick, or manual editing)
- Table movement is a simple cut/paste in paper.md
- Font size changes require modifying the figure generation script


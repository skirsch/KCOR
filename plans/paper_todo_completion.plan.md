---
name: Paper TODO Completion
overview: Complete all TODO items in paper_v8.md including generating missing figures (synthetic negative control, positive control, sensitivity overview), populating data tables, adding placeholder text for administrative sections, and drafting appendix content.
todos:
  - id: pos-control-code
    content: Create positive control test code (generate_positive_control.py)
    status: pending
  - id: pos-control-makefile
    content: Create positive control Makefile and run tests
    status: pending
  - id: pos-control-figure
    content: Generate positive control figure (fig_pos_control_injected.png)
    status: pending
  - id: neg-control-figure
    content: Copy/update synthetic negative control figure reference
    status: pending
  - id: sensitivity-figure
    content: Create sensitivity figure generator and output
    status: pending
  - id: update-tables
    content: Populate positive control results table with test outputs
    status: pending
  - id: admin-placeholders
    content: Add administrative section placeholder text
    status: pending
  - id: appendix-a
    content: Draft Appendix A mathematical derivations
    status: pending
  - id: appendix-b
    content: Draft Appendix B control-test specifications
    status: pending
  - id: appendix-c
    content: Draft Appendix C diagnostics placeholders
    status: pending
  - id: update-paper
    content: Update paper_v8.md with all figure references and content
    status: pending
---

# Complete paper_v8.md TODO Items

## Summary of TODO Items

The paper has 3 categories of incomplete items:

1. **Figures to generate** (3 figures using placeholder)
2. **Tables to populate** (positive control results)
3. **Administrative/appendix content** (authors, ethics, appendices A/B/C)

---

## 1. Figures to Generate

### 1.1 Synthetic Negative Control Figure (line 352)

- **Current**: References `figures/fig_todo_placeholder.png`
- **Solution**: Copy existing [`figs/neg_control_pathological_kcor.png`](documentation/preprint/figs/neg_control_pathological_kcor.png) to `figures/fig_neg_control_synthetic.png` and update reference
- This figure already shows KCOR ~1.0 under null with different frailty mixtures

### 1.2 Positive Control Figure (line 395)

- **Current**: References `figures/fig_todo_placeholder.png` with TODO note
- **Solution**: Create new positive control test infrastructure:

1. Create [`test/positive_control/code/generate_positive_control.py`](test/positive_control/code/generate_positive_control.py) - generates KCOR_CMR-format data with injected hazard multiplier
2. Create [`test/positive_control/Makefile`](test/positive_control/Makefile) - runs the test
3. Generate figure showing KCOR deviation from 1.0 under injected effect

**Data generation approach**:

- Start from synthetic gamma-frailty null data (two cohorts, same baseline hazard)
- Inject hazard multiplier `r` (e.g., 1.2 for harm) into cohort A over window [t1, t2]
- Output as KCOR_CMR format Excel file
- Run KCOR.py to compute KCOR(t)
- Plot showing clear deviation from 1.0

### 1.3 Sensitivity Analysis Figure (line 416)

- **Current**: References `figures/fig_todo_placeholder.png`
- **Solution**: Create [`code/generate_sensitivity_figure.py`](code/generate_sensitivity_figure.py) that:

1. Reads existing [`test/sensitivity/out/KCOR_SA.xlsx`](test/sensitivity/out/KCOR_SA.xlsx)
2. Generates heatmap or distribution plot of KCOR values across parameter grid
3. Outputs to `figures/fig_sensitivity_overview.png`

---

## 2. Tables to Populate

### 2.1 Positive Control Results Table (lines 399-400)

- **Current**: TODO placeholders for injection window and observed behavior
- **Solution**: After running positive control tests, populate:
- Benefit scenario: r=0.8, window [week 20, week 60], expected KCOR < 1
- Harm scenario: r=1.2, window [week 20, week 60], expected KCOR > 1

---

## 3. Administrative Sections

### 3.1 Manuscript Metadata (lines 19-22)

Add standard placeholders:

- Authors: "[Author names to be added]"
- Affiliations: "[Affiliations to be added]"
- Corresponding author: "[To be added]"
- Word count: "[To be calculated]"

### 3.2 Declarations (lines 487-518)

Add standard placeholder text for:

- Ethics approval: "Not applicable (methods paper using synthetic and publicly available aggregated data)"
- Consent for publication: "Not applicable"
- Data availability: Draft text referencing repository
- Code availability: Draft text with GitHub URL placeholder
- Competing interests, Funding, Authors' contributions, Acknowledgements: "[To be added]"

---

## 4. Appendix Content

### 4.1 Appendix A: Mathematical Derivations (lines 529-530)

Draft content for:

- Derivation of gamma-frailty identity (from individual to cohort hazards)
- Inversion formula derivation
- Variance propagation for KCOR ratio

### 4.2 Appendix B: Control-Test Specifications (lines 532-534)

Document:

- Negative control construction (age-shift design)
- Synthetic gamma-frailty null parameters
- Positive control injection parameters (r values, windows, seeds)

### 4.3 Appendix C: Additional Diagnostics (lines 536-538)

Placeholder structure for:

- Fit residual plots
- Parameter stability analysis
- Quiet-window robustness checks

---

## Files to Create/Modify

| File | Action |
|------|--------|
| [`test/positive_control/code/generate_positive_control.py`](test/positive_control/code/generate_positive_control.py) | Create |
| [`test/positive_control/Makefile`](test/positive_control/Makefile) | Create |
| [`code/generate_sensitivity_figure.py`](code/generate_sensitivity_figure.py) | Create |
| [`documentation/preprint/figures/fig_neg_control_synthetic.png`](documentation/preprint/figures/fig_neg_control_synthetic.png) | Copy from figs/ |
| [`documentation/preprint/figures/fig_pos_control_injected.png`](documentation/preprint/figures/fig_pos_control_injected.png) | Generate |
| [`documentation/preprint/figures/fig_sensitivity_overview.png`](documentation/preprint/figures/fig_sensitivity_overview.png) | Generate |
| [`documentation/preprint/paper_v8.md`](documentation/preprint/paper_v8.md) | Update TODOs |


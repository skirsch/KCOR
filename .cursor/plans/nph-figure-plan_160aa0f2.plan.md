---
name: nph-figure-plan
overview: Create the three NPH figures specified in `documentation/preprint/nph_figures.md`, generate them from the current alpha outputs, and insert the corresponding figure references/captions into `paper.md` Section 3.4.
todos:
  - id: nph-fig-update-alpha-plotting
    content: Extend the alpha plotting pipeline to generate the three manuscript NPH figures from current alpha outputs.
    status: completed
  - id: nph-fig-insert-paper-refs
    content: Insert the three NPH figures and matching references/captions into `paper.md` Section 3.4.
    status: completed
  - id: nph-fig-qa
    content: Verify figure/spec alignment, manuscript references, and edited-file lint/build health.
    status: completed
isProject: false
---

# NPH Figure Plan

## Goal

Add three publication-quality NPH figures to support `§3.4` in the manuscript and wire them into the current alpha workflow and Results text.

## Figure Outputs

Create these figure assets under the manuscript figure area so they are available to the paper build:

- `fig_alpha_synthetic_recovery.png`
- `fig_alpha_czech_objective.png`
- `fig_alpha_czech_diagnostics.png`

Use the styling constraints from [c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\nph_figures.md](c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\nph_figures.md): consistent styling with existing figures, pairwise as solid and collapse as dashed, descriptive legends, and publication-quality resolution.

## Data Sources And Generation Logic

Reuse the current alpha outputs already present in [c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out](c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out):

- [c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_synthetic_recovery.csv](c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_synthetic_recovery.csv)
- [c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_objective_curves.csv](c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_objective_curves.csv)
- [c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_best_estimates.csv](c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_best_estimates.csv)
- [c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_leave_one_out.csv](c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_leave_one_out.csv)
- [c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_bootstrap.csv](c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_bootstrap.csv)
- [c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_theta_scale_summary.csv](c:\Users\stk\Documents\GitHub\KCOR\test\alpha\out\alpha_theta_scale_summary.csv) as a cross-check while plotting the pooled primary specification.

Update the alpha plotting code in [c:\Users\stk\Documents\GitHub\KCOR\test\alpha\code\estimate_alpha.py](c:\Users\stk\Documents\GitHub\KCOR\test\alpha\code\estimate_alpha.py) so these figures are generated reproducibly from the existing pipeline rather than manually.

Alpha grid requirements:

- must be monotonic and evenly spaced
- expected range: `1.00` to `1.30`
- if the input grid differs, resample or interpolate to a consistent plotting grid before plotting
- keep the plotting grid explicit rather than inferred from partial slices

Source of pooled estimates:

- use `alpha_best_estimates.csv` as the single source of truth for reported pooled estimate markers and manuscript-linked values
- do not recompute pooled estimates from `alpha_objective_curves.csv`
- if plotted minima and `alpha_best_estimates.csv` differ by more than `0.01`, warn in the run output but continue

Reproducibility target:

- the full figure set should be reproducible from a single alpha command path, ideally via the existing `estimate_alpha.py` entrypoint rather than manual post-processing

## Planned Figures

### Figure 1

Purpose: show estimator validity under synthetic data.

Source:

- `alpha_synthetic_recovery.csv`

Plot:

- x-axis: `alpha_true`
- y-axis: mean recovered `alpha`
- series: pairwise mean, collapse mean, identity line
- uncertainty: SD or SE bars computed from replicate dispersion
- include both current synthetic branches so the baseline and harder heteroskedastic recovery are both visible

Insertion point in [c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md](c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md): after the first paragraph of `§3.4.1`, plus a short sentence at the end of that subsection pointing to the figure.

### Figure 2

Purpose: show that the pooled Czech specification has a non-flat, well-defined objective minimum.

Source:

- `alpha_objective_curves.csv`
- `alpha_best_estimates.csv`

Plot:

- pooled primary Czech specification only

Primary specification (must be enforced in code):

- `anchor_mode = "dose0"`
- `excess_mode = "exclude_nonpositive"`
- `time_segment = "pooled"`
- `age_band = "pooled"`

Do not infer defaults; explicitly filter for this combination and fail loudly if multiple or zero matches exist.

- two curves: pairwise and collapse
- normalize each curve so its minimum is 0 for comparability
- vertical reference lines at the pooled reported estimates (`1.19`, `1.18`)

Objective normalization rule:

- for each estimator curve, subtract the minimum value so `min = 0`
- do not scale, standardize, or otherwise transform the curve beyond this additive shift

Insertion point in `paper.md`: after the first paragraph of `§3.4.2`, with a sentence explicitly linking the reported estimates to the objective minima.

### Figure 3

Purpose: show both stability and fragility diagnostics.

Sources:

- `alpha_leave_one_out.csv`
- `alpha_bootstrap.csv`
- `alpha_best_estimates.csv`

Three-panel layout:

- Panel A: leave-one-cohort-out pairwise estimates with pooled estimate reference line
- Panel B: bootstrap distribution for the primary pooled pairwise estimate with pooled reference line
- Panel C: segmented pooled estimates (`pooled`, `early_wave`, `late_wave`) for pairwise and collapse, using identical filtering except for `time_segment`

Panel C filter rule:

- include `pooled`, `early_wave`, and `late_wave`
- hold `anchor_mode`, `excess_mode`, `age_band`, and theta setting fixed to the primary pooled specification
- vary only `time_segment`
- show both pairwise and collapse estimates for each segment

Insertion point in `paper.md`: during `§3.4.3`, after the perturbation discussion, with a sentence pointing readers to the combined diagnostic view.

## Manuscript Updates

Update [c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md](c:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md) to:

- add the optional one-line intro before `§3.4` if it still reads naturally with the current Results flow
- insert the three figures with final captions adapted from `nph_figures.md`
- add the corresponding in-text references in `§3.4.1`, `§3.4.2`, and `§3.4.3`
- ensure captions stay conservative and aligned with the current text: estimator works, pooled signal exists, diagnostics show stability plus fragility

## QA

- Verify filenames and figure labels match manuscript references.
- Confirm the pooled Czech figure uses the same primary specification described in `§3.4.2`.
- Confirm the segmented panel matches the values described in `§3.4.2` and `§3.4.3`.
- Run lints on edited files and, if available, perform a manuscript figure build/check so missing-file references are caught.

Failure conditions:

- if any required CSV input is missing, stop and report the missing file
- if primary-spec rows are not found uniquely, stop and report the mismatch
- if pooled estimates used by the figures differ from the manuscript text by more than `0.01`, warn but proceed


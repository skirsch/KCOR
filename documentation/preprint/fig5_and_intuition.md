Absolutely. Here are **Cursor-ready instructions** for:

1. **Fig 5** polish
2. a new **KCOR intuition figure** that explains the method visually

These are designed to fit the current manuscript structure, where Fig 5 is the synthetic (\alpha) recovery figure in §3.4.1 and Figs 6–7 are already the Czech objective/diagnostic figures. 

---

# Cursor instructions: Fig 5 polish + new KCOR intuition figure

## Goal

Improve **Figure 5** so it matches the quality and clarity of Figures 6–7, and add one new **intuition figure** early in the paper that visually explains:

* the problem KCOR solves
* what raw hazard curvature looks like
* what KCOR normalization does

Do not change the scientific claims. Keep all captions conservative and aligned with the current non-causal, diagnostic-first framing. The NPH results text in §3.4.1–§3.4.3 should remain substantively unchanged except for figure references.

---

## Part A — Upgrade Figure 5

### Figure identity

Current Figure 5 is the synthetic (\alpha) recovery figure in §3.4.1. It already appears in the manuscript as the synthetic validation figure for the NPH exponent module. 

### What to change

Update the plotting code so Figure 5 is visually stronger and more consistent with Figures 6–7.

### File/code target

Update the alpha plotting pipeline in:

* `test/alpha/code/estimate_alpha.py`

### Output file

Keep the same output filename:

* `figures/fig_alpha_synthetic_recovery.png`

### Plot requirements

Use the same underlying synthetic recovery data already being used for the current figure.

#### Layout

Use a **two-panel horizontal layout**:

* **Panel A:** baseline multiplicative-noise branch
* **Panel B:** heteroskedastic branch

Do not combine both branches into one undifferentiated panel.

#### Axes

For both panels:

* x-axis: `True alpha`
* y-axis: `Estimated alpha`

Use the same axis ranges across both panels.

#### Curves

Plot:

* identity line (y=x)
* pairwise estimator mean
* collapse estimator mean

Use:

* pairwise = solid blue with circular markers
* collapse = dashed orange with square markers
* identity = thin gray dotted line

#### Uncertainty

Show error bars from replicate dispersion:

* use SD if that is the current implementation
* do not silently switch to SE unless the current manuscript text is also updated

#### Visual improvements

* keep legend small and unobtrusive
* use consistent font size with Figures 6–7
* enlarge plotting area so points and error bars are easy to read
* make panel titles explicit:

  * `A. Baseline synthetic branch`
  * `B. Heteroskedastic branch`

#### Add error summary annotation

Within each panel, add a small text annotation in a corner reporting:

* mean absolute error for pairwise
* mean absolute error for collapse

Only use values computed from the plotted data. Do not hardcode.

### Caption replacement

Replace the current Figure 5 caption with:

```text
Figure 5: Synthetic validation of NPH exponent recovery. Estimated α versus true α under the working model, shown separately for the baseline synthetic branch (A) and a harder heteroskedastic branch (B). Both pairwise and collapse estimators recover α with low bias when cross-cohort depletion geometry differs. The heteroskedastic branch shows reduced precision rather than structural failure. Error bars show replicate standard deviations.
```

### Optional text tweak in §3.4.1

After the sentence that currently ends with “This recovery behavior is illustrated in Figure 5,” add one short sentence:

```text
Recovery remains visually close to the identity line in the baseline branch and degrades mainly through increased dispersion in the heteroskedastic branch.
```

Do not otherwise rewrite the subsection.

---

## Part B — Add a new KCOR intuition figure

### Purpose

Add one new early figure that helps a reader immediately understand:

1. why raw cohort hazards can mislead
2. what depletion-induced curvature looks like
3. what KCOR normalization does

This figure is for intuition, not for new empirical claims.

### Placement

Insert this figure in **Methods**, not Results.

Best placement:

* immediately after **§2.1 Conceptual framework and estimand**
* before or near the strategy bullets that say:

  1. estimate depletion geometry
  2. invert into depletion-neutralized space
  3. compare after normalization

This keeps it close to the conceptual explanation of curvature and normalization already in the paper. 

### New figure label and filename

Use:

* label: `#fig:kcor_intuition`
* filename: `figures/fig_kcor_intuition.png`

### Figure design

Create a **three-panel horizontal figure** with simple, clean, publication-style plotting.

#### Panel A — Raw hazards

Title:

* `A. Raw cohort hazards`

Show two schematic or representative curves:

* Cohort A
* Cohort B

Requirements:

* both hazards should show curvature over time
* one cohort should start higher and decelerate faster
* the other should start lower and decelerate differently
* emphasize that non-parallel shape can arise from depletion geometry, not necessarily from a treatment effect

This can be schematic if needed. Do not imply it is an empirical Czech result unless it actually uses real data and the caption explicitly says so.

#### Panel B — Depletion-neutralized cumulative hazards

Title:

* `B. After KCOR normalization`

Show the same two cohorts after normalization:

* curves should be much closer to linear
* any remaining separation should appear as level/slope difference after curvature removal

The visual point is:

* curvature is removed
* comparison becomes geometrically cleaner

#### Panel C — KCOR(t)

Title:

* `C. KCOR(t) as the normalized contrast`

Show:

* a simple KCOR trajectory derived from the normalized quantities

You may show either:

* approximately flat KCOR for a null-style schematic, or
* a gently diverging trajectory if that makes the explanation clearer

Preferred choice:

* use a **near-flat null-style schematic**
  because it matches the paper’s diagnostic-first framing better

### Styling rules

* use the same general style family as existing manuscript figures
* avoid clutter
* label cohorts directly when possible
* keep axes simple:

  * x-axis: `Follow-up time`
  * y-axis:

    * Panel A: `Observed hazard`
    * Panel B: `Normalized cumulative hazard`
    * Panel C: `KCOR(t)`

### Important scientific constraint

This figure should be explicitly framed as **schematic / conceptual** unless you are actually pulling a specific real-data example and the manuscript text is updated accordingly.

Do not create a figure that looks empirical if it is schematic.

### Caption for new intuition figure

Insert with this caption:

```text
Figure X: Conceptual intuition for KCOR. (A) Raw cohort hazards can differ in both level and curvature because selection-induced depletion changes the composition of the surviving risk set over time. (B) KCOR estimates cohort-specific depletion geometry and maps observed cumulative hazards into a depletion-neutralized space, where quiet-period trajectories are expected to be approximately linear under the working model. (C) KCOR(t) is then computed as a cumulative contrast on that normalized scale. This figure is schematic and is intended to illustrate the logic of the method rather than report an empirical result.
```

Replace `Figure X` automatically through normal manuscript numbering.

### Add in-text reference

Add one sentence in §2.1 after the strategy bullets:

```text
The geometric logic of this normalization is illustrated schematically in Figure @fig:kcor_intuition.
```

Do not add more than that unless needed for flow.

---

## Part C — QA requirements

After making the changes:

1. confirm `fig_alpha_synthetic_recovery.png` still builds and is referenced correctly in §3.4.1
2. confirm the new `fig_kcor_intuition.png` is referenced correctly in Methods
3. rebuild the manuscript PDF
4. verify:

   * no broken figure references
   * caption numbering updates correctly
   * Figure 5 remains in §3.4.1
   * the new intuition figure appears in Methods and not in Results

If any filename/reference mismatch occurs, stop and report it.

---

## Part D — what not to change

Do **not**:

* renumber figures manually
* rewrite the NPH methods sections
* change the reported Czech (\hat\alpha) values
* change the conservative interpretation of (\alpha)
* replace the current Figure 6 or Figure 7 structure unless needed for build reasons

---

## Suggested commit message

```text
Polish Fig 5 synthetic alpha recovery and add schematic KCOR intuition figure
```

If you want, I can also give you a **second Cursor block** specifically for a stronger empirical alternative to the schematic intuition figure, using a real before/after hazard example instead of a conceptual one.

Yes — this is the better version.

A real **before/after empirical figure** will do much more work than a schematic. It lets the reader see:

* the distortion
* the normalization
* the resulting contrast

That is exactly the kind of figure that makes a methods paper “click.”

Below is a **Cursor-ready block** for the stronger empirical alternative.

---

# Cursor instructions: replace schematic intuition figure with an empirical before/after KCOR figure

## Goal

Instead of a purely schematic intuition figure, create a **real-data or real-analysis-derived before/after figure** that shows:

1. the raw hazard or cumulative-hazard distortion,
2. the same cohorts after KCOR normalization,
3. the resulting KCOR contrast.

This figure should make the method visually intuitive for a skeptical reader without requiring them to understand the full math first.

---

## Placement

Insert this figure in **Methods**, near the conceptual explanation of curvature and normalization.

Best placement:

* after the strategy bullets in **§2.1 Conceptual framework and estimand**
* or immediately before **§2.4 Selection model: gamma frailty and depletion normalization**

Add one sentence in the main text:

```text
The normalization logic is illustrated empirically in Figure @fig:kcor_empirical_intuition using a representative cohort comparison from the analysis pipeline.
```

Do not add more than one or two sentences around it.

---

## Figure identity

Use:

* filename: `figures/fig_kcor_empirical_intuition.png`
* label: `#fig:kcor_empirical_intuition`

---

## Figure concept

Create a **three-panel horizontal figure** using the **same cohort comparison in all three panels**.

Preferred cohort choice:

* use a representative comparison that already exists in the project pipeline and is diagnostically clean
* prioritize a cohort pair where the raw curves show obvious curvature and the normalized curves look substantially simpler
* if available, use a comparison from the Czech primary analysis or a strong negative-control/example cohort already used internally
* do **not** use a pathological or noisy case

The point is not to show the final substantive result.
The point is to show the geometry:

* raw observed shape
* normalized shape
* resulting contrast

---

## Panel structure

### Panel A — Raw observed hazards or cumulative hazards

Title:

```text
A. Raw cohort trajectories
```

Preferred y-axis:

* `Observed hazard` if the hazard curves are smooth and readable
* otherwise use `Observed cumulative hazard`

Use the format that gives the clearest visual intuition.

Plot:

* the two cohorts used in the comparison
* same follow-up axis as Panels B and C
* direct labels or a clean legend

Goal:

* show clearly that the cohorts differ in curvature, not just level

If hazard is too noisy, use lightly smoothed hazard or cumulative hazard, but:

* smoothing must already be part of the analysis style or be minimal and disclosed
* do not over-smooth

---

### Panel B — Depletion-neutralized cumulative hazards

Title:

```text
B. After KCOR normalization
```

Y-axis:

* `Normalized cumulative hazard`

Plot:

* the two corresponding depletion-neutralized cumulative hazards for the same cohorts

Goal:

* show that the large curvature difference seen in Panel A is reduced
* show that the comparison becomes more nearly linear / geometrically interpretable

This is the most important panel visually.

If helpful, include faint linear reference fits in the quiet period only, but only if this improves clarity and does not clutter the panel.

---

### Panel C — KCOR(t)

Title:

```text
C. KCOR(t)
```

Y-axis:

* `KCOR(t)` or `Anchored KCOR(t; t_0)` if that is the cleaner form for this example

Preferred choice:

* use the form that is already most interpretable for the chosen comparison
* if the comparison has large pre-existing level differences, anchored KCOR is acceptable
* if anchored is used, say so clearly in caption

Goal:

* show the final normalized contrast derived from the same pair of cohorts

Add a horizontal reference line at:

* `KCOR = 1`

If anchored KCOR is used, make the reference interpretation explicit.

---

## Data consistency rules

All three panels must be generated from the **same exact cohort pair and preprocessing choice**.

Do not mix:

* one cohort pair in Panel A
* another in Panel B
* another in Panel C

Do not use one analysis for raw hazards and another for KCOR.

The figure must represent a single coherent example from start to finish.

---

## Cohort selection rules

Choose a cohort pair that satisfies all of the following:

1. raw trajectories visibly show curvature or non-parallel evolution,
2. normalization visibly reduces the curvature difference,
3. resulting KCOR panel is not too noisy,
4. the example is representative rather than cherry-picked.

If more than one candidate exists, choose the cleanest and most interpretable one.

If no real-data pair gives a clear figure, then fall back to a synthetic or negative-control example from the main pipeline and state that explicitly in the caption.

---

## Styling

Use the same style family as the manuscript’s other figures.

Requirements:

* publication-quality resolution
* readable axis labels
* consistent fonts
* no cluttered legends
* same cohort colors across all three panels
* horizontal layout if it fits comfortably; otherwise 2-row layout with Panel C full width is acceptable

Use direct line labels if cleaner than a legend.

---

## Caption

Use this caption, adapting only the cohort description and whether the final panel is anchored:

```text
Figure X: Empirical illustration of the KCOR workflow on a representative cohort comparison. (A) The raw observed trajectories differ in both level and curvature, reflecting selection-induced depletion geometry rather than a simple proportional scaling. (B) After estimating cohort-specific depletion geometry and applying gamma-frailty inversion, the normalized cumulative hazards become substantially more linear and directly comparable. (C) KCOR(t) is then computed as the cumulative contrast on that normalized scale. This figure is intended to illustrate the geometric effect of normalization on a real analysis example rather than to serve as a standalone substantive result.
```

If anchored KCOR is used in Panel C, replace the last part of Panel C sentence with:

```text
(C) Anchored KCOR(t; t_0) is then computed as the cumulative contrast on that normalized scale after removing the pre-anchor level difference.
```

---

## Text insertion

In §2.1, after the strategy bullets, add:

```text
The normalization logic is illustrated empirically in Figure @fig:kcor_empirical_intuition using a representative cohort comparison drawn from the analysis pipeline.
```

Do not otherwise rewrite the surrounding section unless needed for flow.

---

## Preferred implementation path

If possible, generate this figure from an existing script or analysis output already in the repository rather than manually assembling arrays.

Best outcome:

* a reproducible plotting function that can regenerate the figure from the existing KCOR workflow outputs

If new code is needed, place it near the manuscript plotting utilities, not as an ad hoc one-off.

---

## QA requirements

After implementation:

1. rebuild the manuscript PDF
2. confirm the new figure appears in Methods, not Results
3. verify all three panels use the same cohort pair
4. confirm the caption accurately matches whether Panel C is KCOR or anchored KCOR
5. confirm the figure is visually legible in the PDF at normal zoom
6. stop and report if the selected cohort pair is too noisy or visually unclear

---

## What not to do

Do not:

* use three different examples across the three panels
* create a stylized schematic while calling it empirical
* use a visually impressive but diagnostically messy example
* rewrite the Results section around this figure
* overclaim that the figure proves causal validity

---

## Suggested commit message

```text
Add empirical before-and-after KCOR intuition figure to Methods
```

One judgment call: I would strongly prefer this figure to use a **clean negative-control or representative null-style comparison** rather than a controversial vaccinated-vs-unvaccinated result, because that keeps the figure focused on method rather than on policy interpretation.

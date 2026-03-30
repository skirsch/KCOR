Perfect — this is exactly the right moment to lock in the figures. I went through your Section 3 in the manuscript  and built **Cursor-ready instructions** that map cleanly to your structure, labels, and claims.

These are designed so Cursor can execute without ambiguity.

---

# 🎯 OVERVIEW (what we’re inserting)

You will add **3 new figures into Section 3.4**:

| Figure | Purpose                                    | Where  |
| ------ | ------------------------------------------ | ------ |
| Fig α1 | Synthetic recovery (validates estimator)   | §3.4.1 |
| Fig α2 | Czech objective curvature (identification) | §3.4.2 |
| Fig α3 | Czech diagnostics (stability / fragility)  | §3.4.3 |

---

# 🧠 GLOBAL INSTRUCTIONS (important)

Add these BEFORE giving Cursor the figure tasks:

```
All figures must:

- Use consistent font and style with existing figures
- Use alpha grid from 1.00 to 1.30 unless otherwise specified
- Label pairwise vs collapse clearly
- Use solid line for pairwise, dashed for collapse
- Include vertical line at estimated alpha
- Save under figures/ with descriptive names
- Use publication-quality resolution (300 dpi)
```

---

# 🧩 FIGURE 1 — Synthetic α recovery

## 📍 Placement

Insert **after first paragraph of §3.4.1**

---

## 🛠️ Cursor instruction

```
Create a new figure: fig_alpha_synthetic_recovery.png

Purpose:
Show that both estimators recover the true alpha under synthetic data.

Data:
Use existing synthetic experiments from §3.4.1:
alpha_true ∈ {1.05, 1.15, 1.25}

Plot:
- x-axis: true alpha
- y-axis: estimated alpha

Plot 3 series:
1) Pairwise estimator (mean)
2) Collapse estimator (mean)
3) Identity line (y = x)

Add:
- Error bars (± SD or ± bootstrap SE if available)
- Legend: Pairwise, Collapse, Identity

Style:
- Pairwise: solid line + circles
- Collapse: dashed line + squares
- Identity: thin gray diagonal

Caption:

"Synthetic validation of NPH exponent recovery. Estimated α versus true α under the working model. Both pairwise and collapse estimators recover α with low bias across the tested range, supporting identifiability when cross-cohort depletion geometry differs."
```

---

## ✍️ Insert reference text

Add at end of §3.4.1:

> “This recovery behavior is illustrated in Figure @fig:alpha_synth_recovery.”

---

# 🧩 FIGURE 2 — Czech objective curvature (THE key figure)

## 📍 Placement

Insert **after first paragraph of §3.4.2**

---

## 🛠️ Cursor instruction

```
Create a new figure: fig_alpha_czech_objective.png

Purpose:
Show that alpha is identified (non-flat objective).

Data:
Use Czech pooled primary specification.

Compute:
For alpha grid from 1.00 to 1.30 (step ~0.005):

1) Pairwise objective:
   sum of squared log-ratio residuals

2) Collapse objective:
   cross-cohort variance

Plot:
- x-axis: alpha
- y-axis: objective value (normalized so min = 0)

Plot two curves:
- Pairwise objective
- Collapse objective

Add:
- Vertical lines at:
   alpha_pair = 1.19
   alpha_coll = 1.18

Style:
- Pairwise: solid blue
- Collapse: dashed orange

Important:
Normalize each curve so minimum = 0 for visual comparability

Caption:

"Objective functions for α under the pooled Czech specification. Both pairwise and collapse estimators exhibit well-defined minima near α ≈ 1.18–1.19, indicating that cross-cohort structure provides identifying information. Flat or ill-defined objectives would indicate weak identifiability."
```

---

## ✍️ Insert reference text

Add after α estimates paragraph:

> “Figure @fig:alpha_czech_objective shows the corresponding objective functions, which exhibit clear minima near the reported values.”

---

# 🧩 FIGURE 3 — Czech diagnostics (stability + fragility)

## 📍 Placement

Insert **midway through §3.4.3**, after perturbation discussion

---

## 🛠️ Cursor instruction

```
Create a new figure: fig_alpha_czech_diagnostics.png

Purpose:
Show stability AND fragility of alpha estimation.

Structure:
Create a 3-panel figure (horizontal panels A, B, C)

-------------------------------------
Panel A: Leave-one-cohort-out (LOO)
-------------------------------------

- x-axis: omitted cohort index
- y-axis: estimated alpha (pairwise)

Plot:
- scatter of alpha estimates
- horizontal line at pooled estimate (1.19)

-------------------------------------
Panel B: Bootstrap distribution
-------------------------------------

- x-axis: alpha
- y-axis: density (or histogram)

Plot:
- histogram of bootstrap alpha estimates
- vertical line at pooled estimate

-------------------------------------
Panel C: Segmented estimates
-------------------------------------

- Categories: pooled, early-wave, late-wave
- y-axis: alpha

Plot:
- pairwise + collapse estimates for each segment

-------------------------------------

Style:
- consistent colors across panels
- label panels A, B, C clearly

Caption:

"Diagnostics for α estimation under the Czech specification. (A) Leave-one-cohort-out analysis shows that most omissions leave the estimate unchanged, while a small number materially affect α, indicating uneven identifying information. (B) Bootstrap distribution shows dispersion consistent with finite-sample uncertainty. (C) Segmented analyses exhibit reduced agreement and stability, illustrating weaker identifiability outside the pooled specification."
```

---

## ✍️ Insert reference text

Add:

> “These diagnostic patterns are summarized in Figure @fig:alpha_czech_diagnostics.”

---

# 🔥 OPTIONAL (but HIGHLY recommended)

Add this 1-line intro before 3.4:

> “Figure-based diagnostics are used throughout to assess identifiability, estimator agreement, and stability of the NPH exponent.”

---

# 🧠 WHY THIS SET WORKS

This trio does something very important:

### Figure 1 → “Estimator works”

### Figure 2 → “Signal exists”

### Figure 3 → “Signal is fragile but real”

That combination is exactly what reviewers want:

* Not overclaiming
* Not hand-wavy
* Visually obvious identification

---

# 🏁 Final take

You’re now at the stage where:

* The math is solid ✅
* The writing is solid ✅
* The **figures will make or break acceptance**

This set is exactly what gets a methods paper over the line.

---

If you want next step after this, I’d suggest:

👉 tightening captions to match *Statistics in Medicine* style
👉 or adding one “summary schematic” panel tying α into KCOR pipeline

But first — run these in Cursor.

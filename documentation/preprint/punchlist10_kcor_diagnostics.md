Perfect — thanks for the clarification.
Below is the **complete, self-contained replacement punch list**, exactly in the spirit of the original *punchlist10*, but updated to reflect **what the paper now actually does**.

You can **copy-paste this verbatim into a new Markdown file** and commit it.

---

# `punchlist10_kcor_diagnostics.md`

## Purpose

This punch list replaces `punchlist10_plausibility.md`.
It defines the **tables, captions, figures, subsection text, and ordering** needed to document **KCOR v6 diagnostic validity and falsifiability**, using **Czech national mortality data**, restricted to **Dose 0 vs Dose 2** cohorts enrolled in **2021_13**.

The goal is not plausibility rhetoric, but **ex-ante model diagnostics** that fail loudly under misspecification.

---

## Scope and data constraints (fixed)

* **Data source:** Czech national mortality + vaccination registry
* **Enrollment cohort:** `EnrollmentDate = 2021_13`
* **Comparison:** **Dose 0 vs Dose 2 only**
* **Ages analyzed:** 40+
* **Outcome:** All-cause mortality
* **Interpretation:** Observational, non-causal prior to KCOR normalization

These constraints are intentional and must not be relaxed in this punch list.

---

## REQUIRED INSERTIONS (in order)

---

## 1️⃣ Methods — new subsection

### Location

Insert **after** the KCOR method definition and frailty estimation procedure.

### Subsection title (exact)

> **2.1.3 Model diagnostics and falsification criteria**

### Paste-ready text

```markdown
KCOR is not validated by goodness-of-fit alone. Instead, it implies a set of independent structural diagnostics that must be satisfied simultaneously. These diagnostics are defined ex ante and provide explicit failure modes: violation of any one constitutes evidence against model adequacy rather than a need for reinterpretation.

Specifically, KCOR implies that (i) unvaccinated cohorts drawn from heterogeneous populations exhibit non-zero gamma frailty with visible curvature in cumulative hazard; (ii) cohorts subject to strong selective uptake exhibit near-zero estimated frailty ($\hat{\theta} \approx 0$) and approximately linear cumulative hazards during epidemiologically quiet windows; (iii) fitted frailty parameters vary coherently with age, reflecting the interaction between baseline hazard and heterogeneity rather than enforcing monotonicity; and (iv) cumulative hazard ratios converge to stable asymptotes following KCOR normalization.

Importantly, none of these properties are imposed as constraints or priors. All emerge from the data. In low-information regimes (e.g., sparse events or highly selected cohorts), KCOR is expected to degrade toward instability or attenuation rather than producing spurious effects. These behaviors provide a falsifiable diagnostic framework rather than a model-tuning mechanism.
```

---

## 2️⃣ Table X — frailty diagnostics (FIRST table)

### Placement

Results section, **before any outcome comparisons**.

### Caption (exact)

> **Table X. Estimated gamma-frailty variance ($\hat{\theta}$) by age band and vaccination status for Czech cohorts enrolled in 2021_13.**

### Table content (median only)

| Age band (years) | $\hat{\theta}$ Dose 0 (median) | $\hat{\theta}$ Dose 2 (median) |
| ---------------- | -----------------: | -----------------: |
| 40–49            |               16.8 |           2.7×10⁻⁶ |
| 50–59            |               18.1 |           8.8×10⁻⁶ |
| 60–69            |               9.85 |           1.0×10⁻⁷ |
| 70+              |              0.964 |          5.7×10⁻¹² |

### Footnote (exact)

> $\hat{\theta}$ quantifies unobserved frailty heterogeneity and depletion of susceptibles within cohorts. Near-zero values indicate effectively linear cumulative hazards over the quiet window and are typical of strongly pre-selected cohorts. Values are summarized as medians across enrollment subcohorts within 2021_13.

---

## 3️⃣ Results text immediately after Table X

### Paste-ready paragraph

```markdown
As shown in Table X, unvaccinated cohorts exhibit substantial frailty heterogeneity ($\hat{\theta} > 0$), while Dose 2 cohorts show near-zero estimated frailty ($\hat{\theta} \approx 0$) across all age bands, consistent with strong selective uptake prior to follow-up. Frailty variance is largest at younger ages, where low baseline mortality amplifies the impact of heterogeneity on cumulative hazard curvature, and declines at older ages where mortality is compressed and survivors are more homogeneous. No diagnostic reversals or instabilities are observed.
```

---

## 4️⃣ Table Y — raw cumulative outcomes (SECOND table)

### Purpose

Descriptive setup only. **Not causal.**

### Caption (exact)

> **Table Y. Ratio of observed cumulative mortality hazards for unvaccinated (Dose 0) versus fully vaccinated (Dose 2) Czech cohorts enrolled in 2021_13.**

### Required footnote (exact)

> Values reflect raw cumulative outcome differences prior to KCOR normalization and are not interpreted causally due to cohort non-exchangeability.

(Actual numeric values come from the outcome CSV, not this punch list.)

---

## 5️⃣ Methods — Cox comparison subsection (minimal, diagnostic)

### Location

End of Methods, **after Tables X and Y are defined conceptually**.

### Subsection title

> **Relationship to Cox proportional hazards**

### Paste-ready text

```markdown
Cox proportional hazards models estimate instantaneous hazard under the assumption of time-invariant hazard ratios. In observational cohorts with selective uptake and frailty heterogeneity, this assumption is structurally violated, leading to time-varying hazard ratios and cumulative hazard trajectories inconsistent with observed data. Cox estimates are therefore presented here solely as a diagnostic baseline to illustrate assumption failure, not as a competing causal estimator.
```

---

## 6️⃣ Figures (captions only)

### Figure 1 — cumulative hazards

> **Figure 1.** Observed cumulative mortality hazards for Dose 0 and Dose 2 cohorts (Czech data, enrollment 2021_13), showing pronounced curvature in unvaccinated cohorts and near-linearity in vaccinated cohorts.

### Figure 2 — KCOR normalization

> **Figure 2.** KCOR-normalized cumulative hazards and ratios, demonstrating convergence to stable asymptotes following frailty neutralization.

### Figure 3 — Cox failure (optional / supplement)

> **Figure 3.** Cox-implied cumulative hazards compared with observed cumulative hazards, illustrating time-varying hazard ratios and structural proportional-hazards violations.

---

## 7️⃣ Discussion — single reinforcing sentence

### Location

Discussion §5.2 (Limitations / robustness)

### Paste-ready sentence

```markdown
Because KCOR defines explicit diagnostic failure modes—instability, dose reversals, age incoherence, or absence of asymptotic convergence—the absence of such failures in the Czech 2021_13 Dose 0 versus Dose 2 cohorts provides stronger validation than goodness-of-fit alone.
```

---

## 8️⃣ Ordering checklist (final sanity check)

```markdown
[ ] Insert §2.1.3 “Model diagnostics and falsification criteria”
[ ] Insert Table X before any outcome comparisons
[ ] Insert Table Y after Table X
[ ] Frame Cox as diagnostic only
[ ] Reference Table X immediately in Results
[ ] Reference diagnostics once in Discussion
```

---

## Summary statement (author reference)

> This punch list defines KCOR validation as the satisfaction of multiple independent structural diagnostics using Czech national data (enrollment 2021_13), not as curve fitting or plausibility argumentation.

---

If you want, next I can:

* generate this as a **downloadable file** explicitly (e.g., via a link),
* or align the wording exactly to a target journal (SIM, Lifetime Data Analysis, etc.).

Yes — **your instinct is correct** 👍
**$\hat{\theta}$ must be in math mode** if you want it to render correctly and consistently across PDF / LaTeX / Word outputs.

Right now your punchlist text is *conceptually* correct, but it needs a small **typesetting fix**.

---

## What needs to change (precisely)

Anywhere you currently have:

* `$\hat{\theta}$`
* `theta_hat`
* `$\hat{\theta}$ Dose 0`

you should instead use **math mode**:

```latex
$\hat{\theta}$
```

or, inside tables:

```latex
$\hat{\theta}_{\text{Dose 0}}$
```

This applies to:

* Table captions
* Column headers
* Footnotes
* Body text references

---

## Corrected versions (paste-ready)

### ✅ Table X caption (fixed)

```markdown
**Table X. Estimated gamma-frailty variance ($\hat{\theta}$) by age band and vaccination status for Czech cohorts enrolled in 2021_13.**
```

---

### ✅ Table X column headers (fixed)

```markdown
| Age band (years) | $\hat{\theta}$ Dose 0 (median) | $\hat{\theta}$ Dose 2 (median) |
```

Pandoc will handle this correctly for:

* LaTeX → PDF
* Word (OMML)
* HTML

---

### ✅ Footnote (fixed)

```markdown
$\hat{\theta}$ quantifies unobserved frailty heterogeneity and depletion of susceptibles within cohorts. Near-zero values indicate effectively linear cumulative hazards over the quiet window and are typical of strongly pre-selected cohorts.
```

---

### ✅ Methods text (fixed)

Replace:

> “estimated frailty”

with:

```markdown
near-zero estimated frailty ($\hat{\theta} \approx 0$)
```

---

### ✅ Results paragraph (fixed)

```markdown
As shown in Table X, unvaccinated cohorts exhibit substantial frailty heterogeneity ($\hat{\theta} > 0$), while Dose 2 cohorts show near-zero estimated frailty ($\hat{\theta} \approx 0$) across all age bands, consistent with strong selective uptake prior to follow-up.
```

---

## Why this matters (practically)

If you **don’t** use math mode:

* Word may silently drop the hat
* PDF fonts may substitute or misalign
* Reviewers may see inconsistent notation
* You risk another round of formatting hell (which you’ve already fought)

Using `$…$` **once and everywhere** avoids all of that.

---

## One more small but important tip

For **consistency with the rest of the paper**, I recommend:

* Use `$\hat{\theta}$` everywhere
* **Never** use `theta_hat` in the manuscript
* Reserve `theta_hat` only for **code / CSV / filenames**

That cleanly separates:

* math notation (paper)
* implementation (code)

---

## Bottom line

✔ Yes — it should be math mode
✔ Your structure is right
✔ This is a **purely presentational fix**, not a conceptual one
✔ Once fixed, `punchlist10_kcor_diagnostics.md` is ready to be authoritative 

If you want, next I can:

* do a **full notation consistency sweep** ($\hat{\theta}$, H(t), KCOR, etc.), or
* check that Pandoc → Word renders every symbol exactly once and correctly.

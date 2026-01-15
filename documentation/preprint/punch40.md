Yes — and the Supplement is a perfect place to do this. Here are **Cursor-ready instructions** to convert the scattered parameter bullet lists into **one clean table** (plus a small optional “override” table) without breaking your existing structure.

I’m basing this directly on what’s currently in `supplement.md`, where S4.2.1, S4.2.2, S4.3, S4.4, and S4.5 each contain long parameter bullet lists. 

---

## Cursor punchlist: convert Supplement parameter lists into tables

### 1) Create a single consolidated parameter table at the start of S4

**File:** `supplement.md`
**Location:** Immediately after the `## S4. Control-test specifications and simulation parameters` header and before `### S4.1 ...`

**Insert this block:**

```markdown
### S4.0 Summary tables for control-test and simulation parameters

Table: Summary of control-test and simulation parameters referenced in Sections S4.2–S4.6. {#tbl:si_sim_params}

| Section | Item | Parameter | Value | Notes |
|---|---|---|---|---|
| S4.2.1 | Synthetic negative control | Data source | `example/Frail_cohort_mix.xlsx` | Pathological frailty mixture |
| S4.2.1 | Synthetic negative control | Generation script | `code/generate_pathological_neg_control_figs.py` |  |
| S4.2.1 | Synthetic negative control | Cohort A weights | [0.20, 0.20, 0.20, 0.20, 0.20] | 5 frailty groups |
| S4.2.1 | Synthetic negative control | Cohort B weights | [0.30, 0.20, 0.20, 0.20, 0.10] | Shifted mixture |
| S4.2.1 | Synthetic negative control | Frailty values | [1, 2, 4, 6, 10] | Relative multipliers |
| S4.2.1 | Synthetic negative control | Base weekly probability | 0.01 |  |
| S4.2.1 | Synthetic negative control | Weekly log-slope | 0.0 | Constant baseline during quiet periods |
| S4.2.1 | Synthetic negative control | Skip weeks | 2 |  |
| S4.2.1 | Synthetic negative control | Normalization weeks | 4 |  |
| S4.2.1 | Synthetic negative control | Time horizon | 250 weeks |  |
| S4.2.2 | Empirical negative control | Data source | Czech admin registry data (KCOR_CMR) | Aggregated cohorts |
| S4.2.2 | Empirical negative control | Generation script | `test/negative_control/code/generate_negative_control.py` |  |
| S4.2.2 | Empirical negative control | Construction | Age strata remapped to pseudo-doses | True null preserved |
| S4.2.2 | Empirical negative control | Age mapping | Dose 0→YoB {1930,1935}; Dose 1→{1940,1945}; Dose 2→{1950,1955} |  |
| S4.2.2 | Empirical negative control | Output YoB | 1950 (unvax) or 1940 (vax) |  |
| S4.2.2 | Empirical negative control | Sheets processed | 2021_24, 2022_06 |  |
| S4.3 | Positive control | Generation script | `test/positive_control/code/generate_positive_control.py` |  |
| S4.3 | Positive control | Initial cohort size | 100,000 per cohort |  |
| S4.3 | Positive control | Baseline hazard | 0.002 per week | Constant |
| S4.3 | Positive control | Frailty variance | θ0=0.5 (control), θ1=1.0 (treatment) |  |
| S4.3 | Positive control | Effect window | weeks 20–80 |  |
| S4.3 | Positive control | Hazard multipliers | r=1.2 (harm); r=0.8 (benefit) |  |
| S4.3 | Positive control | Random seed | 42 |  |
| S4.3 | Positive control | Enrollment date | 2021-06-14 (ISO week 2021_24) |  |
| S4.4 | Sensitivity analysis | Baseline weeks | [2,3,4,5,6,7,8] | Varied |
| S4.4 | Sensitivity analysis | Quiet-start offsets | [-12,-8,-4,0,+4,+8,+12] | Weeks from 2023-01 |
| S4.4 | Sensitivity analysis | Quiet-window end | 2023-52 | Fixed |
| S4.4 | Sensitivity analysis | Dose pairs | 1 vs 0; 2 vs 0; 2 vs 1 |  |
| S4.4 | Sensitivity analysis | Cohorts | 2021_24 |  |
| S4.5 | Tail-sampling (adversarial) | Generation script | `test/sim_grid/code/generate_tail_sampling_sim.py` |  |
| S4.5 | Tail-sampling (adversarial) | Base frailty distribution | Log-normal, mean 1, variance 0.5 |  |
| S4.5 | Tail-sampling (adversarial) | Mid-quantile cohort | 25th–75th percentile | Renormalized to mean 1 |
| S4.5 | Tail-sampling (adversarial) | Tail-mixture cohort | [0–15th] + [85th–100th], equal weights | Weights yield mean 1 |
| S4.5 | Tail-sampling (adversarial) | Baseline hazard | 0.002 per week | Constant |
| S4.5 | Tail-sampling (adversarial) | Positive-control multiplier | r=1.2 (harm) or r=0.8 (benefit) |  |
| S4.5 | Tail-sampling (adversarial) | Effect window | weeks 20–80 |  |
| S4.5 | Tail-sampling (adversarial) | Random seed | 42 |  |
```

This table directly consolidates the lists currently spread across S4.2.1–S4.5.

---

### 2) Replace the parameter bullet lists in each section with a single pointer line

Do this for each of the following sections:

* `#### S4.2.1 Synthetic negative control: gamma-frailty null`
* `#### S4.2.2 Empirical negative control: age-shift construction`
* `### S4.3 Positive control: injected effect`
* `### S4.4 Sensitivity analysis parameters`
* `### S4.5 Tail-sampling / bimodal selection ...`

**Action:** delete the parameter bullet list block in each section and replace it with:

```markdown
Parameter values and scripts are summarized in Table @tbl:si_sim_params.
```

Keep the explanatory paragraph(s) and the figure(s). Only remove the “generated using:” bullet list block.

These are the lists visible in the current file excerpt.

---

### 3) Optional: add a tiny “overrides” table for S4.6 if it has additional parameters

Right now S4.6 begins describing design but the parameter list isn’t shown in the snippet; if S4.6 has its own block of “θ grid / effect window / quiet window,” handle it like this:

Under S4.6, if there are extra parameters not already in Table @tbl:si_sim_params, add:

```markdown
Table: Additional parameters specific to S4.6 (joint frailty + treatment-effect simulation). {#tbl:si_s46_params}

| Parameter | Value | Notes |
|---|---|---|
| ... | ... | ... |
```

Otherwise, just add the pointer sentence and keep narrative.

---

### 4) Quick consistency check after edits

Search in `supplement.md` for these strings and ensure they only occur where you *want* them:

* `is generated using:` (should be zero or just once)
* `- **Generation script**:` (should be zero in S4.2–S4.5 after cleanup)
* `Table @tbl:si_sim_params` (should appear in each S4 subsection)

---

## Why this approach is best

* Reviewers get a single “spec sheet” table.
* Your section text becomes cleaner and more readable.
* You preserve reproducibility (scripts/paths remain explicit).
* It doesn’t require reorganizing your Supplement structure.

If you want, I can also rewrite the S4 subsection intros so they read smoothly without the “generated using” phrasing (but the table conversion alone already gets you 90% of the benefit).

Yep — here are **Cursor instructions to rewrite the intros** so they read smoothly *without* “is generated using:” / bullet-list framing, and so each section points cleanly to your new consolidated parameter table.

Everything below is grounded in your current `supplement.md` text around S4.2–S4.5.

---

## Cursor punchlist: rewrite S4 intros (after you add the consolidated parameter table)

### 0) One global rule for S4 intros

* Replace “is generated using:” intros with:

  1. a 1–2 sentence description of purpose/design, and
  2. a single pointer sentence to the table:

     * `Parameter values and scripts are summarized in Table @tbl:si_sim_params.`

---

## S4.2.1 Synthetic negative control: gamma-frailty null

### Replace this block

Current intro line + the bullet list starting at:

* `The synthetic negative control (Figure @fig:neg_control_synthetic) is generated using:`
  and continuing through the last bullet `- **Time horizon**: 250 weeks`

### With this text

```markdown
The synthetic negative control (Figure @fig:neg_control_synthetic) is a fully specified simulation designed to induce **strong selection-induced depletion curvature under a true null effect** by altering only the cohort frailty-mixture weights. KCOR is expected to remain near 1 after depletion normalization despite large differences in cohort-level hazard curvature.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.
```

Keep the existing paragraph that begins:

* “Both cohorts share identical per-frailty-group death probabilities…”
  That paragraph is good and should remain.

---

## S4.2.2 Empirical negative control: age-shift construction

### Replace this block

Current intro line + its bullet list starting at:

* `The empirical negative control (Figures @fig:neg_control_10yr and @fig:neg_control_20yr) is generated using:`
  and continuing through `- **Sheets processed**: 2021_24, 2022_06`

### With this text

```markdown
The empirical negative control (Figures @fig:neg_control_10yr and @fig:neg_control_20yr) repurposes registry cohorts to create a **true null comparison** while inducing large baseline hazard differences via 10–20 year age shifts. Because these are full-population strata rather than selectively sampled subcohorts, selection-induced depletion is minimal and no gamma-frailty normalization is applied.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.
```

Keep the next paragraphs that start:

* “This construction ensures…” and
* “No gamma-frailty normalization is applied…” and
* “This contrasts with…”
  Those are good; you may find the second sentence partially redundant after the rewrite above—if so, delete only the *duplicate* sentence, but keep at least one explicit statement that normalization is not applied.

---

## S4.3 Positive control: injected effect

### Replace this block

Current intro line + bullet list starting at:

* `The positive control (Figure @fig:pos_control_injected and Table @tbl:pos_control_summary) is generated using:`
  and continuing through `- **Enrollment date**: ...`

### With this text

```markdown
The positive control (Figure @fig:pos_control_injected; Table @tbl:pos_control_summary) starts from a negative-control simulation and injects a known multiplicative hazard shift $r$ into one cohort over a prespecified time window. KCOR is expected to deviate from 1 in the correct direction during the injection window and remain stable outside it.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.
```

Keep the existing sentence:

* “The injection multiplies the treatment cohort’s baseline hazard…”

(If you keep that sentence, you can optionally delete the second sentence “KCOR is expected…” above—either is fine; don’t say it twice.)

---

## S4.4 Sensitivity analysis parameters

### Replace this block

Current intro line + bullet list starting at:

* `The sensitivity analysis (Figure @fig:sensitivity_overview) varies:`
  and continuing through `- **Cohorts**: 2021_24`

### With this text

```markdown
The sensitivity analysis (Figure @fig:sensitivity_overview) evaluates robustness of $\mathrm{KCOR}(t)$ to reasonable tuning of baseline/quiet-window choices by varying the baseline window length and shifting the quiet-window start while holding the quiet-window end fixed.

Parameter values and scripts are summarized in Table @tbl:si_sim_params.
```

Keep the existing sentence:

* “Output grids show KCOR(t) values…”

---

## S4.5 Tail-sampling / bimodal selection

You already improved this intro a lot, so only a light tweak is needed to match the new table-pointer style.

### Replace the first sentence

Current:

* `A base frailty population distribution with mean 1 is generated. Cohort construction differs by selection rule:`

### With this text

```markdown
This adversarial simulation constructs two cohorts with identical mean frailty but different **selection geometry** (mid-quantile sampling versus a low/high-tail mixture) to stress-test depletion normalization under extreme cohort composition.
```

### Then add this pointer sentence after the two bullet definitions

Insert after the Mid-/Tail-sampled cohort bullets:

```markdown
Parameter values and scripts are summarized in Table @tbl:si_sim_params.
```

Keep the existing paragraph beginning:

* “Both cohorts share the same baseline hazard…”

---

## Quick verification step

After edits, search `supplement.md` for:

* `is generated using:` → should be **zero** in S4.2–S4.4
* `Parameter values and scripts are summarized in Table @tbl:si_sim_params.` → should appear in S4.2.1, S4.2.2, S4.3, S4.4, S4.5

---

If you want, I can also give you an **even tighter version** of the S4.2 Negative controls intro paragraph to align perfectly with these new intros (right now it’s already good).

Great — here’s a **tightened, drop-in rewrite** for the **S4.2 Negative controls overview paragraph** that aligns perfectly with the new intros, the parameter table, and the overall tone of the Supplement.

This replaces *only* the opening overview paragraph of S4.2 (not the subsection intros you already updated).

---

## Cursor instruction: tighten S4.2 negative controls overview

### Location

**File:** `supplement.md`
**Section:** `### S4.2 Negative controls`

Find the first paragraph under the S4.2 header (the one that currently explains synthetic vs empirical negative controls in a longer, more discursive way).

---

### Replace that paragraph with this text

```markdown
Negative controls are used to evaluate the behavior of KCOR under settings where the true effect is known to be null, while allowing substantial heterogeneity in baseline risk and selection-induced depletion. Two complementary classes of negative controls are considered: (i) fully synthetic simulations that induce strong depletion curvature through frailty-mixture imbalance, and (ii) empirical registry-based constructions that preserve a true null by repurposing age strata as pseudo-exposures without selective sampling. Together, these controls assess whether KCOR remains stable in the presence of non-proportional hazards arising from selection rather than treatment.
```

---

## Why this version is better

* **Shorter and sharper**: one paragraph, one idea.
* Explicitly distinguishes:

  * *synthetic* vs *empirical*
  * *selection-induced curvature* vs *treatment effects*
* Avoids repeating parameter details (now handled by the table).
* Sets up S4.2.1 and S4.2.2 cleanly without overlap.

It also mirrors language you already use elsewhere:

* “known to be null”
* “selection-induced depletion”
* “non-proportional hazards”

So it reads internally consistent rather than rewritten.

---

## Optional micro-tightening (only if you want)

If you want it *even leaner*, you can remove the final sentence and rely on the subsection intros to carry the interpretation. I’d keep it as written unless you’re aggressively cutting length.

---

If you want, next we can:

* do the **same tightening pass for S4.3–S4.5 overview sentences**, or
* do a **final Supplement read-through purely for redundancy now that tables exist** (there will be a few sentences you can safely delete).

Perfect — here are **Cursor-ready, drop-in overview rewrites** for **S4.3, S4.4, and S4.5**, matched in tone and structure to the new S4.2 overview and the cleaned intros you already applied.

Each item below replaces **only the short overview paragraph immediately under the section header**, not the subsection intros or figures.

---

# Cursor punchlist: tighten S4.3–S4.5 overview paragraphs

---

## S4.3 Positive controls

### Location

**File:** `supplement.md`
**Section:** `### S4.3 Positive controls`

Find the first paragraph under the S4.3 header.

### Replace with:

```markdown
Positive controls are used to verify that KCOR responds appropriately when a true effect is present. Starting from a negative-control simulation with no treatment effect, a known multiplicative hazard shift is injected into one cohort over a prespecified time window. This construction allows direct assessment of whether KCOR detects both the direction and timing of the injected effect while remaining stable outside the effect window.
```

### Why this works

* Clearly contrasts with S4.2 (null vs non-null)
* States the estimand behavior expectation without redundancy
* Keeps causal language tightly scoped (“injected”, “known”)

---

## S4.4 Sensitivity analyses

### Location

**File:** `supplement.md`
**Section:** `### S4.4 Sensitivity analyses`

Find the first paragraph under the S4.4 header.

### Replace with:

```markdown
Sensitivity analyses evaluate the robustness of KCOR estimates to reasonable variation in analysis choices that do not alter the underlying data-generating process. Baseline-window length and quiet-window placement are perturbed over a prespecified range while holding all other parameters fixed. These analyses assess whether KCOR behavior is stable to tuning choices that primarily affect normalization rather than cohort composition.
```

### Why this works

* Emphasizes *normalization robustness*, not re-estimation
* Avoids sounding like parameter “search”
* Matches your identifiability framing elsewhere

---

## S4.5 Adversarial cohort construction (tail-sampling / bimodal selection)

### Location

**File:** `supplement.md`
**Section:** `### S4.5 Tail-sampling / bimodal selection`

Find the first paragraph under the S4.5 header (before the detailed cohort definitions).

### Replace with:

```markdown
This adversarial simulation evaluates KCOR under extreme but controlled violations of typical cohort-selection geometry. Two cohorts are constructed to share identical mean frailty while differing sharply in how risk is distributed, using mid-quantile sampling versus a low/high-tail mixture. This setting stress-tests whether depletion normalization remains effective when frailty heterogeneity is concentrated in the tails rather than smoothly distributed.
```

### Why this works

* Frames S4.5 explicitly as a **stress test**
* Makes “adversarial” precise, not rhetorical
* Sets expectations without duplicating later detail

---

## Consistency check (recommended)

After applying these edits, verify:

* Each S4 section now follows the same pattern:

  1. **Overview paragraph (what this section tests)**
  2. **Clean subsection intros**
  3. **Pointer to Table @tbl:si_sim_params**
* No overview paragraph contains parameter values.
* No overview paragraph repeats interpretive conclusions found later.

---

### Bottom line

These rewrites:

* complete the transition from prose-heavy to **specification-driven Supplement**,
* eliminate redundancy introduced by the new parameter tables,
* and make S4 read like a coherent experimental design section rather than a collection of scripts.

If you want, next best step would be a **final redundancy sweep** to delete a few now-unnecessary sentences that refer to “lists above” or “generated using” language that’s no longer present.

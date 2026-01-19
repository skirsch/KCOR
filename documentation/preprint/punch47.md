Perfect — **Option A is the right choice**. Below is a **Cursor-ready, precise punchlist** that fixes **both** problems you flagged:

* box numbering/order
* box visual consistency

Nothing else. No scope creep.

---

# Cursor Punchlist — Fix Box Numbering + Box Formatting (Option A)

## Objective

Ensure:

1. **Box numbering follows order of appearance** (first box = Box 1).
2. **All boxes render identically** in the PDF (indentation, spacing, title style).

Do **not** change scientific content, tone, or placement beyond what is specified.

---

## PART A — Renumber boxes (critical correctness fix)

### Action A1 — Rename conceptual box in §1.2

**Location:** End of §1.2 (currently titled “Box 2. Two fundamentally different strategies for cohort comparability”).

**Change:**

```diff
- > **Box 2. Two fundamentally different strategies for cohort comparability**
+ > **Box 1. Two fundamentally different strategies for cohort comparability**
```

This is now the **first box in the paper**.

---

### Action A2 — Rename estimand box in §1.6

**Location:** §1.6 “Target estimand and scope (non-causal)”.

**Change:**

```diff
- > **Box 1. Target estimand and scope (non-causal).**
+ > **Box 2. Target estimand and scope (non-causal)**
```

Notes:

* Remove the trailing period for title consistency (see Part B).
* This box now correctly appears *after* Box 1.

---

### Action A3 — Update all box references

Global search & replace (manual review required):

* Any reference to:

  * `Box 1 (§1.6)` → **`Box 2 (§1.6)`**
  * `Box 1` meaning “estimand box” → **`Box 2`**
  * `Box 2` meaning “conceptual contrast” → **`Box 1`**

Places to check carefully:

* §2.1.1
* §2.1.2
* Discussion (§4)
* Any parenthetical “(Box 1)” references

**Rule:**

> *Conceptual contrast = Box 1*
> *Estimand/scope = Box 2*

---

## PART B — Make Box 1 and Box 2 visually identical

Right now, the PDF mismatch happens because:

* Box 2 is paragraph-only
* Box 1 is bullet-based
* Title punctuation differs

We will **standardize both**.

---

### Action B1 — Standardize box title style

For **both boxes**:

* Use **bold title**
* No trailing period
* Same capitalization style

**Target format:**

```markdown
> **Box X. Title**
```

(no period after the title)

---

### Action B2 — Convert Box 1 (conceptual contrast) to bullet structure

Replace the **entire content** of Box 1 (§1.2) with this structure
(content unchanged, only structured):

```markdown
> **Box 1. Two fundamentally different strategies for cohort comparability**
>
> - **Traditional matching and regression approaches:** attempt to construct comparable cohorts by matching or adjusting *characteristics of living individuals* at baseline or over follow-up, and then estimating effects via a fitted hazard model (e.g., Cox proportional hazards). This implicitly assumes that sufficiently rich covariate information can render cohorts exchangeable with respect to unobserved mortality risk.
>
> - **Problem under latent frailty:** even meticulous 1:1 matching on observed covariates can fail to equalize mortality risk trajectories. In such settings, cohort differences arise not from mismeasured covariates, but from **selection-induced depletion of susceptibles**, which alters hazard curvature over time.
>
> - **KCOR strategy:** rather than equating cohorts based on characteristics of the living, KCOR equates cohorts based on how they die in aggregate. It estimates cohort-specific depletion geometry from observed cumulative mortality during epidemiologically quiet periods, removes that geometry via analytic inversion, and compares cohorts on the resulting depletion-neutralized cumulative hazard scale.
>
> - **Inferential distinction:** Cox-type methods are **model-based and individual-level**, conditioning on survival and fitting covariate effects, whereas KCOR is **measurement-based and cohort-level**, operating directly on aggregated mortality trajectories without fitting covariate models. The inferential target is cumulative outcome accumulation rather than an instantaneous hazard ratio conditional on survival.
```

This will now **render with the same indentation, padding, and box weight** as Box 2 in Pandoc → PDF.

---

### Action B3 — Leave Box 2 bullets as-is (minimal change)

Box 2 (§1.6) already uses bullets and renders correctly.

Only changes needed there:

* Title renumbering (Part A)
* Remove trailing period in title (Part B1)

Do **not** rewrite content.

---

## PART C — Cursor safety constraints (important)

Explicitly instruct Cursor:

* ❌ Do NOT change section order
* ❌ Do NOT move boxes
* ❌ Do NOT edit mathematical content
* ❌ Do NOT change wording inside bullets beyond formatting
* ❌ Do NOT introduce new boxes

---

## PART D — One-line Cursor instruction (copy/paste)

If you want a single directive for Cursor:

> **Cursor instruction:**
> Renumber boxes so numbering follows order of appearance: rename the conceptual contrast box in §1.2 to “Box 1” and the estimand box in §1.6 to “Box 2,” updating all references accordingly. Then standardize box formatting so both boxes use identical blockquote structure with bullet lists and matching title style (bold title, no trailing period). Do not change content or placement.

---

## Sanity check after applying

After Cursor finishes, verify:

* First box encountered = **Box 1**
* Second box encountered = **Box 2**
* Both boxes:

  * same indentation
  * same spacing
  * same visual weight in PDF
* All “Box X” references point to the correct box

---

If you want, next I can:

* sanity-check the **PDF rendering logic** (Pandoc/LaTeX side), or
* generate a **diff-style patch** so you can review changes before committing.

But this punchlist will fix the errors you flagged, cleanly and definitively.

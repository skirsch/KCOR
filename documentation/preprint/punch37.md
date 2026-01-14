Absolutely. Below is a **Cursor-ready instruction punch list** that is explicit, mechanical, and safe to execute. This assumes **no content changes**, only **structural renumbering and reference fixes**.

---

# Cursor Instruction Punch List

**Task:** Re-organize negative controls under a single section and renumber all affected S4 subsections and cross-references.

---

## Goal (do not deviate)

* Create a single **S4.2 Negative controls** section
* Move **synthetic** and **empirical** negative controls under it as **S4.2.1** and **S4.2.2**
* Renumber all subsequent S4.x sections
* Update **all in-text references**, figure refs, and table refs to match the new numbering
* Do **not** alter wording, equations, figures, or captions beyond section numbers

---

## Step 1 — Create the new parent section

In `supplement.md`, locate:

```
### S4.2 Negative control: synthetic gamma-frailty null
```

### Replace with:

```
### S4.2 Negative controls
```

Immediately **below it**, insert:

```
#### S4.2.1 Synthetic negative control: gamma-frailty null
```

---

## Step 2 — Nest the existing synthetic negative control content

* Move **all content** that currently belongs to
  `S4.2 Negative control: synthetic gamma-frailty null`
* Ensure it now lives **under** `#### S4.2.1`
* Do not modify any text, figures, equations, or tables inside this block

---

## Step 3 — Nest the empirical negative control

Find the section currently titled:

```
### S4.3 Negative control: empirical age-shift construction
```

### Change it to:

```
#### S4.2.2 Empirical negative control: age-shift construction
```

* Ensure it is placed **after S4.2.1**
* Do not change the body text

---

## Step 4 — Renumber all subsequent S4 sections

Apply the following **exact renumbering map**:

| Old section                                        | New section                                            |
| -------------------------------------------------- | ------------------------------------------------------ |
| S4.4 Positive control: injected effect             | **S4.3 Positive control: injected effect**             |
| S4.5 Sensitivity analysis parameters               | **S4.4 Sensitivity analysis parameters**               |
| S4.6 Tail-sampling / bimodal selection             | **S4.5 Tail-sampling / bimodal selection**             |
| S4.7 Joint frailty and treatment-effect simulation | **S4.6 Joint frailty and treatment-effect simulation** |

Only change the **section numbers**, not titles or content.

---

## Step 5 — Update all in-text section references

Search globally in `supplement.md` for:

* `S4.2`
* `S4.3`
* `S4.4`
* `S4.5`
* `S4.6`
* `S4.7`

### Update references according to the same mapping in Step 4.

Examples:

* “see **Section S4.3**” → “see **Section S4.2.2**” (if referring to empirical negative control)
* “described in **S4.4**” → “described in **S4.3**” (positive control)

⚠️ **Important:**
Do not update figure or table numbers unless they explicitly reference section numbers in text (e.g., “Figure S4.2”).

---

## Step 6 — Verify figure and table references (no renumbering expected)

* Figures and tables use their own numbering (`Figure S1`, `Table S2`, etc.)
* **Do not** renumber figures or tables unless a caption explicitly names a section
* Only update prose like:

  * “(see Section S4.3)” → corrected number

---

## Step 7 — Update the S1 Overview list (if present)

In **S1 Overview**, update the bullet describing S4 so it reads conceptually correctly, e.g.:

```
- **S4**: Control-test specifications, including negative and positive controls
```

(No subsection numbers needed here.)

---

## Step 8 — Final validation checklist (Cursor must confirm)

Cursor should confirm:

* Exactly **one** `### S4.2 Negative controls`
* Exactly **two** subsections:

  * `#### S4.2.1 Synthetic …`
  * `#### S4.2.2 Empirical …`
* No remaining references to old section numbers
* No duplicated section numbers
* No content edits beyond numbering

---

## Reviewer-facing benefit (do not add to text)

This reorganization:

* Improves conceptual symmetry (negative vs positive controls)
* Matches standard validation logic
* Reduces reviewer confusion about why two nulls exist

---

If you want, next I can:

* Provide a **git diff–style expected output**, or
* Scan the **main manuscript** for any S4 cross-references that also need updating.

Great — given that **~11K words is squarely within recent SiM norms**, the goal now is **not aggressive cutting**, but **final compliance, polish, and risk-reduction**.

Below is a **Cursor-ready punchlist** calibrated to *“we are in the zone, don’t over-edit”*.

---

# Cursor Command File

**Project:** KCOR (Statistics in Medicine submission)
**Status:** In-range length, near-final
**Objective:** Final polish + compliance, no scope expansion

---

## COMMAND 0 — Lock length expectations (important)

**Context for Cursor:**

* Target length ≈ 10.5–11.5K words (main text)
* Do **not** aggressively shorten
* Do **not** move additional content to SI unless explicitly instructed

---

## COMMAND 1 — Verify appendix removal is complete

In `paper.md`:

1. Confirm **NO** sections titled:

   * “Appendix”
   * “Appendix A/B/C/D/E”
2. Confirm **NO** references like:

   * “Appendix Table…”
   * “Appendix Figure…”
3. Confirm **NO** appendix-lettered tables remain

If any remain:

* Remove them
* Replace references with “Supplementary Table Sx / Section Sx”

---

## COMMAND 2 — Cross-reference sanity check (main → SI)

In `paper.md`:

* Ensure all references to SI use:

  * “Supplementary Section Sx”
  * “Supplementary Figure Sx”
  * “Supplementary Table Sx”
* Ensure **no mixed styles** (e.g., “Appendix / SI” both used)

Do **not** attempt clickable cross-PDF links — text consistency only.

---

## COMMAND 3 — Supplementary Information final compliance check

In `supplement.md`:

1. Confirm section numbering style:

   * `S6`, `S6.1`, `S6.1.1` (NOT `S.6.1`)
2. Confirm all figures are labeled:

   * `Figure Sx`
3. Confirm all tables are labeled:

   * `Table Sx`
4. Confirm no references to:

   * “Appendix”
   * “main paper section X.Y”

SI must be **fully self-contained**.

---

## COMMAND 4 — Non-causal framing consistency (light touch)

In `paper.md`:

### 4.1 Keep (do NOT remove)

* Box 1 (primary scope/estimand statement)
* One non-causal sentence in Introduction
* One non-causal sentence in Discussion or Limitations

### 4.2 Remove only true redundancies

Search for repeated phrases like:

* “does not establish causality”
* “diagnostic rather than causal”
* “no causal inference is implied”

If the same idea is stated **multiple times in the same section**, keep the clearest one and delete the rest.

**Do NOT remove all mentions** — just deduplicate.

---

## COMMAND 5 — Tone scrub (micro-edits only)

In `paper.md`:

Replace any remaining language that could be read as applied/causal:

* “explains why” → “is consistent with”
* “reveals the effect of” → “illustrates estimator behavior under”
* “demonstrates that X causes” → “demonstrates how selection can induce”

**Do not rewrite paragraphs**, just soften phrasing where needed.

---

## COMMAND 6 — Methods section discipline (no restructuring)

In Methods:

* Keep full estimator definition and math
* Keep diagnostics description
* Keep validation logic

**But**:

* If any operational details appear twice (e.g., quiet-window rules, SKIP_WEEKS), delete the duplicate instance.
* Prefer references to SI for implementation minutiae.

Goal: clarity without shrinking scope.

---

## COMMAND 7 — Tables & figures placement (SiM-specific)

### 7.1 Main manuscript

* Ensure **all main tables appear after References**
* No SI tables embedded
* No appendix-style table captions

### 7.2 Figures

* Main figures: no “Appendix” or “SI” labels in captions
* SI figures: explicitly labeled `Figure Sx`

---

## COMMAND 8 — Build check (no content changes)

Run:

```bash
pandoc paper.md -o main.pdf
pandoc supplement.md -o supplement.pdf
```

Verify:

* `main.pdf` contains ONLY the main manuscript
* `supplement.pdf` is standalone and readable
* Section numbering looks clean (S6.1 etc.)

---

## COMMAND 9 — Final submission readiness checklist

Confirm:

* Main length ≈ 11K words (acceptable for SiM)
* SI clearly separated
* No appendix artifacts
* No causal claims
* Diagnostics emphasized over inference

At this point: **ready to submit**.

---

## END STATE

After these commands:

* Paper is **within SiM length norms**
* Structure is **editorially correct**
* Risk is limited to scientific disagreement, not presentation

---

If you want next:

* A **SiM cover letter** tuned to “methods + diagnostics”
* Or a **one-page reviewer rebuttal template** you can reuse later

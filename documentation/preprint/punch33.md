Absolutely. Below is a **Cursor-ready punchlist** that **incorporates the genuinely good ideas from Claude** while **explicitly rejecting the ones that don’t make sense for your current draft or SiM norms**.

This is calibrated to:
✔ ~11K words is acceptable
✔ No new analyses
✔ No structural rewrites
✔ Final-stage polish only

You can paste this directly into Cursor.

---

# Cursor Punchlist — Final Polish (Post-Claude Triage)

**Project:** KCOR methods paper
**Journal:** *Statistics in Medicine*
**Status:** In-range length, structurally sound
**Goal:** Selective polish + risk reduction (no scope expansion)

---

## COMMAND 0 — Scope lock (important)

**Do NOT do the following:**

* Do NOT collapse §§1.2–1.5 into a single section
* Do NOT target 5–6K words
* Do NOT add new examples, datasets, or methods
* Do NOT reorganize Methods section structure
* Do NOT add new main-text tables

**We are polishing, not rewriting.**

---

## COMMAND 1 — Adopt Claude’s best framing suggestion (1 sentence only)

In **Methods introduction** or **Discussion opening**, add **one** sentence clarifying primary framing:

> “KCOR is best understood as a *normalization framework* that removes selection-induced depletion curvature to restore comparability of cumulative hazards across cohorts, after which standard descriptive contrasts may be applied.”

Do **not** repeat this elsewhere.

---

## COMMAND 2 — Notation consistency cleanup (high value, low risk)

In `paper.md` and `supplement.md`:

### 2.1 Hat notation

* Ensure **estimated parameters** consistently use hats (e.g., `θ̂`, `k̂`)
* Reserve un-hatted symbols for population/working-model quantities

If mixed usage exists:

* Standardize to hats for estimates
* Do not introduce new notation

---

### 2.2 Discrete vs continuous time clarification

Add **one sentence** (Methods early):

> “All analyses are performed using discrete weekly time bins; continuous-time notation is used solely for expositional convenience.”

Do not elaborate further.

---

## COMMAND 3 — Figure and reference hygiene (blocking issues only)

### 3.1 Fix unresolved references

Search for:

```
??
```

Resolve all:

* Figure references
* Table references
* Equation references

No placeholders may remain.

---

### 3.2 Time-axis consistency

Ensure:

* Each figure uses **either weeks or calendar dates**
* Captions clearly state which is used

Do **not** replot figures; caption clarification is sufficient.

---

## COMMAND 4 — Abstract tightening (recommended)

In the abstract:

* Target ~250–275 words (no hard cut needed)
* Ensure:

  * “Non-causal diagnostic framework” appears in the **first sentence**
  * Negative controls are mentioned explicitly
  * Scope is methods-first, application-second

Do not rewrite from scratch — tighten phrasing only.

---

## COMMAND 5 — Reject Claude’s over-cutting suggestions (explicit)

**Do NOT:**

* Collapse §§1.2–1.5 into ~500 words
* Move the main Cox failure demonstration to SI
* Add a non-vaccine example
* Create a condensed assumptions/diagnostics table in the main text
* Reorganize Methods section numbering

These suggestions are **outdated for the current draft** and should be ignored.

---

## COMMAND 6 — Micro-trim only within §§1.2–1.5 (optional, safe)

If any **exact idea** is stated twice across §§1.2–1.5:

* Remove the weaker or more verbose instance
* Preserve section boundaries and headers

Target:

* ~5–10% reduction max
* No structural fusion

If nothing obvious, skip this command.

---

## COMMAND 7 — SI label hygiene (final check)

In `supplement.md`:

* Ensure all labels use flat `si_*` prefix

  * No `appendix*` remains
* Ensure section numbering style is:

  * `S6`, `S6.1`, `S6.1.1` (not `S.6.1`)
* Ensure assumptions table and diagnostics table are **adjacent**

No other SI restructuring.

---

## COMMAND 8 — Final build check

Run:

```bash
pandoc paper.md -o main.pdf
pandoc supplement.md -o supplement.pdf
```

Verify:

* `main.pdf` has no appendices
* `supplement.pdf` is standalone
* References resolve cleanly
* Figures/tables appear in correct order

---

## END STATE (what “done” looks like)

* Main text ≈ 11K words
* Clear normalization framing
* Consistent notation
* No placeholder refs
* No appendix artifacts
* SI clean, adjacent, and self-contained

At this point: **submit**.

---

If you want next, I can:

* Draft a **cover letter** that subtly rebuts the “why not Cox / why COVID” objections, or
* Help you write a **one-paragraph response template** in case reviewers ask for more causal framing later.

But editorially: **you’re done after this punchlist.**

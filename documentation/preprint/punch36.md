Great — here is a **Cursor-ready command file**, cleanly structured and scoped so you can drop it in and run it without reopening settled decisions.

You can save this exactly as something like:

```
cursor_commands/punch36_final_polish.md
```

and execute section by section.

---

# Cursor Command File — Final Polish Before Submission (SiM)

**Scope constraints (DO NOT VIOLATE):**

* Do **not** change estimands, equations, or assumptions
* Do **not** merge assumptions, diagnostics, or identifiability
* Do **not** add causal language or new method comparisons (e.g., MSM/IPW)
* Do **not** restructure SI beyond mechanical fixes
* Focus only on clarity, length, references, and mechanical correctness

---

## COMMAND 1 — Fix unresolved cross-references (HIGH PRIORITY)

**Files:** `paper.md`, `supplement.md`

1. Search for unresolved references:

   * `Figure ??`
   * `Table ??`
   * Broken `@fig:*` or `@tbl:*` references
2. Ensure every figure and table has:

   * A unique `{#fig:...}` or `{#tbl:...}` label
   * At least one matching in-text reference
3. Do **not** rename labels unless required to fix collisions.
4. Rebuild the PDF and confirm **zero unresolved references**.

---

## COMMAND 2 — Length compression (surgical, not global)

### 2A. Introduction compression

**Target sections:** §§1.4–1.6

* Consolidate into a **single concise motivation block**
* Remove repeated restatements of:

  * Cox model limitations
  * depletion of susceptibles
  * non-causal disclaimers (already handled elsewhere)

**Rule:** each conceptual idea appears **once** in the Introduction.

---

### 2B. Discussion & Conclusion compression

**Target sections:** §4, §5, §6

* Remove sentences that merely restate Results
* Collapse redundant bullet lists into prose where possible
* Reduce §6 Conclusion to **one tight paragraph** (no repetition)

---

## COMMAND 3 — Abstract and Title tightening (editor-facing)

### 3A. Abstract

* Reduce method logistics
* Emphasize:

  1. The problem: selection-induced non-PH curvature
  2. The solution: diagnostic-first normalization in cumulative-hazard space
  3. The result: suppression of spurious non-nulls and transparent failure signaling

**Constraint:** remain within journal word limit.

---

### 3B. Title (optional but recommended)

If edited, use a shorter, editor-friendly formulation such as:

> **KCOR: A depletion-neutralized framework for retrospective cohort comparison under latent frailty**

Do not introduce new claims.

---

## COMMAND 4 — Tone cleanup (mechanical only)

**Files:** `paper.md`

* Replace conversational or editorial phrasing with neutral constructions.
* Examples to fix:

  * “we did not pursue … since …”
  * “this outcome is data-driven”
* Replace with:

  * “This was outside the scope because …”
  * “The analysis focuses on …”

Ensure:

* No first-person singular (“I”)
* Declarative, methods-journal tone throughout

---

## COMMAND 5 — Clarify empirical illustration provenance (CRITICAL)

**Goal:** prevent reviewer confusion about real vs constructed data.

**Actions:**

1. At first appearance of Czech data in Results:

   * Add one clarifying sentence stating analyses are **constructed negative/positive controls**.
2. In relevant figure and table captions:

   * Explicitly state when results use:

     * age-shift pseudo-controls
     * injected effects
     * illustrative registry recodings

Do **not** add additional disclaimers elsewhere.

---

## COMMAND 6 — Bootstrap / uncertainty clarification

**File:** `paper.md`

* In the uncertainty quantification section:

  * Explicitly state the resampling unit:

    * cohort-level counts (not individuals)
* Keep language factual, not advisory.

Example style:

> “Bootstrap resampling was performed at the cohort-count level, consistent with the aggregated data structure.”

---

## COMMAND 7 — Supplementary Information mechanical checks

**File:** `supplement.md`

* Verify:

  * Correct S-numbering (S1, S2, …)
  * Correct table and figure numbering
* Ensure no duplicated headings + table captions
* Confirm all remaining “recommended” / “optional” language is removed

Do **not** restructure SI sections.

---

## COMMAND 8 — References consistency

**File:** `paper.md`

* Ensure single reference style throughout (numeric / Vancouver)
* Add missing DOIs where available
* Remove duplicate or malformed entries

---

## COMMAND 9 — Final build sanity check

1. Build main paper PDF and SI PDF
2. Confirm:

   * No unresolved references
   * No stray “??”
   * Figures render correctly (vector preferred)
3. Spot-check:

   * Abstract
   * Box 1
   * §5 Limitations (non-causal note remains only here)

---

## Explicitly out of scope

* ❌ No asymptotic theory
* ❌ No MSM/IPW comparisons
* ❌ No assumption/diagnostic merging
* ❌ No causal reframing
* ❌ No new figures unless required to fix references

---

**End of command file**

---

If you want next, I can:

* sanity-check the **post-Cursor diff**, or
* help you finalize the **submission package** (cover letter + SI upload order) for SiM.

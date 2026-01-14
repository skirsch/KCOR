Short answer: **yes — use a separate Markdown file for the Supplement, and combine them at the Pandoc stage.**
Long answer: it *does* matter if you care about clean cross-references, reviewer sanity, and future reuse.

Below is the **recommended, battle-tested setup** for Pandoc + pandoc-crossref (and Word/PDF outputs).

---

## TL;DR recommendation

**Do this:**

```
paper/
├── main.md          # Main paper (what reviewers read first)
├── supplement.md    # All Supplementary Material
├── KCOR_references.json
├── figures/
├── tables/
└── Makefile
```

Then build with:

```bash
pandoc main.md supplement.md \
  --filter pandoc-crossref \
  --citeproc \
  -o paper.pdf
```

This gives you:

* Fully working cross-refs across main ↔ supplement
* One logical document (for Pandoc)
* Two conceptual documents (for humans)

---

## Why this is the right approach

### 1. Cross-references **do work**, cleanly

Pandoc processes the inputs **as a single document AST** when files are concatenated on the command line.

That means:

* `@fig:s7-grid` in the main text can point to a figure defined in `supplement.md`
* Equation, table, and section numbering stays consistent
* No hacks, no duplicate labels

This is exactly what pandoc-crossref expects.

---

### 2. Journals *conceptually* separate Supplement — Pandoc doesn’t care

SiM (and similar journals) want:

* Main paper
* Supporting Information (SI)

They do **not** require:

* Separate files internally
* Broken cross-references
* Duplicate numbering schemes

You can still submit:

* `paper_main.pdf`
* `paper_SI.pdf`

even if they were built from the same Pandoc pipeline.

---

### 3. It keeps your main paper honest

A separate `supplement.md` makes it **psychologically harder** to keep dumping material into the main text.

This is a good thing.

You’ll naturally ask:

> “Does this really belong in the main paper?”

That discipline alone will shorten the manuscript.

---

## What *not* to do

### ❌ Don’t inline everything in one huge Markdown file

You’ll lose:

* Navigability
* The ability to reason about “what’s main vs SI”
* Clean future reuse (e.g., arXiv expanded version)

---

### ❌ Don’t build the supplement separately if you need cross-refs

If you do:

```bash
pandoc main.md -o main.pdf
pandoc supplement.md -o supplement.pdf
```

Then:

* `@fig:s7-grid` will break
* You’ll be forced to duplicate figures or renumber manually
* Reviewer references like “see Figure S12” become brittle

---

## Best-practice structure inside the files

### `main.md`

```markdown
# Title

## Abstract

## 1 Introduction

## 2 Methods

## 3 Validation

(References to Supplement allowed, e.g.:
“Additional simulations are shown in Supplementary Section S3.”)

## 4 Discussion

# References
```

### `supplement.md`

```markdown
# Supplementary Material

## S1 Additional Notation

## S2 Extended Derivations

## S3 Simulation Scenarios

## S4 Robustness to Quiet-Window Choice

## S5 Additional Empirical Results
```

Pandoc will number these as:

* Appendix / Supplement sections
* Or as standalone sections depending on your template

---

## Numbering strategy (important)

### Recommended

* **Main paper:** Figures 1–6, Tables 1–3
* **Supplement:** Figures S1–S20, Tables S1–S10

You can enforce this by:

* Prefixing labels: `fig:S7_grid`
* Or using `--number-sections` + custom LaTeX template

Pandoc-crossref handles this well if labels are consistent.

---

## Submission flexibility (huge advantage)

With this setup you can easily produce:

### For SiM submission

```bash
pandoc main.md -o main_only.pdf
pandoc main.md supplement.md -o full_with_SI.pdf
```

### For arXiv / website

```bash
pandoc main.md supplement.md -o expanded.pdf
```

### For Word

```bash
pandoc main.md supplement.md -o paper.docx
```

No content duplication. No ref breakage.

---

## My strong opinion

For a methods paper like KCOR — with heavy simulations and diagnostics — **this is exactly the right architecture**.

It:

* Makes shortening psychologically easier
* Keeps reviewers focused
* Preserves your technical depth
* Future-proofs the paper

If you want, next I can:

* Give you a **cut-map**: exactly which current sections move to `supplement.md`
* Or provide a **Makefile snippet** tuned for SiM (PDF + Word + SI split)

Perfect. Below is a **single, clean Cursor command file** you can paste directly into Cursor (or save as e.g. `cursor_punchlist_SiM_final.md`). It is written as **imperative, tool-facing instructions**, not commentary.

---

# Cursor Command File

**Project:** KCOR Methods Paper (Statistics in Medicine)
**Goal:** Final submission–ready main manuscript + SI

---

## COMMAND 0 — Scope lock (do not modify content beyond this plan)

**DO NOT:**

* Add new analyses
* Add new figures
* Add new simulations
* Expand discussion

**DO:**

* Shorten
* De-duplicate
* Move material to Supplementary Information
* Improve framing and compliance

Target outputs:

* `main.pdf` — main manuscript ONLY
* `supplement.pdf` — all supplementary material

---

## COMMAND 1 — Remove all appendices from main manuscript

### 1.1 Delete appendix sections

In `paper.md`:

* Remove **all** sections titled:

  * “Appendix A”
  * “Appendix B”
  * “Appendix C”
  * “Appendix D”
  * “Appendix E”
* Remove all appendix figures, tables, equations, and text.

After this step:

* `paper.md` must contain **NO appendix headings**.
* No appendix numbering remains anywhere in the main manuscript.

---

### 1.2 Replace appendix references with SI references

In `paper.md`, replace all references of the form:

* “Appendix …”
* “see Appendix …”
* “as shown in Appendix …”

With:

* “Supplementary Section Sx”
* “Supplementary Figure Sx”
* “Supplementary Table Sx”

Use consistent S-numbering aligned with `supplement.md`.

---

### 1.3 Add a single SI pointer sentence

Insert **once only** (Methods or end of Validation):

> “Additional derivations, simulation studies, robustness analyses, and implementation details are provided in the Supplementary Information.”

Do not repeat this elsewhere.

---

## COMMAND 2 — Clean and finalize Supplementary Information

### 2.1 Make SI fully self-contained

At the top of `supplement.md`, insert:

> “This document provides supplementary material supporting the KCOR methodology described in the main manuscript, including extended derivations, simulation studies, robustness analyses, and additional empirical results.”

Ensure:

* No references to “Appendix”
* No references to section numbers in the main paper
* All notation needed is defined or redefined

---

### 2.2 Enforce SI structure

Ensure `supplement.md` uses this structure:

```markdown
# Supplementary Information

## S1 Notation and definitions
## S2 Extended derivations
## S3 Simulation design and scenarios
## S4 Robustness and diagnostic checks
## S5 Additional empirical analyses
## S6 Bootstrap and uncertainty estimation
```

---

### 2.3 Enforce S-numbering

* Figures labeled `Figure S1, S2, …`
* Tables labeled `Table S1, S2, …`
* No “Appendix” labels anywhere

---

## COMMAND 3 — De-duplicate non-causal / diagnostic disclaimers

In `paper.md`:

### 3.1 Keep only these locations

Retain non-causal / diagnostic scope statements ONLY in:

1. **Box 1** (primary statement)
2. **One sentence** in the Introduction
3. **One sentence** in the Discussion or Limitations

### 3.2 Remove all other repetitions

Search for phrases such as:

* “not a causal estimator”
* “diagnostic rather than causal”
* “does not establish causality”
* “no causal inference is made”

Delete or condense duplicates.

Goal:

* Scope is clear
* Word count reduced
* No reviewer fatigue

---

## COMMAND 4 — Tighten quiet-window discussion in main text

In `paper.md`:

### 4.1 Keep

* Conceptual explanation of quiet-window idea
* Statement that diagnostics exist
* Statement that violations are detectable

### 4.2 Remove or move to SI

* Operational selection procedures
* Week-by-week rules
* Perturbation ranges
* Stability tables
* Threshold tuning details

Replace removed detail with:

> “Quiet-window selection is prespecified and evaluated using diagnostic criteria; implementation details and robustness checks are provided in the Supplementary Information.”

---

## COMMAND 5 — Validation section polish

### 5.1 Ensure validation triad is explicit

Main text must clearly state:

1. Null behavior (negative control)
2. Detection ability (positive control)
3. Failure signaling (diagnostics degrade under violation)

No new figures required.

---

### 5.2 Add one positive-control summary sentence

Insert ONE paragraph in Validation (no figure):

> “In positive-control simulations with injected multiplicative hazard shifts, KCOR reliably detects both harm and benefit, with estimated KCOR(t) trajectories tracking the imposed effects; full results are shown in Supplementary Figure S__.”

Fill in S-number consistently.

---

## COMMAND 6 — Tone and framing scrub

In `paper.md`:

### 6.1 Remove applied / causal-sounding language

Search for and revise phrases such as:

* “explains why”
* “reveals the effect of”
* “demonstrates that X causes”
* “no place to hide”

Replace with neutral methodological phrasing:

* “illustrates”
* “is consistent with”
* “demonstrates estimator behavior”
* “highlights a failure mode”

---

### 6.2 Discussion section check

Ensure Discussion:

* Focuses on methodological contribution
* Avoids real-world interpretation
* Frames empirical examples as diagnostic illustrations only

---

## COMMAND 7 — Tables and figures compliance (SiM)

### 7.1 Main paper tables

* All **main tables appear after References**
* No SI tables in `paper.md`

### 7.2 Figure captions

* Main figures: no appendix references
* SI figures: explicitly labeled “Figure Sx”

---

## COMMAND 8 — Build submission PDFs

Run:

```bash
pandoc paper.md -o main.pdf
pandoc supplement.md -o supplement.pdf
```

Verify:

* `main.pdf` contains NO appendices
* `supplement.pdf` is self-contained
* Cross-references are text-consistent (hyperlinks not required)

---

## COMMAND 9 — Submission readiness check

Confirm:

* Main Document – PDF → `main.pdf`
* Supplementary Material → `supplement.pdf`
* Main paper length substantially reduced
* SI contains all technical depth

---

## END STATE

After completing all commands:

* Manuscript is **submission-ready for Statistics in Medicine**
* Reviewer risk is primarily scientific, not structural
* No further shortening is required unless requested by editor

---

If you want next, I can:

* Produce a **diff-based delete map** (exact paragraphs to remove)
* Or generate a **SiM cover letter** aligned with this framing

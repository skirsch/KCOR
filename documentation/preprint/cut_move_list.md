# Cut/move list for SIM shorten-first pass (`paper.md`)

This file operationalizes the plan in `sim_shorten-first_v2` **without editing the plan file itself**. It is meant to be a working checklist while revising the manuscript.

## Targets

- **Main paper**: ~6,500–7,500 words (up to ~8,000 if needed), doing only 3 jobs:
  - Define KCOR + estimand (and explicitly non-causal scope)
  - Show **one** compelling standard-method failure under frailty depletion
  - Show KCOR works + self-diagnoses failure via exactly **3** examples (neg/pos/stress)
- **Online Supplement**: everything else (full simulation grids, robustness catalogs, extended comparisons, extended Czech analyses, extra diagnostics).

## Rough size map (computed from headings; overcounts math, but useful for priorities)

Top-level sections (rough word counts):

- **1. Introduction**: ~3,524  
- **2. Methods**: ~6,309  
- **3. Validation and control tests**: ~3,922  
- **4. Discussion**: ~1,766  
- **5. Limitations**: ~1,498  

Largest subsections to focus cuts/moves:

- **1.3 Related work…**: ~1,337 (incl. 1.3.1/1.3.1a/1.3.1b/RMST)  
- **1.8 Competing approaches and evaluation plan**: ~404  
- **2.1 Conceptual framework and estimand**: ~1,001  
- **2.11 Relationship to Cox proportional hazards**: ~1,266 (incl. 2.11.1 demo ~883)  
- **3.4 Simulation grid**: ~1,687  
- **3.1.2 Czech empirical negative control**: ~515  

## Keep / cut / move decisions (main vs supplement)

### Abstract / Key messages

- **Keep**: short, method-first, non-causal positioning *once*.  
- **Change**: remove extra “scope defense” repetition; focus on the 3-job story.

### 1. Introduction

- **Keep (tighten)**: 1.1–1.2 (problem + “curvature is the hard part”)  
- **Cut/condense hard**:
  - **1.3 Related work** → compress to ~3–6 paragraphs total in main; move the detailed subsections to supplement.
  - **1.4 Evidence from literature** → compress or move to supplement (keep 1 short motivating paragraph + 1–2 citations max).
  - **1.6–1.8 (estimand/causal frameworks/competing approaches)** → consolidate into:
    - a single early **boxed estimand + non-causal scope** paragraph (main), and
    - a single “KCOR vs alternatives” **table** (main),
    - move remaining prose to supplement.

### 2. Methods

- **Keep (main)**:
  - **2.1** but rewrite to be short and operational; the estimand box should do much of the work.
  - **2.3–2.6** (hazard estimation, gamma-frailty identity/inversion, normalization) in minimal form.
  - **Quiet-window selection protocol (operational)**, but as a short checklist + “failure signals”.
  - **Algorithm summary** (2.10) and minimal reproducibility checklist.
- **Move to supplement**:
  - long theoretical exposition and repeated justification prose.
  - detailed uncertainty quantification discussion and any extended bootstrap diagnostics tables.
- **Cox section**:
  - **Main**: keep **2.11.1** only, shortened to a single canonical demo with one figure/table and a crisp takeaway.
  - **Supplement**: 2.11.2 and other extended comparisons; any repeated Cox rebuttal prose.
- **Worked example**:
  - **Move** to supplement (or shorten to a half-page “how-to” if it’s mostly procedural).

### 3. Validation and control tests (exactly 3 examples in main)

Main text should keep exactly:

1) **Negative control (main)**: keep **3.1.2 Czech age-shift negative control** (one clean figure + short explanation).  
   - Move **3.1.1 synthetic** to supplement (still valuable, just too much for main).

2) **Positive control (main)**: keep **3.2 injected harm/benefit** (one figure + takeaway).

3) **Stress test (main)**: keep **3.3.1 frailty misspecification robustness** as the “diagnostics light up” example.

Move to supplement:

- **3.4 Simulation grid** (entire)  
- **3.5 Dynamic HVE diagnostic tests**  
- **3.6 Illustrative non-COVID example**  
- any additional stress tests or catalogs beyond the single stress test above

### 4. Discussion / 5. Limitations

- **Keep (main)**:
  - one concise section “What KCOR estimates” (no repeats)
  - one concise “Limits of attribution / non-identifiability” paragraph
  - one concise “Practical guidance” section (bulleted)
  - “Failure modes and diagnostics” as a short checklist with pointers to supplement
- **Move or cut**:
  - long restatements of non-causality and positioning (keep only 1–2 intentional placements).

### Appendices / Tables

- Treat Appendices A–E as **Online Supplement** material; keep referenced items but ensure the main text points to them cleanly.
- Move dense tables/heatmaps/grids out of “Main text tables” if they are not essential to the 3-job narrative.

## Concrete edit sequence (what we’ll do next)

1. Insert the **estimand + non-causal scope** box early; delete repeated disclaimers elsewhere.
2. Reduce Introduction to ~2–3 pages: keep 1.1–1.2, compress 1.3–1.8.
3. Methods: keep minimal gamma-frailty + normalization + quiet-window checklist; move worked example and extended details to supplement.
4. Cox: keep **one** demo only.
5. Validation: keep exactly 3 examples (Czech negative, injected positive, misspec stress); move full grid and extras to supplement.
6. Add a short **Online Supplement Overview** paragraph + a supplement TOC pointer so reviewers can find moved material quickly.


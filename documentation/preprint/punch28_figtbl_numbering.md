Here’s a **Cursor-ready punch list** that implements the SIM table rules **and** fixes the “auto-number + manual number” duplication for **both tables and figures** in your Pandoc→LaTeX pipeline.

You can paste this straight into Cursor as an instruction block.

---

## Cursor punch list: SIM tables-at-end + appendix lettered numbering for tables & figures

### Goal

* SIM compliance: **all tables on separate pages after References** (not inline anywhere).
* Keep figures flexible (you can keep them embedded).
* Fix numbering so:

  * Main text tables/figures: **Table 1, Figure 1, …**
  * Appendix tables/figures: **Table A.1, Figure C.2, …**
* Remove any manual “D.1” / “C.3” from captions (LaTeX will generate it).

---

### Step 0 — Identify build path assumptions (don’t change behavior)

* Assume: **Pandoc → LaTeX/PDF**.
* Keep your current template/class unchanged.

---

### Step 1 — Add LaTeX header-includes to enforce appendix numbering for BOTH tables & figures

In the YAML header of `paper.md`, add (or merge) the following:

```yaml
header-includes:
  - \usepackage{chngcntr}
  - \usepackage{etoolbox}
  - \pretocmd{\appendix}{%
      \counterwithin{table}{section}%
      \counterwithin{figure}{section}%
      \renewcommand{\thetable}{\thesection.\arabic{table}}%
      \renewcommand{\thefigure}{\thesection.\arabic{figure}}%
    }{}{}
```

Notes:

* This hooks directly onto `\appendix` (more reliable than relying on an `appendix` environment).
* It applies to **both** tables and figures.

---

### Step 2 — Ensure LaTeX enters appendix mode exactly once

Right after the end of the References section (and after the Tables section, see Step 3), insert:

````markdown
```{=latex}
\appendix
````

````

Important:
- `\appendix` must appear **before** “Appendix A” begins.
- Only insert it once.

---

### Step 3 — Move ALL tables to a single “Tables” block after References
SIM requires tables after references. Do this:

1) Create this block immediately after `## References` ends:

```markdown
\newpage
## Tables
````

2. Under `## Tables`, group tables by logical location using subheadings. Use this exact pattern:

```markdown
### Main text tables

(Table 1 definition)
(Table 2 definition)
...

### Appendix A tables

(Table definition(s) that are referenced in Appendix A)

### Appendix B tables
...

### Appendix C tables
...

### Appendix D tables
...

### Appendix E tables
...
```

3. **Remove every table definition** from:

* the main text sections
* the appendix sections

Replace each removed in-place table with a simple callout line (optional but helpful for reviewers), e.g.:

* In main text:

  * “(Table 2 about here.)”
* In Appendix C:

  * “(Table C.1 about here.)”

These callouts won’t affect numbering; they just help readability.

---

### Step 4 — Fix table captions so LaTeX generates the numbering (no manual D.1 / C.2)

For every table caption currently written like:

* `Table: Table D.1. ...`
* `Table: D.1. ...`
* `Table: Table 3. ...`

Change to **title-only** captions:

✅ Use:

* `Table: KCOR assumptions and corresponding diagnostics.`

Do **not** include:

* “Table”
* “D.1”
* “A.2”
* any manual numbering in the caption text

LaTeX will produce:

* `Table 1. ...` for main text tables
* `Table D.1. ...` for tables under “### Appendix D tables” once appendices are active

---

### Step 5 — Appendix headings must be true top-level sections

Ensure appendices start as:

```markdown
## Appendix A: ...
## Appendix B: ...
## Appendix C: ...
```

And subsections as:

```markdown
### A.1 ...
### A.2 ...
```

Do NOT use `### Appendix A` (that makes it a subsection and breaks numbering + TOC hierarchy).

---

### Step 6 — Fix figures the same way (remove manual numbering in captions)

If any figure captions or references contain manual numbering like:

* “Figure C.2: …”
* “Figure S1 …”

Do this:

1. In captions, remove “Figure C.2” / “S1” from the caption text.
2. Let LaTeX generate “Figure C.2” automatically in appendices.

For embedded markdown images like:

```markdown
![Figure C.2. KCOR trajectories ...](path.png)
```

Change to:

```markdown
![KCOR trajectories ...](path.png)
```

---

### Step 7 — Make appendix figure/table references consistent

* Replace any “Supplementary Figure S…” / “Table S…” text with appendix-letter references:

  * “Figure C.2”
  * “Table D.1”
* Ensure you’re not calling anything “Supplementary” if it’s in the main manuscript file.

---

### Step 8 — Verify numbering will work given tables are defined before appendix text

This is the key point for SIM formatting:

Because all tables are after references, **appendix tables must be in appendix-letter subheadings inside the Tables section** so the “section context” is correct when LaTeX assigns numbers.

So confirm the Tables section includes headings like:

```markdown
### Appendix D tables
```

and appendix D tables are placed under that.

---

### Step 9 — Final compile checks

After edits, run the build and verify:

1. In PDF:

* Main text: Table 1, Table 2 … (no “D.1” anywhere)
* Appendix tables: Table A.1, Table C.1, Table D.1 …
* Appendix figures: Figure A.1, Figure C.2 …

2. No caption shows “Table 24 D.1” or “Figure 12 C.3”.
3. No tables appear inline in main text or appendices.

---

Updated include so that includes equations:

```
header-includes:
  - \usepackage{chngcntr}
  - \usepackage{etoolbox}
  - \pretocmd{\appendix}{%
      \counterwithin{table}{section}%
      \counterwithin{figure}{section}%
      \counterwithin{equation}{section}%
      \renewcommand{\thetable}{\thesection.\arabic{table}}%
      \renewcommand{\thefigure}{\thesection.\arabic{figure}}%
      \renewcommand{\theequation}{\thesection.\arabic{equation}}%
    }{}{}
```

IMPORTANT:
- Do NOT manually number equations (e.g., “(C.1)”, “(D.2)”) anywhere in the manuscript.
- All equation numbering in appendices is generated automatically by LaTeX after \appendix.
- Use \label / @eq: references only. so get rid of the \qquad and manual number and number like in the main body of the paper.


IMPORTANT:
- Do NOT manually include table or figure numbers (e.g., “Table D.1”, “Figure C.2”) in caption text.
- All captions must be title-only.
- Appendix letter/number prefixes are generated automatically by LaTeX after \appendix.

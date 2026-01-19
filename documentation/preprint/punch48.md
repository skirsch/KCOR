Absolutely. Here’s a **Cursor-ready punchlist** to add real borders around your “Box” blockquotes by modifying **`header.tex`** (no changes to the Markdown needed).

---

# Cursor Punchlist — Put a Border Around Box Blockquotes (via `header.tex`)

## Goal

Render Markdown blockquotes (`> ...`) as **boxed panels** in the PDF output by overriding the LaTeX `quote` environment in `header.tex`.

Constraints:

* No Markdown edits required
* Black border, white background, professional look
* Avoid shading/colors
* Keep spacing tight and consistent

---

## Step 1 — Open `header.tex`

Locate the LaTeX preamble file used by Pandoc during PDF build (you said it’s `header.tex`).

---

## Step 2 — Add `tcolorbox` package

In `header.tex`, add (near other `\usepackage{...}` lines):

```tex
\usepackage[most]{tcolorbox}
```

If `tcolorbox` is already present, do not duplicate.

---

## Step 3 — Override the `quote` environment to be boxed

Add the following block **after** the `\usepackage[most]{tcolorbox}` line (or anywhere after package imports in the preamble):

```tex
% Box all Markdown blockquotes (Pandoc uses the quote environment)
\tcolorboxenvironment{quote}{
  boxrule=0.4pt,
  colframe=black,
  colback=white,
  left=6pt,
  right=6pt,
  top=6pt,
  bottom=6pt,
  sharp corners
}
```

Notes:

* `boxrule=0.4pt` gives a thin, journal-safe rule.
* `sharp corners` keeps it conservative.
* White background avoids “styled callout” vibes.

---

## Step 4 — Ensure you *only* use blockquotes for Boxes (sanity check)

Search `paper.md` for any other uses of blockquotes (`^>`).
If there are blockquotes not meant to be boxed, convert those non-box blockquotes to something else (e.g., italics, a normal paragraph, or a bold “Note:” line).

Cursor instructions:

* If blockquotes are used only for Box 1/Box 2, do nothing.
* If there are other blockquotes, flag them and convert them so they won’t be boxed.

---

## Step 5 — Build the PDF and visually verify

Run your normal build (whatever you use: `make paper-full`, or your Pandoc command). Verify in the PDF:

* Box 1 and Box 2 have:

  * a thin black rectangular border
  * consistent padding
  * no indentation weirdness
  * no page overflow

* No unintended content is boxed.

---

## Step 6 — If margins look weird, apply this optional tweak

If the box is slightly indented relative to the main text, add `enhanced jigsaw` + set `before/after skip` explicitly:

```tex
\tcolorboxenvironment{quote}{
  enhanced jigsaw,
  boxrule=0.4pt,
  colframe=black,
  colback=white,
  left=6pt,
  right=6pt,
  top=6pt,
  bottom=6pt,
  sharp corners,
  before skip=10pt,
  after skip=10pt
}
```

Only do this if the default spacing looks cramped.

---

## Single Cursor instruction (copy/paste)

If you want one message to Cursor:

> Add boxed borders around Markdown blockquotes in the PDF by editing `header.tex`: include `\usepackage[most]{tcolorbox}` and override the LaTeX `quote` environment with `\tcolorboxenvironment{quote}{...}` (thin black border, white background, modest padding, sharp corners). Then scan `paper.md` for any non-box blockquotes and convert them to non-blockquote formatting so only the Box callouts are boxed. Rebuild and confirm Box 1/Box 2 render as bordered boxes with consistent spacing and no unintended boxed text.

---

If you tell me how you invoke Pandoc (or show the relevant `Makefile` snippet), I can tune the placement in `header.tex` to avoid any package-order conflicts—but the above works in most setups.

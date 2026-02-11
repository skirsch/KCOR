
# Cursor instruction: Insert quiet-window robustness scan into supplement.md 

Goal:
Add a new SI subsection + figure showing empirical robustness of frailty parameter estimates to quiet-window placement (Czech 2021_24; YoB 1940s/1950s; Dose 0/1/2). This rebuts “quiet windows are rare / fragile” and concretely instantiates the “quiet-window perturbation” diagnostic already listed in S2 and S5.

Files:
- Edit: supplement.md
- Add figure file: figures/fig_si_quiet_window_theta_scan_czech_2021_24.png  (or your preferred naming; just keep the reference consistent)

The figure generated can be found in test/quiet_window/out/fig_quiet_window_theta_scan_czech_2021_24.png

----------------------------------------------------------------------
STEP 1 — Add the figure file to repo
----------------------------------------------------------------------
1) Copy the final PNG you generated into:
   figures/fig_si_quiet_window_theta_scan_czech_2021_24.png

(If you prefer to keep SI figures in a subfolder, use:
   figures/supplement/fig_si_quiet_window_theta_scan_czech_2021_24.png
and adjust the path in the markdown below.)

----------------------------------------------------------------------
STEP 2 — Insert a new subsection in S5
----------------------------------------------------------------------
Insertion point:
In supplement.md, find the line:

### S5.4 Quiet-window overlay plots

Insert the following block IMMEDIATELY BEFORE that heading.

---------- BEGIN INSERT ----------

### S5.4 Quiet-window perturbation scan (empirical Czech data)

To assess whether diagnostically valid quiet windows are rare or fragile in applied registry data, we performed a systematic scan of quiet-window placements in the Czech Republic 2021–2024 cohort (enrollment ISO week 2021-24). We re-estimated gamma-frailty parameters $(\hat{k}_d,\hat{\theta}_d)$ across a sequence of overlapping quiet windows of fixed duration, holding all other settings fixed.

Specifically, we considered 12-month windows beginning at successive monthly offsets between April 2022 and April 2023 (inclusive). For each window placement, parameters were estimated independently using nonlinear least squares in cumulative-hazard space over bins within the window, subject to the same diagnostic pass/fail criteria used elsewhere in this SI. Results are shown for two birth-decade strata (1940s and 1950s), and for dose groups $d \in \{0,1,2\}$.

Figure @fig:si_quiet_window_theta_scan shows $\hat{\theta}_d$ as a function of quiet-window midpoint date. For Dose 0 (unvaccinated) cohorts, $\hat{\theta}_0$ is consistently non-zero and stable across many window placements, indicating that depletion curvature and the associated frailty parameters are not driven by a narrowly tuned quiet interval. For Dose 1 and Dose 2 cohorts, $\hat{\theta}_d$ collapses toward zero across placements (note the panel-specific y-axis scales), reflecting near-linear cumulative hazards and weak or absent depletion curvature rather than estimator instability. Open markers denote windows failing diagnostics (treated as non-identified and not interpreted).

![Quiet-window robustness scan: estimated frailty variance $\hat{\theta}_d$ vs. quiet-window midpoint date for Czech 2021–2024 enrollment cohort, using 12-month windows shifted monthly from April 2022 through April 2023. Marker shape denotes birth decade (1940s, 1950s). Filled markers indicate diagnostic pass; open markers indicate diagnostic fail. Note the changing y-axis scales across panels: Dose 1 and Dose 2 estimates collapse toward zero.](figures/fig_si_quiet_window_theta_scan_czech_2021_24.png){#fig:si_quiet_window_theta_scan}

---------- END INSERT ----------

----------------------------------------------------------------------
STEP 3 — Renumber the existing S5.4 section
----------------------------------------------------------------------
Because you inserted a new "### S5.4 ...", you must rename the existing section:

Current heading:
### S5.4 Quiet-window overlay plots

Change it to:
### S5.5 Quiet-window overlay plots

Then increment all subsequent S5 subsection numbers by +1:
- S5.5 Robustness to age stratification  -> S5.6 Robustness to age stratification
(and any later S5 subsections if they exist in your local version).

Make sure internal cross-references (if any) remain correct.

----------------------------------------------------------------------
STEP 4 — Optional: strengthen linkage to S2 diagnostic table (no reflow)
----------------------------------------------------------------------
No table edits required, but if you want a single sentence to connect this to the diagnostic framework:
In S2 Table @tbl:si_diagnostics row “Quiet-window perturbation”, optionally add:
“See Figure @fig:si_quiet_window_theta_scan for an empirical scan over monthly-shifted 12-month windows in Czech data.”

(This is optional; the S5 subsection already stands on its own.)

FURTHER clarification of placement....

Good question — and this is one of those places where placement really matters.

You should add that sentence **inside Section S2**, immediately after the diagnostic table — not in S5.

Let me walk you through it clearly.

---

## Where exactly to insert it

Open `supplement.md` and go to:

### **S2 Diagnostics and identifiability**

Find the diagnostic summary table (the one labeled something like):

```
Table S2: Diagnostic checks and interpretations
```

or the table that lists rows such as:

* Quiet-window perturbation
* Gamma-frailty adequacy
* Baseline stability
* etc.

---

## Insert the sentence **immediately after the table block**, before the next subsection header.

So structurally it should look like this:

```markdown
Table S2: Diagnostic checks and interpretations
| Diagnostic | What it tests | Failure meaning | ...
|------------|---------------|----------------| ...

Empirical illustration of the “Quiet-window perturbation” diagnostic is provided in Figure @fig:si_quiet_window_theta_scan, which scans monthly-shifted 12-month windows in Czech registry data.

### S2.3 (next subsection heading)
```

That’s it.

---

## Why this is the right place

* It connects the **conceptual diagnostic framework (S2)**
  to
* the **empirical demonstration (S5)**

Without:

* duplicating text,
* bloating S2,
* or breaking flow.

It reads as:

> “Here is the diagnostic framework.
> Here is a real-data example of one of those diagnostics.”

Very clean.

---

## Why not put it somewhere else?

* ❌ Not at the start of S2 — too early
* ❌ Not inside S5 — that’s already descriptive
* ❌ Not in the main paper — too detailed
* ❌ Not as a new paragraph far away from the table — weak linkage

Right under the table is ideal.

---

## Final wording (use this exact sentence)

I recommend this slightly tightened version:

Empirical illustration of the “Quiet-window perturbation” diagnostic is provided in Figure @fig:si_quiet_window_theta_scan, which scans monthly-shifted 12-month windows in Czech registry data.



---

# Cursor instruction: Add main-text pointer to quiet-window robustness scan (paper.md)

Goal:
Add a single sentence in the main paper pointing readers to the SI quiet-window perturbation scan figure that demonstrates robustness of (k, theta) to quiet-window placement (Czech 2021_24). Keep it short and non-argumentative.

Files:
- Edit: paper.md
- Assumes SI figure label: @fig:si_quiet_window_theta_scan (added in supplement.md)
  and the SI subsection title is “S5.4 Quiet-window perturbation scan (empirical Czech data)”.

----------------------------------------------------------------------
STEP 1 — Find the best insertion point
----------------------------------------------------------------------
In paper.md, locate the section where you discuss:
- identifiability / quiet windows / diagnostics, or
- the “quiet-window perturbation” diagnostic, or
- the assumptions/diagnostics summary (often near the Methods / Assumptions / Diagnostics section).

Search for one of these strings (pick whichever exists):
- "quiet window"
- "quiet-window"
- "diagnostic"
- "perturbation"
- "Assumption 3"
- "identifiability"

Choose the paragraph that introduces quiet windows as an identifiability condition OR where you list/describe the diagnostics.

----------------------------------------------------------------------
STEP 2 — Insert one sentence (exact text)
----------------------------------------------------------------------
Insert the following single sentence as the LAST sentence of that paragraph:

Empirical robustness of the fitted frailty parameters to quiet-window placement (12-month windows shifted monthly) is demonstrated in Supplementary Figure @fig:si_quiet_window_theta_scan using Czech registry data.

Do not add any additional text.

----------------------------------------------------------------------
STEP 3 — If crossrefs require “Figure Sx” wording
----------------------------------------------------------------------
If your main paper style uses “Figure Sx” instead of direct crossref labels, replace the sentence with:

Empirical robustness of the fitted frailty parameters to quiet-window placement (12-month windows shifted monthly) is demonstrated in Figure Sx (Supplementary Information).

(Keep the @fig:... reference in the SI caption; only the main-text wording changes.)

----------------------------------------------------------------------
STEP 4 — Build check
----------------------------------------------------------------------
Run your normal build (pandoc/quarto) and verify:
- the reference resolves (no “??”)
- the SI figure appears once and is numbered correctly
- no duplicate S5 numbering remains after the SI insertion file changes

END
```

If you tell me the exact section header in `paper.md` where you want this (e.g., “Diagnostics,” “Assumptions,” “Identifiability,” etc.), I can tighten this into a *line-anchored* instruction (“insert after sentence containing …”).

Yes — **the Assumptions section is exactly the right place**, and there’s a *very specific* spot within it that will work best.

Let me be precise.

---

## Best placement for the main text reference: **Assumption A3 (Quiet window / baseline stability)**

Wherever you currently introduce the quiet-window concept as an assumption (often phrased like “existence of a sufficiently long epidemiologically quiet period”), that’s the **ideal insertion point**.

Why?

* The SI figure **validates the assumption**, it doesn’t introduce a new result
* Reviewers naturally ask “is this assumption realistic?”
* Your figure answers that *immediately*, without argument

This keeps Results clean and avoids overclaiming.

---

## Exact placement within the Assumptions section

### Insert **after**:

* You define what a quiet window is
* You explain why it is needed for identifiability

### Insert **before**:

* Any transition to estimation, KCOR construction, or comparisons
* Any math-heavy exposition

In other words:

> **Definition → justification → empirical validation (pointer) → move on**

That flow is ideal.

---

## The exact sentence (final recommendation)

Use this exact wording — it’s neutral, strong, and non-defensive:

> *Empirical robustness of the fitted frailty parameters to quiet-window placement (12-month windows shifted monthly) is demonstrated in Supplementary Figure @fig:si_quiet_window_theta_scan using Czech registry data.*

Why this works:

* “empirical robustness” not “proof”
* no argumentative adjectives
* doesn’t claim universality
* cleanly defers details to the SI

---

## Why not put it elsewhere?

* **Methods section** → feels like over-explaining
* **Results section** → makes it seem like a new finding
* **Diagnostics table** → too compressed
* **Discussion** → too late; reviewers already formed an opinion

Assumptions is the *sweet spot*.

---

## Final guidance (do this)

1. Place the sentence in **Assumption A3 (Quiet window)**
2. Make it the **last sentence of that assumption**
3. Leave everything else unchanged

This is exactly how a careful statistician would signal:

> “Yes, we thought about this — and here’s where to look if you care.”

If you want, next I can:

* help you tweak the *wording of Assumption A3 itself* to align even more tightly with the diagnostic framing, or
* do a final “reviewer-eye” pass over the Assumptions section to see if anything else could trigger confusion.

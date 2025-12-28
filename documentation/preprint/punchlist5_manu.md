INSERTION 1 — Explicit positioning (Introduction)
Location

Section 1.5 “Contribution of this work”
Insert after the first paragraph, before the bullet list.

Exact wording to insert

KCOR is not merely a frailty‐normalization method.
While gamma-frailty inversion is a necessary step, the central contribution of KCOR is the end-to-end comparison system that follows normalization. KCOR transforms observed cumulative hazards into a depletion-neutralized space and then defines the correct comparison operator in that space—a cumulative hazard ratio—together with diagnostics that determine when such comparisons are interpretable. Normalization alone does not yield a signal; the signal emerges only through the KCOR comparison itself. In this sense, KCOR should be understood as a complete retrospective comparison framework rather than a preprocessing adjustment that can be substituted into standard estimators.

Why this works:

Explicitly rejects “just normalization”

States clearly that signal extraction is part of the method

Preempts “why not normalize then Cox?”

INSERTION 2 — Methodological clarification (Methods section)
Location

Section 2.6 “Normalization (depletion-neutralized cumulative hazards)”
Insert as the final paragraph of §2.6, immediately before §2.6.1.

Exact wording to insert

Normalization is necessary but not sufficient.
The depletion-neutralized cumulative hazard $\tilde H_{0,d}(t)$ is not itself the estimand of interest. Its role is to place cohorts into a common comparison space in which selection-induced depletion dynamics have been removed. The substantive comparison—and therefore the inferential signal—arises only when these normalized cumulative hazards are compared across cohorts via the KCOR estimator (§2.8). Applying standard proportional-hazards or regression-based estimators after normalization is generally inappropriate, because the comparison is cumulative by construction and because residual non-proportionality is precisely what KCOR is designed to reveal. KCOR therefore integrates normalization and comparison into a single, internally consistent system.

Why this works:

Clearly says “normalization ≠ result”

Explicitly blocks “normalize then Cox”

Frames KCOR as inseparable stages

INSERTION 3 — Scope + strength (Discussion)
Location

Section 4.1 “What KCOR estimates”
Insert as the opening paragraph of the section.

Exact wording to insert

KCOR operates at a specific but critical layer of the retrospective inference stack: it both neutralizes selection-induced depletion dynamics and defines how the resulting depletion-neutralized hazards must be compared. The method’s strength is not the frailty inversion in isolation, but the fact that inversion, diagnostics, and cumulative comparison are mathematically and operationally coupled. Once cohorts are mapped into depletion-neutralized hazard space, KCOR$(t)$ directly answers whether one cohort experienced higher or lower cumulative event risk than another over follow-up, conditional on the stated assumptions. Interpreting normalized hazards without this comparison step discards the central inferential content of the method.

Why this works:

Explicitly frames KCOR as a layered system

Reasserts that it does answer “higher or lower risk”

Defuses the “just a diagnostic” framing without overclaiming causality

OPTIONAL (but strong): one-sentence guardrail against misinterpretation
Location

End of Section 4.3 “Practical guidance for use”

Sentence

KCOR should therefore be applied and reported as a complete pipeline—from cohort freezing, through depletion normalization, to cumulative comparison and diagnostics—rather than as a standalone adjustment step.

This sentence quietly but firmly enforces correct usage.

Why this fully addresses Manu without conceding ground

You accept his concern about scope

You reject the implication that KCOR is “just preprocessing”

You explicitly state that the comparison operator is part of the invention

Reviewers cannot say “this is only a normalization” without contradicting the paper’s own language

Importantly:
You do not say “KCOR is not an effect estimator.”
You say: KCOR defines how effects must be extracted under depletion.

That’s exactly right — and defensible.


## The comparison table
See the punchlist5_positioning_table.md for the table to insert

### Where to put it in the paper (very specific)

Insert as a new table in:

Section 1.5 – Contribution of this work
Immediately after the first paragraph, before the bullet list

This is exactly where reviewers decide:

“Is this just another adjustment method, or is it a real system?”

Placing it here ensures:

The reader understands KCOR’s role before seeing equations

You preempt the “just normalization” misread

The Methods section is interpreted correctly from the start

Why this table solves the Manu / reviewer issue cleanly

It accepts the need for positioning without conceding capability

It explicitly states that:

Normalization alone is not the signal

The KCOR ratio is the estimand

It contrasts KCOR with Cox, MSM, negative controls without attacking them

It frames KCOR as the missing layer in the stack, not a competitor doing the same job

This aligns perfectly with your “apples → apples → compare apples” explanation, but in methods-journal language.

Next optional upgrades (only if you want)

Add a one-panel schematic visually showing
Observed H(t) → Normalized Ĥ(t) → KCOR(t) = signal

Add a single sentence cross-reference in §2.8 pointing back to the table

Slightly tighten the caption to meet a specific journal’s word limit

But as-is, this file is ready to drop in and materially strengthens the paper.

### the "one-panel schematic" explained

Short answer: **it can be either — but you do *not* need a new figure.**
Given where you are, I recommend **re-using an existing figure with a revised caption**, not adding a new one.

Here’s the clean decision logic.

---

## Option A (RECOMMENDED): Reframe an existing figure with a stronger caption

If you already have a figure that shows something like:

> observed hazards → normalized hazards → comparison

then **do NOT add a new schematic**.

Instead:

* Keep the figure number the same
* Replace or augment the caption to explicitly state:

  * normalization is *not* the endpoint
  * the **KCOR ratio is the signal**

### Example caption upgrade (drop-in)

> **Figure X. KCOR as an integrated comparison system.**
> Observed cohort cumulative hazards are first mapped into depletion-neutralized hazard space via gamma-frailty inversion. **This normalization step alone does not constitute inference.** The KCOR estimator then compares normalized cumulative hazards via a ratio, which is the object that answers whether one cohort experienced higher or lower cumulative event risk than another under the stated assumptions. Diagnostics shown alongside the curves indicate when depletion-neutralization is valid.

This alone solves the “just normalization” risk.

---

## Option B: Add a *small* new schematic ONLY if you lack such a figure

Add a new figure **only if**:

* You currently have *no* figure that visually links normalization → comparison
* Reviewers would otherwise have to infer that flow from equations

If you do add one:

* It should be **one panel**, not multi-panel
* It should be conceptual, not data-driven
* It should live in the **Introduction**, not Methods

### Minimal schematic concept (textual)

```
Observed H_d(t)
      ↓
Frailty normalization
      ↓
Ĥ_d(t)  (depletion-neutralized)
      ↓
KCOR(t) = Ĥ_A(t) / Ĥ_B(t)
      ↓
Interpretable cumulative comparison
```

Caption would mirror the table you just added.

---

## What NOT to do

* ❌ Do **not** add a schematic that re-explains gamma frailty math
* ❌ Do **not** add a figure that could be interpreted as “normalization is the output”
* ❌ Do **not** introduce a new figure late unless it clarifies scope (this one would)

---

## My strong recommendation for *this* paper

Given your current manuscript already includes workflow figures:

**Do not add a new figure.**
**Reframe an existing workflow figure with a caption that explicitly says:**

> *Normalization is necessary but not sufficient; KCOR is the comparison.*

That keeps scope tight, avoids reviewer fatigue, and reinforces the table you just added.

If you want, paste the figure filenames you currently have (e.g., `fig_kcor_workflow.png`, etc.), and I’ll tell you **exactly which one to use and give you the exact replacement caption**.


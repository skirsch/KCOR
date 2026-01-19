

# Cursor Punchlist (Math Style–Corrected: `$...$ only`)

## Goal

Reduce reader confusion from undefined symbols **without simplifying math or changing estimands**, by fixing *symbol introduction order* and adding a *notation preview*.

Audience: technically strong **non-statisticians** (epi, clinical, applied).

Constraint: **use `$...$ math only` everywhere**.

---

## PART A — Add a “Notation preview” paragraph (highest impact)

### Location

Insert **immediately before the first displayed equation** in the manuscript
(early in §2.1, before any math blocks).

---

### Action A1 — Insert this paragraph verbatim

```markdown
**Notation preview.**  
Throughout this paper, $t$ denotes event time since cohort enrollment and $d$ indexes cohorts. Let $H_{\mathrm{obs},d}(t)$ denote the observed cohort-level cumulative hazard, computed from fixed risk sets, and let $\tilde H_{0,d}(t)$ denote the corresponding *depletion-neutralized baseline cumulative hazard* obtained after frailty normalization. Individual hazards are modeled using a latent multiplicative frailty term $z$, with cohort-specific variance $\theta_d$, which governs the strength of selection-induced depletion and resulting curvature in observed cumulative hazards. Full notation is summarized in Table @tbl:notation.
```

**Do not** add equations here — prose only.

---

## PART B — Fix first appearances of major symbols (surgical reorder)

You only need to do this for the **first appearance** of each symbol.

### Symbols to target

* $H_{\mathrm{obs},d}(t)$
* $\tilde H_{0,d}(t)$
* $\theta_d$
* $k_d$
* $z_{i,d}$

---

### Action B1 — Gamma-frailty identity (definition before equation)

#### BEFORE (problematic pattern)

```markdown
$$
H_{\mathrm{obs},d}(t) = \frac{1}{\theta_d}\log\!\left(1 + \theta_d \tilde H_{0,d}(t)\right).
$$
```

#### AFTER (replace with this structure)

```markdown
Let $\tilde H_{0,d}(t)$ denote the depletion-neutralized baseline cumulative hazard for cohort $d$, and let $\theta_d$ denote the cohort-specific frailty variance governing selection-induced depletion. Under a gamma-frailty working model, the observed cohort-level cumulative hazard satisfies:
$$
H_{\mathrm{obs},d}(t) = \frac{1}{\theta_d}\log\!\left(1 + \theta_d \tilde H_{0,d}(t)\right).
$$
```

**Rule:**

> Every symbol must be defined in prose *before* it appears inside a displayed equation.

---

### Action B2 — Gamma-frailty inversion (same rule)

#### AFTER pattern

```markdown
Given an estimate $\hat{\theta}_d$, the observed cumulative hazard can be mapped into depletion-neutralized baseline cumulative hazard space via exact inversion:
$$
\tilde H_{0,d}(t) = \frac{\exp\!\left(\hat{\theta}_d H_{\mathrm{obs},d}(t)\right) - 1}{\hat{\theta}_d}.
$$
```

---

### Action B3 — Individual-level hazard definition

#### BEFORE

```markdown
$$
h_{i,d}(t) = z_{i,d}\,\tilde h_{0,d}(t).
$$
```

#### AFTER

```markdown
Let $z_{i,d}$ denote an individual-specific latent frailty term with mean $1$ and variance $\theta_d$, and let $\tilde h_{0,d}(t)$ denote the depletion-neutralized baseline hazard for cohort $d$. Individual hazards are modeled as:
$$
h_{i,d}(t) = z_{i,d}\,\tilde h_{0,d}(t).
$$
```

---

## PART C — Optional navigation hint (safe, non-math)

### Location

At the **end of the Notation Preview paragraph** (Part A)

```markdown
Readers primarily interested in conceptual structure may wish to read Box 2 (§1.2) before returning to the formal development.
```

Optional but helpful for non-stat reviewers.

---

## PART D — Explicit non-goals (tell Cursor this)

Cursor **must not**:

* change equations
* change notation
* add or remove assumptions
* move the notation table
* simplify or explain math beyond reordering definitions
* introduce `\(...\)` anywhere

---

## PART E — Single Cursor instruction (copy/paste)

> **Cursor instruction:**
> Add a short “Notation preview” paragraph before the first displayed equation defining $t$, $d$, $H_{\mathrm{obs},d}(t)$, $\tilde H_{0,d}(t)$, $z$, and $\theta_d$. Then, for the *first appearance only* of each major symbol, reorder text so that symbols are defined in prose *before* they appear in displayed equations. Use `$...$` math exclusively. Do not change equations, estimands, or notation.

---

## Expected outcome

* Non-stats reviewers stop getting lost
* Stats reviewers see no dilution
* Zero math-style regressions
* You neutralize the “symbols not defined” critique cleanly

If you want next:

* I can generate a **diff-style patch** with exact insert points, or
* annotate **exact section numbers** where Cursor should act.

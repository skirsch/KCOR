# KCOR Markdown Math Style Guide (Cursor + Pandoc Compatible)

This document defines the **canonical Markdown math style** for KCOR documents so that math renders **correctly in Cursor (KaTeX)** *and* **round‑trips cleanly through Pandoc to Word**.

This guide supersedes all previous KCOR math style guidance.

---

## 1. The One Rule That Matters

✅ **Use dollar signs for all math:**

- **Inline math:** `$ ... $`
- **Display math:** `$$ ... $$`

❌ **Do NOT use:**

- `\( ... \)` (breaks inline math in Cursor)
- `\[ ... \]` (breaks in Cursor and VS Code)
- LaTeX math inside code blocks

If you follow this rule, your math will work everywhere KCOR needs to work.

---

## 2. Inline Math (Required Standard)

Use `$ ... $` for all inline symbols and expressions.

### Example (correct)

```markdown
The cumulative hazard is $H^{obs}(t)$ and the frailty variance is $\theta$.
```

### Why this works

- Cursor / KaTeX: ✅
- Pandoc → Word: ✅
- Pandoc → PDF / LaTeX: ✅
- GitHub Markdown: ✅

### Do NOT use

```markdown
The cumulative hazard is \(H^{obs}(t)\).   ❌
```

This will show a literal `(` in Cursor.

---

## 3. Display (Block) Math

Use `$$ ... $$` for all display equations.

Make sure to start in column 1 for the `$$` and start and end should be 
the only thing on the line.

### Example

```markdown
$$
H^{obs}(t) = \frac{1}{\theta}\log\!\left(1 + \theta H_0(t)\right).
$$
```

This renders correctly in:
- Cursor
- VS Code
- Pandoc → Word
- Pandoc → PDF

---

## 4. Multi‑line Equations

Use a single `$$ ... $$` block.

### Example

```markdown
$$
h(t) = \frac{k}{1 + \theta k t},
\qquad
H_0(t) = \frac{e^{\theta H^{obs}(t)} - 1}{\theta}.
$$
```

Avoid splitting a single equation across multiple math blocks.

---

## 5. Canonical KCOR Hazard Definition

For piecewise‑constant hazards, **always** use this exact pattern:

```markdown
Let $s$ denote event time since enrollment, $D(s)$ deaths during interval $s$,
and $N(s)$ the number at risk at the start of $s$.
Hazards are treated as piecewise‑constant and computed as

$$
h(s) = -\ln\left(1 - \frac{D(s)}{N(s)}\right).
$$
```

This formulation is:
- mathematically correct,
- numerically stable,
- and renders cleanly in both Cursor and Word.

---

## 6. Code Blocks and Math

🚫 **Never put LaTeX math inside code blocks.**

Code blocks are rendered literally; math will not render.

### Correct pattern

```markdown
We fit the cumulative‑hazard model

$$
H^{obs}(t) = \frac{1}{\theta}\log\!\left(1 + \theta k t\right).
$$

```python
# Python implementation here
```
```

### Incorrect pattern

```python
# H^{obs}(t) = (1/theta) * log(1 + theta*k*t)   ❌
```

---

## 7. Greek Letters and Functions

Always use LaTeX commands *inside math mode*:

- `$\theta$`, `$\gamma$`, `$\lambda$`
- `$\log$`, `$\ln$`, `$\exp$`
- `$\sum$`, `$\int$`, `$\frac{a}{b}$`

Example:

```markdown
$$
H_0(t) = \frac{e^{\theta H^{obs}(t)} - 1}{\theta}.
$$
```

---

## 8. Copy‑Paste Prompt for Future Requests

When asking ChatGPT or Cursor to generate KCOR Markdown, include **this exact sentence**:

> **“Format math for Cursor (KaTeX) *and* Pandoc → Word: use `$ … $` for inline math and `$$ … $$` for display math. Do not use `\( … \)` or `\[ … \]`. Do not put LaTeX inside code blocks.”**

This prevents rendering problems before they start.

---

## 9. Summary (TL;DR)

**Always do this**
- Inline math → `$ … $`
- Display math → `$$ … $$`

**Never do this**
- `\( … \)`
- `\[ … \]`
- Math inside code blocks

This is now the **official KCOR Markdown math standard**.

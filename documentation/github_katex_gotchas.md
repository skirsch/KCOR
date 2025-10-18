# KaTeX on GitHub: Rules & Gotchas


most important rule is that `$$` to start and end math must ALWAYS start in column 1 and be on their own lines.

These lines must have a blank line before the `$$` and after the closing one.

The second most important rule is that using \text in a subscript or superscript will not work.

Make sure open and close braces match up.

Using `\%` will not work. You'd need to use two backslashes `\\%` instead to get a % sign to appear in the text.

For all other characters, backslash works, but not %. 


`\!` will NOT create a thin space but print as is. So do not use this.


## Math delimiters GitHub accepts
- **Inline:** `$ … $` or `\(...\)`
- **Display:** `$$ … $$` or `\[…\]`
- Prefer `\(...\)` / `\[…\]` if your text also uses `$` (currency, shell vars); stray `$` will break rendering.

---

## `\text{…}` pitfalls (common reason rendering fails)
Inside `\text{…}`, you’re in **text mode**, but TeX special characters still matter.  
If you include any of these unescaped, KaTeX will fail and GitHub will print the literal source:

| Character | Escape it as |
|------------|--------------|
| `_` | `\_` |
| `^` | `\^{}` |
| `%` | `\%` |
| `#` | `\#` |
| `&` | `\&` |
| `{` or `}` | must be **balanced** |

Also, every backslash must form a valid macro — a stray `\` breaks it.

**✅ Example (good):**  
```
$\text{file\_name \% complete \#tasks \& notes}$
```

If you don’t need proportional text, you can use:
- `\mathrm{…}` — roman upright  
- `\operatorname{…}` — spaced like functions  
- `\texttt{…}` — monospace  

…but you must still escape `_`, `%`, etc.

---

## Markdown vs TeX escaping
- In normal Markdown, a single backslash is for TeX; **don’t double-escape** unless Markdown itself would eat it (like in tables).
- In **code fences** (\`\`\`), math won’t render — GitHub treats it as literal code.

---

## Supported environments
✅ **Works:**  
`aligned`, `cases`, `pmatrix`, `bmatrix`, `matrix`, `split`, `alignedat`, `array`, `gather`.

🚫 **Avoid:**  
`align` (as a standalone block), `eqnarray`.

Use this instead:
```latex
$$
\begin{aligned}
y &= mx + b \\
  &= m(x_0 + \Delta x) + b
\end{aligned}
$$
```

---

## Lists, tables, and spacing
- Display math inside lists/tables can be finicky.  
  - Leave a **blank line** before and after math blocks.  
  - Indent the math block to match the list level.
- Regular spaces are ignored — use `\,`, `\;`, or `\quad` for spacing.

---

## Common causes of “printing the source” instead of rendering
1. Unbalanced braces anywhere (especially inside `\text{…}`).
2. Unescaped `_`, `%`, `#`, `&` inside `\text{…}`.
3. Stray `$` in surrounding Markdown capturing too much or too little text.
4. Unsupported environments (e.g., bare `align`).

---

## Safe reusable patterns

**Inline with text:**
```latex
$\mathrm{AUC}$ of the model is $0.91$
```

**Inline freeform text:**
```latex
$\text{train\_run\%03d started}$
```

**Named operator:**
```latex
$\operatorname{argmin}_{x\in\mathbb{R}^n} f(x)$
```

**Piecewise:**
```latex
$$
f(x)=
\begin{cases}
x^2, & x \ge 0 \\
-x,  & x < 0
\end{cases}
$$
```

**Aligned equations:**
```latex
$$
\begin{aligned}
\log L(\theta)
&= \sum_{i=1}^n \log p(x_i\mid\theta) \\
&= \textstyle\sum_{i=1}^n \left( -\tfrac12 \log(2\pi\sigma^2)
     - \tfrac{(x_i-\mu)^2}{2\sigma^2} \right)
\end{aligned}
$$
```

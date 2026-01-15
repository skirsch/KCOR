Here’s a **Cursor punchlist** to (1) fix italicization / math-mode inconsistencies like `𝐻obs,d` or `Hobs,d`, and (2) enforce **tilde notation** consistently for depletion-neutralized quantities across **paper.md** and **supplement.md**.

---

## Cursor punchlist: notation consistency (paper + supplement)

### 0) Safety rule (do this first)

* Don’t do any blind “Replace All” on ambiguous tokens like `H0` or `H`.
* Only use global replaces for **Unicode math letters** and **specific literal strings** that are clearly wrong (e.g., `Hobs,d`).

---

## Part A — Remove Unicode math italics and force math mode

### A1) Replace Unicode math letters with ASCII equivalents (safe global replace)

In **both** `paper.md` and `supplement.md`, do a **global search** for Unicode math characters (common culprits):

Search (one at a time):

* `𝐻` `ℎ` `𝑯` `𝑯` `𝑑` `𝑡` `𝑧` `𝜃` `𝛉` `𝜇` `𝜎` `𝑁` `𝑘`
* also look for “fancy” subscripts like `₀`, `₁`, etc.

For each hit:

* Replace the Unicode character with the plain ASCII letter:

  * `𝐻` → `H`
  * `𝑡` → `t`
  * `𝑑` → `d`
  * `𝜃`/`𝛉` → `\theta` (only if already inside math; otherwise replace with `theta` temporarily and then fix properly)

**Important:** After converting Unicode to ASCII, **wrap the expression in math mode** if it’s a variable, not a label.

Example:

* `𝐻obs,𝑑` → `Hobs,d` → then fix as `\(H_{\mathrm{obs},d}\)` (see A2).

### A2) Fix “inline variable” strings that are not in math mode

Use Cursor multi-file search for these plain-text patterns (they usually appear in prose/captions):

Search:

* `Hobs,d`
* `H_obs`
* `Hobs`
* `hobs`
* `MR_{d,t}` (if it appears outside `$...$`)
* `theta` (as plain text, only if you mean (\theta))

For each occurrence, convert to math mode with the standardized form (next section).

---

## Part B — Standardize the “observed hazard / cumulative hazard” notation

Pick one canonical notation and enforce it everywhere:

**Canonical forms (recommended):**

* Observed discrete-time hazard: `\(h_{\mathrm{obs},d}(t)\)`
* Observed cumulative hazard: `\(H_{\mathrm{obs},d}(t)\)`

### B1) Replace common variants with canonical notation

In both files, search and replace **case-by-case**:

1. Replace `Hobs,d` (or `Hobs, d`) in text with:

* `\(H_{\mathrm{obs},d}\)` if no `(t)` is present
* `\(H_{\mathrm{obs},d}(t)\)` if time is implied/nearby

2. Replace `hobs,d` / `h_obs,d` similarly with:

* `\(h_{\mathrm{obs},d}(t)\)`

3. Replace plain `H_obs,d(t)` (if you see it) with:

* `\(H_{\mathrm{obs},d}(t)\)` (keep consistent subscript formatting)

### B2) Make sure roman subscripts are roman

Whenever you have `obs`, `eff`, `base`, etc., ensure they are **roman**:

* `H_{obs}` → `H_{\mathrm{obs}}`
* `h_{eff}` → `h_{\mathrm{eff}}`

(Only do this inside math.)

---

## Part C — Unify tilde notation for depletion-neutralized quantities

### C0) Define the rule (enforce everywhere)

* **Observed** quantities: no tilde (e.g., `\(H_{\mathrm{obs},d}(t)\)`).
* **Depletion-neutralized / baseline** quantities: **always tilde** (e.g., `\(\tilde H_{0,d}(t)\)`).
* **Never** refer to a depletion-neutralized quantity as `H_{0,d}(t)` without tilde after you’ve introduced the normalization.

### C1) Create/confirm one explicit “notation sentence” (optional but strong)

If you don’t already have it in Methods, add (once) near where (\tilde H) is first introduced:

```markdown
We use a tilde (e.g., $\tilde H_{0,d}(t)$) to denote depletion-neutralized baseline quantities obtained after frailty normalization; observed cohort-aggregated quantities are written without a tilde (e.g., $H_{\mathrm{obs},d}(t)$).
```

### C2) Scan for likely “tilde leaks”

In both files, multi-file search for these patterns:

Search:

* `H_{0,`
* `h_{0,`
* `H_0`
* `baseline cumulative hazard` (in prose)
* `baseline hazard` (in prose)
* `depletion-neutralized` (to ensure it’s paired with tilde notation nearby)

For each hit, decide which of these you mean:

**If you mean depletion-neutralized baseline:**

* replace `\(H_{0,d}(t)\)` → `\(\tilde H_{0,d}(t)\)`
* replace `\(h_{0,d}(t)\)` → `\(\tilde h_{0,d}(t)\)` (only if you actually define/use (\tilde h); if not, prefer cumulative hazard language)

**If you mean the *underlying* baseline hazard in the generative model (pre-normalization):**

* keep `\(h_{0,d}(t)\)` / `\(H_{0,d}(t)\)` **without tilde**, but add a clarifying phrase like “data-generating baseline” if needed.

  * This is mainly in simulation DGP sections.

### C3) Caption + Discussion sweep (where leaks usually occur)

Do a focused search in captions and discussion sections for:

* `baseline hazard`
* `baseline cumulative hazard`
* `normalized hazard`
* `depletion-neutralized`

Make sure any mention of normalized baseline hazards uses the tilde form.

---

## Part D — Quick consistency checks (fast + reliable)

### D1) Ensure no Unicode math letters remain

Search in both files for any of these characters again:

* `𝐻`, `𝑡`, `𝑑`, `𝜃`, `𝑧`, `₀`, `₁`
  If found: convert + wrap in math mode.

### D2) Ensure no plain-text “Hobs” remains

Search:

* `Hobs`
* `hobs`
  All remaining instances should be inside `$...$` and in canonical form.

### D3) Ensure tilde is used consistently

Search:

* `\tilde{H}` and `\tilde H`
  Make sure your document uses one style consistently (I recommend `\tilde H`, not braces, unless ambiguity requires it).

Search:

* `H_{0,` and `H_0`
  Manually confirm each occurrence is either:
* DGP baseline (no tilde), or
* depletion-neutralized baseline (tilde)

---

## Minimal “done” criteria

You’re done when:

1. No Unicode math letters remain.
2. All observed hazards/cum hazards appear as:

   * `\(h_{\mathrm{obs},d}(t)\)` and `\(H_{\mathrm{obs},d}(t)\)`
3. All depletion-neutralized baseline quantities use tilde:

   * `\(\tilde H_{0,d}(t)\)`
4. No captions/prose refer to neutralized quantities without tilde.

---

If you want, paste 10–20 lines around one example where you suspect a “tilde leak” (often in Discussion or captions), and I’ll tell you exactly which side it belongs on (DGP baseline vs depletion-neutralized).

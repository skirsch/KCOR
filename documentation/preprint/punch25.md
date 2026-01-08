Yep — option A is clean: delete the “improved midpoint” hazard transform entirely (Eq. 11 / `@eq:hazard-from-mr-improved`), switch the implementation to the standard exact discrete-time integrated hazard `ΔH = -log(1 − MR)`, and update the manuscript to reflect **calendar year 2023** as the quiet period.

Below are Cursor-ready instructions for **KCOR.py** and **paper.md**.

---

## Cursor instructions: KCOR.py

### Goal

* Remove the special “midpoint improved” hazard approximation.
* Compute weekly hazard increments as:

  * `mr = D / Y`
  * `dh = -log(1 - mr)` (use `np.log1p(-mr)` for stability)
* Update quiet window defaults to **ISO weeks 2023-W01 … 2023-W52** (calendar year 2023).

### Steps (do these in Cursor)

1. **Find where the hazard increment is computed from MR**

* Use search (ripgrep):

  * `hazard-from-mr`
  * `hazard_from_mr`
  * `mr` near `log`
  * `1.5`
  * `0.5`
  * `midpoint`

2. **Replace the transform with exact discrete hazard**
   Wherever you currently do something like the “improved” transform (anything involving `1.5*mr` and `0.5*mr`), replace with:

```python
# Weekly mortality risk
mr = deaths / at_risk

# Guardrails: avoid log(0) and negative risk from numerical noise
mr = np.clip(mr, 0.0, 1.0 - 1e-12)

# Exact discrete-time integrated hazard under piecewise-constant hazard in the interval
dh = -np.log1p(-mr)   # == -log(1 - mr)

# Then accumulate
H_obs = np.cumsum(dh)
```

Notes you should preserve in code comments:

* This is the exact integrated hazard for an interval when the hazard is constant within the interval.
* `np.log1p` improves numerical accuracy for small `mr`.

3. **Delete dead code / config flags for the old transform**

* Remove helper functions like `hazard_from_mr_improved(...)` or similarly named.
* Remove any constants or docs referencing “second-order midpoint correction”.

4. **Update the quiet window defaults**
   Find wherever the quiet window is defined (likely as ISO week bounds). Set defaults to calendar year 2023, e.g.:

* quiet_start = `2023-01`
* quiet_end   = `2023-52`

(Use whatever internal representation your code uses: ISO week strings, integers, dates — but the semantic window should be **the full CY2023**.)

5. **Run/verify**

* Run your unit tests / figures build (whatever your repo uses) and ensure no references remain to the removed transform.

---

## Cursor instructions: paper.md (the manuscript)

You said the manuscript lives at `/mnt/data/paper.md`. Here are exact edits to make there.

### 1) Update the “Reference implementation defaults” block (Section 2.5)

In `/mnt/data/paper.md`, locate this block (around the “Reference implementation defaults” section):

* Current quiet window bullet:

  * `ISO weeks 2022-24 through 2024-16`
* Current hazard transform equation block labeled:

  * `{#eq:hazard-from-mr-improved}`
  * and the paragraph that claims it reduces to Nelson–Aalen etc.

**Replace that entire hazard-transform bullet + displayed equation + the 3 explanatory lines immediately following it** with this (no numbered equation):

* **Quiet window**: ISO weeks `2023-01` through `2023-52` (inclusive; calendar year 2023).
* **Observed cumulative hazards**: computed from weekly mortality risk $\mathrm{MR}*{d,t}=D*{d,t}/Y_{d,t}$ using the exact discrete-time integrated hazard increment $\Delta H_{d}(t) = -\log!\left(1-\mathrm{MR}_{d,t}\right)$, then accumulated over weeks.

Also keep (or adjust) your existing sentence about Nelson–Aalen sensitivity, but now it should read as a cross-check, not as the “primary” transform.

### 2) Remove Appendix A.4 (it is now obsolete)

In Appendix A, delete the entire section:

* `#### A.4 Derivation and properties of Equation @eq:hazard-from-mr-improved`

…through the end of that section (ending right before `#### A.5 Variance propagation (sketch)`).

Keep the mortality rate definition if you still want it, but you don’t need it as its own numbered sub-derivation anymore. If you keep anything, keep only the simple definitions:

* $\mathrm{MR}_{d,t} = d_d(t)/N_d(t)$
* $\Delta H(t) = -\log(1-\mathrm{MR}_{d,t})$

…but since you want to “totally eliminate equation (11)”, simplest is: **delete A.4 entirely** and rely on the short statement in the main-methods section.

### 3) Update the sensitivity-analysis quiet-window parameters (Appendix B.4)

In `#### B.4 Sensitivity analysis parameters`, update these bullets:

* Change:

  * “Quiet-start offsets … weeks from 2022-24”
  * “Quiet-window end: Fixed at 2024-16”

To:

* **Quiet-start offsets**: [-12, -8, -4, 0, +4, +8, +12] weeks from `2023-01`
* **Quiet-window end**: Fixed at `2023-52`

### 4) Remove any remaining references to `@eq:hazard-from-mr-improved`

Do a global search in `paper.md` for:

* `hazard-from-mr-improved`

Delete or rewrite any remaining sentences that mention it.

---

## Tiny “review-proofing” wording tweak (optional but helpful)

In the methods where you describe computing observed cumulative hazards, add one short line like:

> We use the exact discrete-time integrated hazard increment at weekly resolution; no higher-order midpoint correction is required for the KCOR estimand, and Nelson–Aalen increments yield indistinguishable results in our validations.

That makes it very clear you intentionally removed the controversial/bug-prone optimization.

---

If you paste the relevant KCOR.py snippet (the function that converts weekly deaths/alive into `H_obs`), I can give you an exact patch-style replacement for that specific block using your variable names and shapes.

so

- deaths=deaths that week and at_risk is at risk at the START of the week?

Yes — **that’s exactly the right interpretation**, and it’s the one under which **Option A is mathematically exact**.

### Definitions (what your code is already doing)

* **`deaths`** = number of deaths **during** week *t*
* **`at_risk`** = number alive **at the start of** week *t*

So your weekly mortality risk is
[
\mathrm{MR}_t = \frac{D_t}{Y_t},
]
with (Y_t) measured at **week start**.

### Why `ΔH = -log(1 − MR)` is correct here

If the hazard is assumed **piecewise constant within the week**, then:

* Survival over the week:
  [
  S(t+\Delta) = \exp(-h \Delta)
  ]
* Observed weekly event probability:
  [
  \mathrm{MR}_t = 1 - \exp(-h\Delta)
  ]
* Solve for the integrated hazard:
  [
  \Delta H_t = h\Delta = -\log(1 - \mathrm{MR}_t)
  ]

So with your definitions:

```python
mr = deaths / at_risk
dh = -np.log1p(-mr)
```

is **exact**, not an approximation.

### Why this is strictly better than the old Eq. (11)

* No midpoint/exposure correction needed.
* No Taylor expansion arguments.
* No “reduces to Nelson–Aalen” claims to defend.
* Works for *any* weekly mortality level (important in old-age cohorts).
* Matches standard discrete-time survival theory.

For small MR:
[
-\log(1-\mathrm{MR}) = \mathrm{MR} + \tfrac{1}{2}\mathrm{MR}^2 + O(\mathrm{MR}^3),
]
so it automatically behaves like Nelson–Aalen when deaths are rare.

### One guardrail to keep in the code

Just make sure you clip MR to avoid numerical issues:

```python
mr = np.clip(deaths / at_risk, 0.0, 1.0 - 1e-12)
dh = -np.log1p(-mr)
```

### Bottom line

Given:

* deaths counted **within** the week
* at_risk measured **at week start**

👉 **Option A is the correct and defensible choice**
👉 Removing Eq. (11) strengthens the paper, not weakens it

You’re now aligned with textbook discrete-time survival analysis, and reviewers will have nothing to grab onto here.

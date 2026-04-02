Perfect — this is a *great* next experiment and will make the point crystal clear.

You want to show:

> Big wave ≠ identifiable alpha
> Differential scaling (alpha + theta) = identifiable alpha

Here is a **clean Cursor instruction** to implement that.

---

# Cursor Task: Wave-size vs alpha-identifiability experiment

## Goal

Create a synthetic experiment that isolates the effect of **wave magnitude vs differential scaling** on alpha identifiability.

Answer:

> Does a large wave (e.g., 2x mortality) guarantee alpha identifiability, or is cross-cohort differential scaling the true driver?

---

## Scope

Work only in:

```
test/alpha/
```

Do NOT modify:

```
code/KCOR.py
```

---

# Step 1 — Add wave magnitude axis

In:

```
test/alpha/params_alpha.yaml
```

Add:

```yaml
synthetic_wave_multipliers:
  - 1.25
  - 1.5
  - 2.0
```

Interpretation:

* multiplies the epidemic forcing (A(t))
* affects all cohorts equally at baseline

---

# Step 2 — Add theta (frailty strength) axis

Also add:

```yaml
synthetic_theta_multipliers:
  - 0.5
  - 1.0
  - 2.0
```

Interpretation:

* scales the frailty variance
* controls how strongly cohorts diverge

---

# Step 3 — Modify synthetic generator

In:

```
test/alpha/code/estimate_alpha.py
```

Update synthetic generation to:

### Apply wave multiplier

```python
A_t = base_A_t * wave_multiplier
```

### Apply theta multiplier

```python
theta_effective = theta_base * theta_multiplier
```

Use `theta_effective` everywhere the frailty moment is computed.

---

# Step 4 — Disable VE for this experiment

Set:

```yaml
synthetic_vaccine_effect:
  enabled: false
```

We want **pure alpha identifiability**, no VE confounding.

---

# Step 5 — Run alpha estimator unchanged

Do NOT modify:

* pairwise objective
* collapse objective
* bootstrap
* LOO

This is a pure identifiability test.

---

# Step 6 — Outputs

Create:

```
alpha_wave_identifiability.csv
```

Columns:

* alpha_true
* wave_multiplier
* theta_multiplier
* alpha_hat_raw
* identified (bool)
* curvature
* bootstrap_boundary_fraction

---

# Step 7 — Key figure

Create:

```
fig_wave_vs_identifiability.png
```

## Panel A — Effect of wave size

Fix:

* theta_multiplier = 1.0

Plot:

* x = wave_multiplier
* y = identification rate
* separate lines for different alpha_true

## Panel B — Effect of theta

Fix:

* wave_multiplier = 2.0

Plot:

* x = theta_multiplier
* y = identification rate

---

# Step 8 — Add interpretation logic

In report:

```
alpha_wave_identifiability_report.md
```

Include:

```text
Key test:
Does increasing wave magnitude alone improve identifiability?

Key test:
Does increasing theta (frailty variance) improve identifiability?

Conclusion:
If wave increases without improved identification, but theta increases do improve identification, then alpha is driven by cross-cohort differential scaling, not absolute wave size.
```

---

# Step 9 — Console summary

Print:

```text
WAVE IDENTIFIABILITY TEST

wave=1.25 theta=1.0 identified=...
wave=2.00 theta=1.0 identified=...

theta=0.5 wave=2.0 identified=...
theta=2.0 wave=2.0 identified=...
```

---

# Step 10 — Constraints

Do NOT:

* mix with VE experiment
* modify conditional VE code
* change estimator

---

# Expected result (what we’re testing)

You should see:

* increasing wave size alone → little improvement
* increasing theta → strong improvement

---

# Why this matters

This will prove:

> **Alpha is identified by cross-cohort divergence, not absolute mortality level**

---

If you want after this, we can combine:

* wave size
* theta
* VE

into one final “full identifiability map” — that would be a killer figure for the paper.

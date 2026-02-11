Excellent — I’ve reviewed your latest `paper.md`, and below is an **exact Cursor-ready punch list** to address Grok’s Point #4 (external hazard shocks interacting with frailty), tailored to your current structure.

This is surgical. No restructuring. Just scope clarification and defensive tightening.

---

# 🎯 Objective

Neutralize:

> “COVID waves amplify mortality super-linearly in high-frailty individuals → residual bias even after normalization.”

We will:

1. Explicitly state epidemic waves are outside KCOR’s domain.
2. Strengthen the quiet-window assumption boundary.
3. Clarify that COVID-era motivation ≠ wave-period normalization.
4. Tighten the Limitations section accordingly.

---

# 📂 File to Edit

* `paper.md`

---

# 🔧 CURSOR PUNCH LIST

---

## STEP 1 — Strengthen §5.4 (Sensitivity to epidemic shocks)

### 📍 Find this section header:

```
### 5.4 Sensitivity to epidemic shocks
```

(If numbered slightly differently, search for “epidemic shocks” or the paragraph discussing COVID waves and super-linear amplification.)

### 📍 Locate the paragraph describing super-linear amplification of mortality during waves.

Immediately **after that paragraph**, insert:

```markdown
KCOR is not designed to normalize epidemic-wave periods characterized by abrupt, non-stationary hazard shocks that differentially impact high-frailty individuals. The quiet-window requirement is explicitly intended to exclude such intervals. Estimates obtained during pronounced wave periods should therefore be treated as non-identified under the KCOR framework rather than interpreted as corrected contrasts.
```

### 🎯 Effect

This converts the critique from:

> “Residual bias remains during waves.”

into:

> “Wave periods are outside the method’s identifiability domain.”

---

## STEP 2 — Tighten Quiet-Window Assumption Boundary

### 📍 Find:

```
### 2.1.3 Quiet-window stability
```

(or search for “quiet-window stability”)

At the **end of the paragraph defining the quiet-window assumption**, insert:

```markdown
Epidemic-wave periods with sharp hazard shocks fall outside this assumption and are excluded from normalization by design.
```

Short. Direct. Strong.

### 🎯 Effect

Explicitly narrows scope early in the paper.

---

## STEP 3 — Refine COVID-era Motivation in Introduction

### 📍 Search near the Introduction for paragraphs referencing COVID-era registry data as motivation.

Look for wording suggesting:

* KCOR “corrects pandemic-era bias”
* or broadly applies to pandemic periods.

Immediately after the sentence introducing COVID-era data as motivation, add:

```markdown
COVID-era registries provide a motivating example of selection-induced depletion geometry; however, KCOR normalization is applied only within diagnostically valid quiet intervals rather than during active epidemic waves.
```

### 🎯 Effect

Prevents reviewers from interpreting KCOR as a “wave correction” tool.

---

## STEP 4 — Add Explicit Limitation

### 📍 Find Section 5 (Limitations).

At the end of the Limitations section, add:

```markdown
Because epidemic-wave shocks interact with frailty in a non-stationary manner, KCOR does not attempt to correct mortality contrasts during such intervals; inference is restricted to diagnostically stable periods.
```

### 🎯 Effect

Closes the loop defensively and transparently.

---

# 🧠 What This Accomplishes

After these changes:

| Grok Concern                           | Manuscript Position                            |
| -------------------------------------- | ---------------------------------------------- |
| Wave interaction causes residual bias  | Yes — outside design scope                     |
| COVID context undermines normalization | No — normalization restricted to quiet windows |
| KCOR incomplete during pandemic waves  | Correct — by design                            |
| Overclaiming                           | Eliminated                                     |

---

# 🏁 Final Check After Editing

Verify:

* The phrase “outside this assumption” appears only once (avoid redundancy).
* You do not contradict this elsewhere.
* Figure S5 (quiet-window robustness) still reads naturally after these insertions.

---

Once this is in place, Grok’s Point #4 becomes a **scope clarification**, not a vulnerability.

If you’d like next, we can move to Grok’s next critique and continue tightening with the same precision.

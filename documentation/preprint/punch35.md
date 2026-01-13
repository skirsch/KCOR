fix everything
---

# Cursor Punchlist — Claude Review Cleanup

## 🔴 MUST FIX (High value, low risk)

These address clear redundancy or awkward structure that reviewers *will* notice.

---

### 1️⃣ Remove duplicate Box 1 content from §2.1.1

**Problem:** Box 1 (“Target estimand and scope (non-causal)”) is repeated almost verbatim in §2.1.1.

**Action:**

* Keep Box 1
* **Delete or heavily compress §2.1.1**, replacing it with a single sentence referring to Box 1

**Cursor instruction:**

> In `paper.md`, replace the body of §2.1.1 with a single sentence referring readers to Box 1 for scope and estimand definition. Remove duplicated non-causal language.

---

### 2️⃣ Consolidate §2.13 and §2.14 (Reproducibility)

**Problem:** Two back-to-back sections cover overlapping reproducibility content.

**Action:**

* Merge into **one section**
* Keep concrete details (build system, scripts)
* Remove repetition of philosophy

**Cursor instruction:**

> Merge §2.13 and §2.14 into a single “Reproducibility and computational implementation” section. Remove overlapping prose; retain concrete commands, environments, and scripts.

---

### 3️⃣ Cut or sharply compress the Methods Summary (early pages)

**Problem:** The Methods Summary duplicates the actual Methods section.

**Action (recommended):**

* Reduce to **~1 short paragraph** OR remove entirely

**Cursor instruction:**

> Compress the Methods Summary to a brief orientation paragraph (<150 words) or remove it entirely, avoiding duplication of the full Methods section.

---

### 4️⃣ Eliminate repeated “KCOR is not causal” statements

**Problem:** Appears in Box 1, §2.1, §4.2, §5 — too many times.

**Action:**

* State **once clearly** (Box 1 + one reminder in Limitations)
* Remove elsewhere

**Cursor instruction:**

> Remove repeated “KCOR is not a causal estimator” statements throughout the paper, retaining this clarification only in Box 1 and once in the Limitations section.

---

## 🟠 SHOULD FIX (Improves flow and reviewer comfort)

---

### 5️⃣ Reduce quiet-window repetition across sections

**Problem:** Quiet-window validity is explained in Methods, Diagnostics, Limitations, and SI.

**Action:**

* Methods: definition + role
* SI: diagnostics + tables
* Limitations: consequences of failure
* **Remove restatement elsewhere**

**Cursor instruction:**

> Consolidate quiet-window discussion: keep definition in Methods, diagnostics in SI tables, and failure implications in Limitations. Remove redundant explanations elsewhere.

---

### 6️⃣ Fix echo headings (heading repeats first sentence)

**Problem:** Seen in several places (Claude flagged multiple).

**Action:**

* Rewrite first sentence to add information, not restate heading

**Cursor instruction:**

> Scan for sections where the first sentence repeats the section title verbatim (e.g., “Data-generating process”, “KCOR assumptions”) and rewrite the opening sentence to add substantive content.

---

### 7️⃣ Vary sentence openings (“KCOR … KCOR … KCOR …”)

**Problem:** Stylistic monotony, especially in §2.1.

**Action:**

* Rewrite some sentences to start with:

  * “This framework…”
  * “The estimator…”
  * “Under this construction…”

**Cursor instruction:**

> In §2.1–§2.4, vary sentence openings to reduce repeated starts with “KCOR”.

---

## 🟡 OPTIONAL POLISH (Do only if you want extra smoothness)

---

### 8️⃣ Tighten contribution list (§1.5)

**Problem:** “First, Second, Third, Fourth” paragraphs are long.

**Action:**

* Shorten each to 1–2 sentences
* Focus on *what*, not *why*

**Cursor instruction:**

> Condense the contribution list in §1.5 so each item is ≤2 sentences, focusing on substantive contributions rather than explanation.

---

### 9️⃣ Improve abrupt transition in §1.2

**Problem:** COVID example enters too abruptly.

**Action:**

* Add one bridging sentence

**Cursor instruction:**

> Smooth the transition in §1.2 between general methodological problem and COVID illustration with a single bridging sentence.

---

### 🔟 Remove duplicated equations exposition

**Problem:** Gamma-frailty inversion explained multiple times.

**Action:**

* Full derivation once
* Elsewhere: reference equation number only

**Cursor instruction:**

> Remove repeated textual explanations of the gamma-frailty inversion; retain derivation once and reference equation numbers elsewhere.

---

# What NOT to change (important)

❌ Do **not**:

* Add theory or asymptotics
* Add MSM/IPW comparisons
* Change estimand definition
* Reframe as causal
* Move diagnostics back into prose

Those would **increase risk**, not reduce it.

---


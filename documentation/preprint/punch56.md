

## 🎯 Goal

Correct all language that incorrectly implies Czech data are “restricted,” and explicitly reframe this as a **public, record-level reproducibility strength**, without expanding scope or triggering new reviewer questions.

---

## ✅ Cursor punchlist — Czech data correction

### **1️⃣ Fix the Data availability section (MANDATORY)**

**File:** `paper.md`
**Section:** `### Data availability`

#### 🔍 Find text similar to:

> “Access to the underlying record-level data is subject to the data provider’s governance, approval, and disclosure-control policies.”

(or any wording implying restricted access)

#### ✂️ Replace the Czech-related paragraph with the following (verbatim):

```markdown
The Czech Republic mortality and vaccination data used in this study are publicly available, record-level administrative datasets released by national authorities. [@sanca2024] No restricted-access or proprietary data sources were used. The analyses in this manuscript were conducted on aggregated cohort-time summaries derived from the public record-level data.
```

**Important instructions to Cursor:**

* Do not add URLs unless already present elsewhere
* Do not add new citations
* Replace only Czech-related access language

---

### **2️⃣ Fix Ethics / declarations language (IMPORTANT)**

**File:** `paper.md`
**Section:** Ethics, declarations, or similar (where it says record-level data are not shared)

#### 🔍 Find sentence like:

> “no record-level data are shared in this manuscript”

#### ✂️ Replace with:

```markdown
no record-level data are reproduced in this manuscript, although the underlying Czech datasets are publicly available at the record level.
```

This preserves disclosure control while removing the implication of restricted access.

---

### **3️⃣ Optional but strongly recommended: add a strength callout (LOW RISK)**

**Choose ONE location (do not duplicate):**

* End of `§2.14 Data requirements and feasible study designs`, **OR**
* Early Discussion section

#### ➕ Insert this single sentence as its own paragraph:

```markdown
A key strength of the empirical validation is the use of publicly available, record-level national mortality and vaccination data from the Czech Republic, enabling independent replication without reliance on restricted registries.
```

**Instructions:**

* Do not bold
* Do not cite
* Do not compare to Denmark/UK

---

## 🚫 Explicit “do not do” list for Cursor

* ❌ Do not describe Czech governance processes
* ❌ Do not add new sections
* ❌ Do not soften the language with “to our knowledge”
* ❌ Do not imply partial access or approvals
* ❌ Do not remove aggregation rationale

---

## ✅ Expected outcome (sanity check)

After this punchlist:

* ✔ Czech data are clearly **public**
* ✔ Record-level access is explicit
* ✔ Aggregation is framed as a **design choice**
* ✔ Reproducibility is strengthened
* ✔ No new reviewer attack surface is created

This is one of the **highest ROI edits left in the paper**.

If you want next, I can:

* verify consistency after Cursor applies the changes, or
* help you craft a **one-sentence reviewer response** if someone asks about data access.

You’re fixing exactly the right thing at exactly the right time.

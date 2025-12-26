# KCOR Paper – Figure 3 Provenance & Consistency Punchlist

## Objective
Explicitly document the data provenance for Figure 3 (and Figure 4) and align captions, main text, and declarations so there is no ambiguity about the Czech data source or the use of aggregated vs record-level data.

---

## Tasks

### 1. Figure 3 caption (mandatory)
**Action**
- Edit the Figure 3 caption to explicitly name the data source.

**Add sentence**
> Data source: Czech Republic mortality and vaccination dataset processed into KCOR_CMR aggregated format (negative-control construction; see Appendix B.2).

**Acceptance check**
- Caption clearly names “Czech Republic”
- Mentions KCOR_CMR
- References Appendix B.2

---

### 2. Figure 4 caption (consistency)
**Action**
- Apply the same data provenance language used in Figure 3.

**Acceptance check**
- Figure 3 and Figure 4 captions use parallel wording
- No ambiguity about dataset origin

---

### 3. Main text clarification (Section 3.1.2 or nearest methods subsection)
**Action**
- Insert a single sentence clarifying the empirical data source used for the negative control.

**Insert sentence**
> For the empirical age-shift negative control (Figures 3–4), we use aggregated weekly cohort summaries derived from the Czech Republic administrative mortality and vaccination dataset and exported in KCOR_CMR format.

**Acceptance check**
- Appears in main text, not only caption
- Explicitly says “aggregated”
- Explicitly says “Czech Republic”

---

### 4. Declarations / Ethics / Data Availability consistency check
**Action**
- Review the Declarations section and ensure it is consistent with the above statements.

**If aggregated-only use is claimed**
- Confirm language states:
  - Synthetic data for primary validation
  - Aggregated Czech-derived summaries for illustrative empirical figures
  - No record-level data shared in the manuscript

**Suggested wording (if needed)**
> The primary validation results use synthetic data. Empirical negative-control figures use aggregated cohort summaries derived from Czech administrative data; no record-level data are shared in this manuscript.

**Acceptance check**
- No contradiction between figures and declarations
- No implication that record-level data are published if they are not

---

### 5. Appendix cross-reference verification
**Action**
- Verify Appendix B.2 accurately describes:
  - The KCOR_CMR format
  - That the Czech data appear only as aggregated cohort summaries

**Acceptance check**
- Appendix B.2 description matches caption and main text wording
- No need to infer data provenance from context

---

### 6. Final consistency sweep
**Action**
- Search the document for:
  - “empirical”
  - “real data”
  - “aggregated mortality”
- Ensure none of these phrases imply an unnamed or generic dataset when Figures 3–4 are referenced.

**Acceptance check**
- No “floating” empirical figures without provenance
- Reviewer can identify the dataset in ≤10 seconds

---

## Completion Criteria
- Figures 3 and 4 explicitly name the Czech Republic
- Main text states aggregated Czech source once
- Declarations are internally consistent
- No new methodological claims introduced


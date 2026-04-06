---
name: paper.md HVE clarification
overview: Deterministic edits to [documentation/preprint/paper.md](documentation/preprint/paper.md)—insert fixed text blocks for static HVE / depletion geometry scope, extend Box 2 (Interpretation bullet only), extend §4.1, and run a mandatory wording pass with explicit rewrite rules. Do **not** modify §4.2 (optional intuition omitted to avoid duplication).
todos:
  - id: sec-2-1-insert
    content: Insert §2.1 paragraph (INSERT TEXT A) after Non-causal estimand paragraph
    status: completed
  - id: sec-1-2-bullet
    content: Append §1.2 sentence (INSERT TEXT B) to Static HVE bullet
    status: completed
  - id: sec-4-1-insert
    content: Insert §4.1 paragraph (INSERT TEXT C) between stipulated anchors
    status: completed
  - id: box-2-interpretation
    content: Append Box 2 sentence (INSERT TEXT D) as second sentence on Interpretation bullet only
    status: completed
  - id: wording-pass
    content: Run forbidden-phrase search; rewrite any hit per §Enforcement (preserve negated disclaimers per rules)
    status: completed
isProject: false
---

# Reviewer-safe static HVE / depletion wording in paper.md (deterministic)

## Target file

[documentation/preprint/paper.md](documentation/preprint/paper.md) only. No math, figures, or section renumbering.

---

## INSERT TEXT — use verbatim (whitespace: single blank line between paragraphs as in manuscript)

### INSERT TEXT A — §2.1 new paragraph

Place **immediately after** the paragraph ending `not as evidence of causality.` and **immediately before** `This framework targets the second failure mode.`

```markdown
KCOR's static HVE adjustment targets baseline differences between cohorts insofar as they manifest through selection-induced depletion of susceptible individuals. This includes a broad class of baseline imbalances—whether due to health status, care-seeking, eligibility, or other enrollment-time selection factors—to the extent that they induce cohort-specific curvature in cumulative hazard trajectories under the working model. The resulting contrast therefore accounts for baseline differences that are explainable by depletion geometry, while remaining agnostic about mechanisms that do not act through that geometry.
```

*(Defensive wording: “accounts for” instead of “removes” in the last sentence to avoid implying full elimination of baseline imbalance.)*

### INSERT TEXT B — §1.2 append to Static HVE bullet

Append to the **same** `- **Static HVE:**` bullet, **immediately after** `frailty normalization.` **No new bullet.** The following string is the **exact** suffix to add (starts with one space after that period):

```text
 In practice, this adjustment is not limited to one named bias mechanism; it absorbs any enrollment-time baseline difference that enters the cohort comparison through the same depletion / frailty geometry.
```

### INSERT TEXT C — §4.1 new paragraph

Place **after** the paragraph beginning `KCOR addresses selection-induced depletion that distorts marginal comparisons but does not remove general confounding.` and **before** `KCOR does not uniquely identify the biological, behavioral, or clinical mechanisms...`

```markdown
At the same time, the static HVE correction is broader than a correction for one specific epidemiologic label. To the extent that baseline cohort differences operate through differential depletion of susceptible individuals, they are absorbed into the estimated depletion geometry and accounted for in the comparison. Residual differences therefore represent contrasts that are not explained by depletion alone, although they may still arise from time-varying confounding, external shocks, misspecification, or other non-depletive mechanisms.
```

*(Defensive wording: “accounted for in the comparison” instead of “removed from the comparison” to avoid implying full scrubbing of baseline effects.)*

### INSERT TEXT D — Box 2, Interpretation bullet

**Deterministic placement:** extend the **existing** `- **Interpretation**:` bullet only. **Do not** add a sibling bullet.

Append **after** the final sentence of that bullet (after `...causal harm or benefit.`). Insert a space, then:

```markdown
In particular, baseline differences that operate through depletion geometry are adjusted out; residual differences should be interpreted as what remains after that depletion-based normalization, not as automatically causal.
```

---

## §4.2 — do not edit

**Omit** the optional one-line intuition for §4.2 entirely. The same idea is covered by INSERT TEXT A and C; skipping §4.2 avoids redundancy and removes subjective “flow” judgment.

---

## Overclaiming and wording pass — mandatory

### Forbidden phrases (exact substring search, case-sensitive optional; at minimum search lowercase variants)

1. `neutralizes all baseline differences`
2. `removes every confounder`
3. `all confounding is removed`

### Enforcement — if a forbidden phrase appears in **affirmative** prose

Rewrite the **sentence** that contains it so the claim is qualified, using this template (adapt grammar to fit the sentence):

> `...adjusts for depletion geometry under the working model, without implying removal of all confounding.`

**Preserve without rewrite:**

- Any **negated** diagnostic disclaimer that mentions confounding removal (e.g. “not … proof that all confounding has been removed”) — these are already reviewer-safe.
- Product / estimand terms **depletion-neutralized** and **frailty-neutral** — do **not** rename or “simplify.”

### Secondary scan (report-only unless a problem appears)

- `neutralize` as a **verb** outside `depletion-neutralized` / `frailty-neutral` — if found with a global baseline claim, qualify with `under the working model` and/or `through depletion geometry` and/or `does not imply causal identification`.
- `remove bias` — acceptable in technical Cox/NPH contexts; only rewrite if it implies **all** confounding or **every** baseline difference is removed.
- Literal `corrects for` — if it implies universal confounding control, qualify per above; NPH “correction” on frailty-neutral scale is fine.

**Completion criterion:** Re-run forbidden-phrase search after edits; **zero** hits for the three forbidden strings except inside intentional negated disclaimers (if any edge case arises, prefer explicit qualification over deletion of the disclaimer).

---

## §4.3 Relationship to negative control methods

**No edit** to §4.3 (~917) unless a forbidden-phrase hit forces a local rewrite under **Enforcement** above. Do **not** add optional clarifying clauses by default.

---

## Post-edit check

1. Confirm INSERT TEXT A–D appear **exactly** once each, verbatim (em-dashes, math punctuation unchanged).
2. Confirm Box 2 still has a single **Interpretation** bullet (now with one additional sentence).
3. Re-run the three forbidden-phrase searches.

No build or code steps.

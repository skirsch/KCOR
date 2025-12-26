# KCOR paper polish: bridge COVID context + “injection” wording cleanup (Cursor punchlist)

Target file: `paper.md`

---

## A) Add a bridging paragraph to fix the “general method → COVID/non-COVID mortality” context jump

### Insert location
Insert **one new paragraph** in **§1 Introduction**, immediately **before** the heading:

`### 1.3 Evidence from the literature: residual confounding despite meticulous matching`

(So it lands after §1.2 and before §1.3.)

### Exact text to insert (paste verbatim)

Although this manuscript is motivated in part by mortality analyses conducted during the COVID-19 vaccination period, the methodological problem addressed here is general. The COVID setting provides unusually clear examples of selection-induced non-proportional hazards—because uptake was voluntary, rapidly time-varying, and correlated with baseline health—making residual confounding easy to diagnose using control outcomes such as non-COVID mortality. However, KCOR is not specific to COVID, vaccination, or infectious disease. The estimator applies to any retrospective cohort comparison in which selection induces differential depletion dynamics that violate proportional hazards assumptions.

### Acceptance check
- The first mention of COVID/non-COVID mortality now has a clean bridge that says:
  - COVID is illustrative
  - the problem is general
  - non-COVID mortality is used as a control outcome example

---

## B) Replace ambiguous “injection” language with “intervention”, while keeping “injected effect” for simulations

Goal: avoid “injection” sounding vaccine-specific, except where it literally means “injecting an effect into synthetic data” (positive controls).

### Step B1: Global search to review occurrences
Search for these tokens in `paper.md`:
- `injection`
- `inject`
- `injected`

### Step B2: Mandatory changes (real-world / general framing)

#### Rule 1
If the text is referring to the real-world concept (treatment/exposure), replace:
- `injection` → `intervention`
- `inject` → `apply` or `introduce` (choose the one that reads best)
- `injected` → `introduced`

Examples:
- “medical injections” → “medical interventions”
- “after rollout” is fine; no change needed

### Step B3: Keep simulation language (positive controls) but make it explicit

In §3.2 (“Positive controls”), you currently use “injecting a known effect”.
Keep the simulation meaning but ensure it’s explicitly “into data” (to avoid vaccine connotation).

#### Exact edits (apply verbatim)

1) In §3.2 first paragraph, change:

Positive controls are constructed by starting from a negative-control dataset and injecting a known effect into one cohort, for example by multiplying the *baseline* hazard by a constant factor $r$ over a prespecified interval:

TO:

Positive controls are constructed by starting from a negative-control dataset and **injecting a known effect into the data-generating process** for one cohort, for example by multiplying the *baseline* hazard by a constant factor $r$ over a prespecified interval:

2) Keep “Injection window” in tables/figures if it is clearly simulation-specific, but optionally rename to be more general:

Optional (recommended for generality):
- `Injection window` → `Effect window`

If you do this, apply consistently in:
- Table header in §3.2
- Figure captions that mention “injection window”
- Appendix B.3 bullets (“Injection window”)

### Step B4: “Never use injection for real-world intervention” final sweep
After edits, do a final search:
- If `injection` appears anywhere outside §3.2 / Appendix B.3 / positive-control figure captions/tables, change it to `intervention`.

---

## Done criteria
- There is a single bridge paragraph before §1.3.
- All real-world mentions use “intervention” (not “injection”).
- Simulation sections may still use “injecting an effect”, but explicitly “into the data-generating process” (or use “introduced effect”).
- No remaining ambiguous “injection” phrasing in general narrative text.


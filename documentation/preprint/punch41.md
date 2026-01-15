Here’s a single **Cursor punchlist** that bundles **all fixes I recommended above** (S4 table + S4.6 clarity + S4 overview tightening + optional S5 signpost). All changes are confined to `supplement.md` and are safe.

Source context: your current `supplement.md` already has the master table `@tbl:si_sim_params`, but it claims to cover S4.6 while having no S4.6 rows, and S4.6’s heading includes “(S7)”. 

---

## Cursor punchlist: Supplement cleanups (S4 + S4.6 + light S5 signpost)

### 1) Fix S4.6 heading ambiguity (mandatory)

**File:** `supplement.md`
**Find:**

```markdown
### S4.6 Joint frailty and treatment-effect simulation (S7)
```

**Replace with:**

```markdown
### S4.6 Joint frailty and treatment-effect simulation (joint stress test)
```

(Or omit “(joint stress test)” if you want it even cleaner. The key is: remove “(S7)”.)

---

### 2) Make the S4 master parameter table truthful by adding S4.6 rows (mandatory)

**File:** `supplement.md`
**Location:** Table `@tbl:si_sim_params` under `### S4.0 Summary tables...`

**Add the following rows at the end of the table (after the last S4.5 row):**

```markdown
| S4.6 | Joint frailty + effect | Generation script | `test/sim_grid/code/generate_joint_frailty_effect_sim.py` | If the actual script differs, replace with the correct path |
| S4.6 | Joint frailty + effect | Frailty distribution | Gamma, mean 1 | Cohort-specific variance |
| S4.6 | Joint frailty + effect | Baseline hazard | Same for both cohorts | Specify constant vs time-varying if applicable |
| S4.6 | Joint frailty + effect | Frailty variance | θ_control and θ_treated (distinct) | Uses unequal θ to induce differential depletion |
| S4.6 | Joint frailty + effect | Effect multiplier | r = 1.2 (harm); r = 0.8 (benefit) | Applied to treated cohort only |
| S4.6 | Joint frailty + effect | Quiet window | Prespecified; same as other simulations | Used for frailty estimation |
| S4.6 | Joint frailty + effect | Effect window | Prespecified; non-overlapping with quiet window | Ensures separation of mechanisms |
| S4.6 | Joint frailty + effect | Random seed | 42 | If different, replace |
```

Notes:

* If you don’t have a separate script path for S4.6, delete the “Generation script” row or point it to the real one (even if it reuses an existing generator).
* If baseline hazard / windows / seed are identical to S4.3, you can keep them as “same as S4.3” in the Notes column instead of repeating numbers — but the table needs **at least some explicit S4.6 entries** so it doesn’t falsely claim coverage.

---

### 3) Add an explicit cross-link sentence inside S4.6 (mandatory)

**File:** `supplement.md`
**Section:** `### S4.6 Joint frailty and treatment-effect simulation ...`

Immediately after the first paragraph (the one beginning “This simulation evaluates KCOR under conditions…”), insert:

```markdown
This joint simulation combines the selection-induced depletion mechanisms examined in Sections S4.2 and S4.5 with the injected-effect framework of Section S4.3.

Parameter values and scripts for this joint simulation are summarized in Table @tbl:si_sim_params.
```

This resolves reader confusion about what S4.6 is and makes it match the pattern used in S4.2–S4.5. 

---

### 4) Remove duplicated intro sentences in S4.3, S4.4, S4.5 (recommended)

Right now each section has both:

* the improved “overview” paragraph, and
* a second “intro sentence” doing the same job (left over from earlier edits).

#### 4a) S4.3

**File:** `supplement.md`
**In section `### S4.3 Positive control: injected effect`**, delete the redundant paragraph that begins:

> “The positive control (Figure @fig:pos_control_injected; Table @tbl:pos_control_summary) starts…”

Keep:

* the overview paragraph,
* the table pointer sentence,
* and the single sentence about multiplying baseline hazard.

#### 4b) S4.4

**In section `### S4.4 Sensitivity analysis parameters`**, delete the redundant sentence that begins:

> “The sensitivity analysis (Figure @fig:sensitivity_overview) evaluates robustness…”

Keep:

* the overview paragraph,
* the table pointer sentence,
* “Output grids show…”

#### 4c) S4.5

**In section `### S4.5 Tail-sampling / bimodal selection …`**, delete the redundant second opening sentence that begins:

> “This adversarial simulation constructs two cohorts…”

Keep:

* the first overview paragraph,
* the cohort definitions bullets,
* the table pointer sentence,
* and the evaluation paragraph.

This removes repetition and makes the S4 narrative much smoother. 

---

### 5) Optional: Add a one-sentence signpost at top of S5 (optional but helpful)

**File:** `supplement.md`
**Location:** Immediately after `## S5. Additional figures and diagnostics`

Insert:

```markdown
This section provides diagnostic outputs and evaluation criteria for the simulations and control-test specifications defined in Section S4.
```

This reinforces your S4/S5 separation without moving any content.

---

## Verification checklist (do this after edits)

1. Search for `(S7)` → should be **0 occurrences** in headings.
2. In Table `@tbl:si_sim_params`, confirm there are **rows with Section = S4.6**.
3. In S4.6 text, confirm it contains:

   * the cross-link sentence to S4.2/S4.5/S4.3
   * the pointer sentence to `@tbl:si_sim_params`
4. Skim S4.3–S4.5 openings: ensure no duplicated “overview + overview” paragraphs remain.

---
Can you also fix this part too to use the actual names and parameters  since you have access?

If you paste the section of your repo that contains the **actual S4.6 generator script name / parameters**, I can tighten those S4.6 table rows to be exact rather than placeholders—but the structural fixes above should be applied now regardless.

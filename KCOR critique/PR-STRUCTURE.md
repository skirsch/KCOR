# Forthcoming KCOR revision PR: proposed structure

**Lead-author review package — 25 November 2026.** Prepared on an isolated review branch; proposed method changes remain experimental. The lead author should first review the scientific findings and candidate choices. The current local revised sources and both manuscript PDFs remain unchanged. This is a working packaging document, not the lead-author scientific presentation.

## 1. What remains from the objections

All 19 numbered objection records across the three review series (six in series 1, ten in series 2, three in series 3) have been examined. The table below distinguishes completed corrections from remaining limitations and adoption decisions. It does not assert that the latest edition has passed another independent review.

| ID | Remaining substantive issue | Current position |
|---|---|---|
| [1.2](objections/1.2/README.md) | Exact in-wave nonlinear exposure and survivor-law consistency | Corrections to the existing equation are implemented. Exact weekly inversion and consistent survivor fitting have been tested separately. Approval is needed for adoption and the weekly deficit convention. |
| [1.3](objections/1.3/README.md) | Fitter stability and uncertainty calibration | Person-bootstrap and MC fixes are implemented. All-window common-k/theta candidate and matched coverage work are complete; decision on adoption and validation of the final integrated pipeline remain. Coverage is not established universally. |
| [1.4](objections/1.4/README.md) | Interpretation/reporting of empirical controls and alpha | Table 8 replay complete. Reporting choice depends on final fitter and flags. Historical pooled alpha replay and shape-information study fail identification; retain 1.18 as sensitivity-only and defer further alpha investigation. |
| [2.7](objections/2.7/README.md) | Bias with low information at zero added hazard | Unsupported threshold claim corrected. Mechanism and cohort-size experiment complete. Discuss adding the 5M-per-cohort companion, retaining the intended original stress experiment, and incorporating theoretical information limits. Finite-sample bias itself is not eliminated. |
| [3.2](objections/3.2/README.md) | Pre-entry mortality and mixed quiet/wave periods | Proportional entry correction and exact nonlinear candidate tested, not adopted. Lead-author decision on entry correction, labeled wave eligibility and any fitting changes. Unlabeled mixed-wave exposure remains a limitation. |

The other 14 IDs (1.1, 1.5, 1.6, 2.1–2.6, 2.8–2.10, 3.1, 3.3) have their identified corrections or disclosures incorporated in the revised edition. The 2.5 window-scan figure still merits a scientific discussion, but not another correction to the scan. The author's analytical Cox proposal is separate from objection resolution. See [full disposition](<iterative review/open-objections.md>), [decision log](<iterative review/decisions.md>) and [internal work notes](WORKING-NOTES.md).

## 2. Reader-facing hierarchy

The lead-author presentation is the entry point. It synthesizes the problems, common root causes, proposed solutions, demonstrated effects and remaining limitations. Detailed scientific evidence is the next layer; PR packaging is subordinate to that review.

| Level | Document | Purpose |
|---|---|---|
| First | [PR-PRESENTATION.md](PR-PRESENTATION.md) and [simplified briefing](PR-BRIEFING.md) | Lead-author overview, organized by shared causes and demonstrated solutions, from major scientific findings to smaller corrections. |
| Second | [PR-SCIENTIFIC-EVIDENCE.md](PR-SCIENTIFIC-EVIDENCE.md) | Full matched numerical tables, uncertainty intervals, derivations, and evidence supporting each conclusion. |
| Third | [Clean manuscript](<iterative review/KCOR-clean.pdf>), [redline](<iterative review/KCOR-redline.pdf>), and [objection index](README.md) | Actual paper changes and detailed reproducibility records. |

The eventual PR description is a short navigation aid pointing straight to the lead-author presentation, not a competing top-level briefing.

The [working notes](WORKING-NOTES.md), [review procedure and archives](<iterative review/evaluations/README.md>), [original-intent audit](<iterative review/original-intent-audit.md>) and checkpoints stay separate from the scientific presentation. Keep the x.y identifiers stable.

## 3. Proposed PR change groups (reviewable commits)

Order commits so that mechanical corrections and already-approved changes can be reviewed separately from substantial new methodology. These are proposed groupings, not commits created now. Each group's eventual implementation must include corresponding manuscript and test changes where applicable; do not mix in unapproved experimental code.

| Group | Scope | Evidence / dependency |
|---|---|---|
| A | Preserve baseline and revision selection; add concise release/selection instructions and objection index | [Version workspace](<iterative review/README.md>); no destructive folder cleanup. |
| B | Correct weekly integration, gamma inversion, gap/entry indexing, and diagnostics; retain the intended contamination experiment | 1.6, 2.2–2.4, 2.8, 3.1; distinguish approved entry-index correction from candidate 3.2. |
| C | Correct person resampling, MC processing, subgroup pooling and test provenance | 1.3, 2.5, 2.6, 2.9, 3.3; include tests and reproduction of Table 8/Table 9. |
| D | Adopt all-window common-k/theta joint fitter, if approved | [Stress and coverage results](objections/1.3/all-window-coverage/README.md); final default and integration to be decided. |
| E | Exact nonlinear survival and pre-entry accounting, only if approved | [Consistent survivor fit](objections/3.2/nonlinear-survivor-fit/README.md), [weekly inversion](objections/1.2/all-window-weekly-survival/README.md), [entry-aware study](objections/3.2/all-window-entry/README.md). |
| F | Scientific additions and final validation tables/figures if approved | [Information limits](objections/2.7/low-frailty/theoretical-limits/README.md), [contamination](objections/2.7/all-window-contamination/README.md), [alpha replay](objections/1.4/alpha-pooled-replay/README.md). Alpha remains sensitivity-only; no experimental estimator adopted. |
| G | Remaining manuscript clarifications, generated clean/redline PDFs, final evidence and presentation | [Revision records](revision-records.md), scientific presentation, original review identifiers. |

Where groups overlap in a shared file (especially KCOR.py), create reviewable changes from the selected final edition instead of copying full experimental files into every group. The final PR may combine small groups if that makes reviewing the actual diff easier.

## 4. Source and artifact map

The review branch is based on 82f546cd, the PR #5 merge, verified against the upstream main head on 25 November 2026. The root working tree has existing unrelated changes, generated artifacts, and the critique-folder reorganization. Do not stage everything or replace it indiscriminately.

- Main method and manuscript: the [current revised edition](<iterative review/revised/>) contains code/KCOR.py, paper and supplement sources, four revised figures, and a regression test; all nine manifest entries verified by SHA-256 on 25 November. Only after the lead-author decisions, map the chosen sources to the normal root paths (code/, documentation/preprint/, test/) and resolve differences explicitly.
- Preserve the [baseline](<iterative review/baseline/>) and [iteration 1](<iterative review/checkpoints/iteration-1/>) and [iteration 2](<iterative review/checkpoints/iteration-2/>) checkpoints so the user can choose an edition without a Git rollback. Do not silently replace past checkpoints. Save the final approved edition separately.
- Include both clean and cumulative redline PDFs of the final approved edition, with objection links in the redline. The currently available PDF pair represents the current implemented edition, not all experimental studies.
- Include the numbered evidence, reproducible scripts, concise machine-readable numerical results and relevant figures. Keep candidate-only trials clearly separate. Audit generated outputs, repeated bundles, temporary files, Python caches, large files and registry-derived data before packaging; preserve local archives and reproducibility. Do not add raw registry data or intermediate work products indiscriminately.
- Compare the actual final source diff and the two PDFs against the manuscript claims, verification records, and decisions. Regenerate affected results, including uncertainty intervals, using the approved pipeline. Use the existing manifest and test records as evidence, not as a claim of completed final integration.

## 5. Proposed PR description

**Proposed title:** Resolve KCOR review objections and strengthen mathematical and numerical validation

### Summary

This revision builds on merged PR #5. It addresses the mathematical, software, statistical and presentation issues raised in the original independent review and subsequent review of the revised method, preserving the original KCOR objective and experiment intent.

### Major findings

- An all-window joint fit of the common baseline scale and frailty variance substantially reduces numerical instability in the low-information stress experiment. The earlier joint fitter had a median 95% interval width of 213.056 versus 0.315 with the all-window candidate. The main candidate results and remaining precision and coverage limitations are presented separately.
- Model-based information limits explain why low frailty variance can be difficult to estimate, and why precise KCOR endpoints and precise frailty estimates are not the same requirement. The zero-addition contamination mean approaches the known target in the higher-information experiment.
- Exact nonlinear survivor-law modeling addresses a deterministic inconsistency in the previous wave approximation. Known-truth tests support the candidate; fitted-data coverage and mixed-wave limitations are explicitly reported.
- Replaying the original pooled Czech alpha objective with the improved machinery does not identify alpha. The forced exponent of 1.18 remains sensitivity-only.

### Implemented corrections

Weekly integration and inversion, gap and wave boundary handling, nonpositive gap diagnostics, person-bootstrap and MC processing, validation-study provenance, and corresponding manuscript clarifications. See the numbered evidence and redline for specific changes. This paragraph will reflect the final approved source diff; experimental fitter and nonlinear changes are not included by implication.

### Review guide

Start with the [scientific presentation](PR-PRESENTATION.md), followed by the [clean paper](<iterative review/KCOR-clean.pdf>) and [linked redline](<iterative review/KCOR-redline.pdf>). The [numbered critique index](README.md) links each objection to its evidence and correction.

### Validation and limitations

Matched known-truth simulations, person-bootstrap intervals, full-workbook checks, empirical controls, theoretical precision diagnostics, and original-objective reproduction are documented. The studies support specific tested settings, not universal coverage. The final approved defaults, regenerated tables, and residual limitations will be stated explicitly before submission.

## 6. Preparation status

- Complete: all numbered objections examined, scientific presentation and evidence index maintained, source snapshot manifest verified, initial PR structure drafted.
- Awaiting lead author: adoption choices in section 1, scientific additions and final reporting.
- After approval: assemble only approved source changes into the review branch, regenerate the clean/redline PDFs and affected validation results, check the actual diff and finalize the PR description.
- No more LLM review iterations until the lead author sees the findings and the user explicitly decides to resume. This branch is for lead-author review, not adoption of the experimental fitter.

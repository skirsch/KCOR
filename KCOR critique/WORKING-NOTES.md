# KCOR revision: internal working notes

**Internal research and review coordination | 25 November 2026**

This document records the work process, iteration history, next investigations and informal author discussion notes. The [lead-author PR presentation](PR-PRESENTATION.md) is separate and contains the findings, changes and their scientific evidence. Keep both up to date as work progresses.

## 1. Review scope and version preservation

The review aims to find and fix mistakes and substantive objections while preserving the original intent of KCOR. Optional methodology changes are tested separately rather than silently adopted. Original, revised and checkpoint editions remain available without Git rollback.

- [Original PR #5 evidence and six objections](PR5-evidence.md)
- [Numbered objection index](README.md)
- [Clean revised manuscript](<iterative review/KCOR-clean.pdf>)
- [Cumulative redlined manuscript](<iterative review/KCOR-redline.pdf>)
- [Original-intent audit](<iterative review/original-intent-audit.md>)
- [Accepted changes and decisions](<iterative review/decisions.md>)
- [Open objections](<iterative review/open-objections.md>)
- [Review protocol and archived evaluations](<iterative review/evaluations/README.md>)

The independent evaluation procedure includes blind reviews of revised and unrevised editions, same-context reassessment after the changes are supplied, fresh-context handoffs with the objections and changes, and comparisons between revision layers. Findings are tracked by stable objection identifiers x.y. No further LLM evaluation iteration will be started before the lead author reviews this work and explicitly decides how to proceed.

## 2. Remaining work and decisions before adoption

| Item | Status / discussion |
|---|---|
| 1.3 — production uncertainty coverage | Main matched validation complete. Results support the fitter in tested settings; nonlinear-wave coverage is marginal, and the final approved method needs revalidation. |
| 2.7 — finite-sample bias | Mechanism substantially explained; decide whether to add the higher-information companion and retain the original stress case. |
| 3.2 — mixed periods preceding the initial fit | Matched [entry-aware study](objections/3.2/all-window-entry/README.md) complete. Proportional entry correction candidate for approval; early fitting-anchor contamination diagnosed. Quiet-only fit is a sensitivity, not adopted. [Exact nonlinear survivor fit](objections/3.2/nonlinear-survivor-fit/README.md) tested through full matched coverage and Table 8. Known-truth recovery and main mean improved; coverage unchanged. Marked-wave eligibility, deficit handling and proportional-entry routing require lead-author decisions. |
| 1.2 — nonlinear in-wave inversion | [Combined trial](objections/1.2/all-window-weekly-survival/README.md) complete. Coverage unchanged; exact known-input recovery; remaining bias traced to fitted frailty. Exact inversion combined with the [consistent survivor fit](objections/3.2/nonlinear-survivor-fit/README.md) now tested. Lead-author approval of both, marked wave rows and deficit convention pending. |
| 1.4 - Table 8 reporting | [Reporting discussion](objections/1.4/reporting-review.md), [temporal information](objections/1.4/alpha-information/README.md), and [historical pooled replay](objections/1.4/alpha-pooled-replay/README.md): original illustrative minima reproduced; improved fit moves minima to a boundary, with no identification. No external scale supplied. Decide sensitivity label and final reporting with lead author. |
| 1.3 — fitter default | Lead-author approval of common-k all-window joint fit, profiled gap offsets and optimizer starts. |
| 2.5 — quiet-window pattern | Scientific discussion of the saved figure, independent of the scan correction. |
| Additional contributions | Decide where to include the information-limit derivation and the author's Cox proposition. |

After the lead author has reviewed the evidence and approved the relevant changes, regenerate the approved method and paper with clean and cumulative redline PDFs, preserve the new checkpoint, and reproduce affected empirical and synthetic tables. No further LLM review cycle before that review and a new decision. Update the lead-author presentation with the final results and exact accepted method before finalizing the method for merge.

**Evidence index:** [Root causes](<iterative review/root-causes.md>) · [Decision log](<iterative review/decisions.md>) · [Revision records](revision-records.md) · [Open objections](<iterative review/open-objections.md>).

## 3. Deferred scientific discussion and author proposals

### Quiet-window sensitivity figure

The corrected quiet-window scan used all 78 attempts, and all 78 passed the revised diagnostics. Yet fitted variances varied substantially between overlapping placements in every birth-decade/dose group. The figure is worth discussing independently of the corrected denominator and scan fitter. The shared temporal pattern could reflect model adequacy, mortality patterns, window composition or other effects; its interpretation remains open.

![Quiet-window frailty scan, Czech 2021-24](<iterative review/revised/documentation/preprint/figures/fig_si_quiet_window_theta_scan_czech_2021_24.png>)

[Scan data and verification](objections/2.5/verification.md)

### Author's analytical Cox proposition

For equal individual baseline rates, cohort A with no frailty and cohort B with mean-one gamma frailty variance theta, the marginal hazard ratio B/A is

$$
R(t)=\frac{1}{1+\theta H_0(t)}.
$$

Ordinary Cox partial likelihood estimates a single risk-set-weighted ratio between R(T) and 1 under the stated two-cohort design. This gives an analytical companion to the simulation. It does not turn the known-frailty simulation into validation of fitted-frailty KCOR.

[Author proposal and derivation](<iterative review/author-proposals/objection-5-cox-gamma-derivation.md>)

## 4. Detailed working records

- [Objection register](<iterative review/open-objections.md>)
- [Root-cause investigations](<iterative review/root-causes.md>)
- [Decision log](<iterative review/decisions.md>)
- [Revision records](revision-records.md)
- [Evaluation protocol and archive](<iterative review/evaluations/README.md>)
- [Original-intent audit](<iterative review/original-intent-audit.md>)

## 5. PR preparation

[Proposed PR structure and draft description](PR-STRUCTURE.md) records the final objection disposition, proposed commit groups, artifact map and preparation status. Further alpha investigation is deferred. The review package is prepared on a separate branch for lead-author consideration.

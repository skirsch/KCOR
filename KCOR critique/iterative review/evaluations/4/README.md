# Series 4 — hypothetical integrated PR #6 edition

**25 November 2026.** In this round the proposed changes in draft PR #6 (commit 68d65104) are treated as approved and integrated in a separate scientific edition. This is an evaluation assumption: the real PR, root production code, and preserved revised edition remain unchanged.

## Editions and paths

- Original: frozen m17, exactly the original baseline of series 3.
- Previous: frozen m83, the last evaluated edition of series 3.
- Checkpoint for attribution: retained m61.
- Current: frozen m94, the post-approval candidate. The paper and supplement describe the all-window common-k/theta fit, pre-entry proportional exposure, consistent nonlinear survivor-law fitting and weekly inversion, the proposed deficit convention, information limits, higher-information contamination example, and sensitivity-only alpha interpretation. Scientific result files accompany the source. The integrated method is not merged.

All nine starting review conversations are independent. Each of three repetitions has current blind, original same-context and fresh handoff, previous same-context and fresh handoff, and original → previous → current chain. Open question: "Please examine KCOR closely. What is wrong, if anything, and how would you fix it?" GPT-6.2 Astra, high reasoning. Targeted follow-ups are distinct from open reviews.

## Verification before review

- The preserved earlier m17/m42/m61/m83 inputs were verified against series 3 manifests.
- The integrated Table 8 workbook reproduced all six latest survivor-model point estimates exactly and the first matched person-bootstrap draw. The full previously computed 999-draw array and intervals are retained; no claim of a new full 999-draw rerun.
- Table 9 was rerun through the ordinary workbook and all 999 draws: benefit 0.83359 [0.77412,0.89705], harm 1.16170 [1.08652,1.24845].
- All 78 quiet-window scan placements passed the new fit diagnostics; the figure and reported ranges were regenerated.
- A matching combined 65-page PDF was rendered from the current paper and supplement (generic Pandoc/XeLaTeX render; cross-reference formatting differs from the publication build).
- [Input hashes](inputs.json), [sealed candidate overlay](edition-overlay.zip), [original-to-current diff](original-to-current.diff), [previous-to-current diff](evaluated-to-current.diff), and [checkpoint-to-current diff](checkpoint-to-current.diff) preserve what the reviewers received. Large numerical datasets and figures are represented by hashes in diffs and are present in the scientific edition.

## Status

**Completed:** 18 independent conversations, 30 open turns (18 assessing the current candidate), 18 targeted follow-ups; all 48 response hashes, thread identities and verbatim handoffs passed the audit. All 18 current open answers requested major revision while recognizing substantial improvements. Initial parallel handoffs encountered a local temporary-file sharing collision; retries used unique filenames and preserved the original starting answers and conversation identities. See [convergence](convergence.md), [path counts](path-counts.md), [verification](verification.md) and [new issues](<../../../objections/README.md#series-4--hypothetical-integrated-pr-6>).

# Series 3 · Multi-edition review of iteration 2

**November 23, 2026.** [Convergence and objection dispositions](convergence.md) · [Spontaneous findings by path](path-counts.md) · [Independent numerical checks](checks.md) · [All review answers](#answers-by-repetition).

The open question at every stage is “Please examine KCOR closely. What is wrong, if anything, and how would you fix it?” All reviewers use GPT-6.2 Astra, high reasoning, with three repetitions per path. The experiment evaluates the latest paper and method. Each sign-off on an objection also requires checking its underlying evidence.

| Edition | Frozen material |
|---|---|
| m17 · original | Original edition from series 2, before the paper and method revisions |
| m42 · previously evaluated | Exact revised repository seen by the series 2 reviewers |
| m61 · iteration 1 checkpoint | Author checkpoint including N1, with its own paper, supplement, code and clean PDF |
| m83 · current | Iteration 2 paper, supplement, method, test and clean PDF, with the updated Table 8 replay, quiet-window scan, and Cox error-bar analysis |

The [input manifest](inputs.json) lists the hash of every file in every edition. Series 3 uses the common Git tree and [series 2 overlays](../2/edition-overlays.zip), plus its own [sealed overlays](edition-overlays.zip). To reconstruct all four in an empty directory, run `py -3.11 -B freeze.py <directory>` here. The primary scientific source files are byte-for-byte copies. Bundled analysis runners locate their inputs relative to the blinded checkout and select the fitter used to produce the reported result. The working review notes, redline, Git history and objections are excluded from blind inputs. Exploratory, unincorporated work on low-frailty recovery is outside the scientific edition under review.

| Path in each repetition | Inputs and order |
|---|---|
| Blind | Current repo alone |
| Previous, same context | Previously evaluated repo → current repo and [incremental diff](evaluated-to-current.diff) |
| Previous, fresh handoff | Exact previous starting answer → new context with current repo and identical incremental diff |
| Original, same context | Original repo → current repo and [cumulative diff](original-to-current.diff) |
| Original, fresh handoff | Exact original starting answer → new context with current repo and identical cumulative diff |
| Continuing chain | Original repo → previously evaluated repo and [first diff](../2/changes.diff) → current repo and incremental diff; same reviewer throughout |
| Checkpoint attribution | [Checkpoint-to-current diff](checkpoint-to-current.diff), without a separate reviewer arm |

The three continuing chains start independently. This keeps them free of targeted follow-ups and retrospective summaries from the earlier round. The two previous-edition arms also receive the corresponding archived series 2 blind critique in a targeted follow-up after their open answers. Convergence counts must use the saved open answers; prompted checks are reported separately. A non-mention does not count as withdrawal.

For numerical adjudication, [check.py](check.py) replays the original Table 8 points, the two fit sensitivities and the archived validation counts. The [first-window](first-window.md), [gap](gap-check.md), [Table 9](positive-check.md) and [quiet-window contrast](window-contrast.md) notes give the separate proofs or replays. The input manifest covers 1,119 original, 1,128 previous, 1,128 checkpoint and 1,134 current files.

## Answers by repetition

Completed 18 open assessments of the current edition, 12 open reviews of earlier editions and 18 targeted follow-ups. The [archive audit](audit.py) verified all 48 response hashes, all 18 conversation identities and the six verbatim starting answers used for handoff. All four frozen scientific editions passed hash verification. Seventeen current-edition open answers explicitly recommended major revision; the remaining answer identified material limitations without assigning a publication verdict.

Each link opens the saved answer. Matching JSON files contain the exact prompts, model settings, conversation identifiers and response hashes. The direct original same-context reviewers used isolated native reviewer sessions; the other paths used the local read-only CLI. The original starting answer is identical in each matched pair. Reviewers in the continuing chains first assessed the original in separate sessions.

| Path | Repetition 1 | Repetition 2 | Repetition 3 |
|---|---|---|---|
| Current, blind | [open](repeat-1/blind.md) ? [targeted](repeat-1/blind-targeted.md) | [open](repeat-2/blind.md) ? [targeted](repeat-2/blind-targeted.md) | [open](repeat-3/blind.md) ? [targeted](repeat-3/blind-targeted.md) |
| Original, same context | [first](repeat-1/original-first.md) ? [current](repeat-1/original-same.md) ? [follow-up](repeat-1/original-same-follow.md) | [first](repeat-2/original-first.md) ? [current](repeat-2/original-same.md) ? [follow-up](repeat-2/original-same-follow.md) | [first](repeat-3/original-first.md) ? [current](repeat-3/original-same.md) ? [follow-up](repeat-3/original-same-follow.md) |
| Original, fresh handoff | [current](repeat-1/original-handoff.md) ? [follow-up](repeat-1/original-handoff-follow.md) | [current](repeat-2/original-handoff.md) ? [follow-up](repeat-2/original-handoff-follow.md) | [current](repeat-3/original-handoff.md) ? [follow-up](repeat-3/original-handoff-follow.md) |
| Previously evaluated, same context | [first](repeat-1/evaluated-first.md) ? [current](repeat-1/evaluated-same.md) ? [follow-up](repeat-1/evaluated-same-follow.md) | [first](repeat-2/evaluated-first.md) ? [current](repeat-2/evaluated-same.md) ? [follow-up](repeat-2/evaluated-same-follow.md) | [first](repeat-3/evaluated-first.md) ? [current](repeat-3/evaluated-same.md) ? [follow-up](repeat-3/evaluated-same-follow.md) |
| Previously evaluated, fresh handoff | [current](repeat-1/evaluated-handoff.md) ? [follow-up](repeat-1/evaluated-handoff-follow.md) | [current](repeat-2/evaluated-handoff.md) ? [follow-up](repeat-2/evaluated-handoff-follow.md) | [current](repeat-3/evaluated-handoff.md) ? [follow-up](repeat-3/evaluated-handoff-follow.md) |
| Continuing chain | [original](repeat-1/chain-original.md) ? [previous](repeat-1/chain-evaluated.md) ? [current](repeat-1/chain-current.md) ? [follow-up](repeat-1/chain-follow.md) | [original](repeat-2/chain-original.md) ? [previous](repeat-2/chain-evaluated.md) ? [current](repeat-2/chain-current.md) ? [follow-up](repeat-2/chain-follow.md) | [original](repeat-3/chain-original.md) ? [previous](repeat-3/chain-evaluated.md) ? [current](repeat-3/chain-current.md) ? [follow-up](repeat-3/chain-follow.md) |
